import csv
import os
import time
from typing import Tuple

import jax
import gym
import jax.numpy as jnp
import numpy as np
import pandas as pd
from absl import app, flags
from ml_collections import config_flags
from models import COMFO3Agent, RIQLAgent

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

COMFO_STEP = 400


def get_trj(agent, env: gym.Env, obs, eval_episodes: int = 1) -> Tuple[float, float]:
    t1 = time.time()
    trj = []
    avg_reward = 0.
    for _ in range(eval_episodes):
        done = False
        # obs, done = env.reset(), False
        trj.append((obs, None, None, done))
        for _ in range(1000-COMFO_STEP):
            if done:
                break
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            trj.append((obs, action, reward, done))
    avg_reward /= eval_episodes
    return trj, time.time() - t1

# def analysis_trj(trj, RIQL_agent, COMFO_agent):
#     lenth = len(trj)
#     print(f"lenth:{lenth}")
#     q1_list = []
#     q2_list = []
#     diff_list = []
#     reward_list = []
#     for i in range(2, lenth-1):
#         obs, action, reward, done = trj[i]
#         q11, q12 = RIQL_agent.critic.apply(
#             {"params": RIQL_agent.critic_target_params}, obs, action)
#         # print(q11, q12)
#         q1 = jnp.minimum(q11, q12)
#         q21, q22 = COMFO_agent.critic.apply(
#             {"params": COMFO_agent.critic_target_params}, obs, action)
#         q2 = jnp.minimum(q21, q22)
#         diff = q1-q2
#         q1_list.append(q1)
#         q2_list.append(q2)
#         diff_list.append(diff)
#         reward_list.append(reward)

#         # print(f"{i}:{q1:.2f},\t{q2:.2f},\t{diff:.2f}")
#     print(len(q1_list),len(q2_list),len(diff_list),len(reward_list))
#     # print(reward_list)
#     data = jnp.array([q1_list, q2_list, diff_list, reward_list])
#     return data


def analysis_trj(trj, agent):
    lenth = len(trj)
    print(f"lenth:{lenth}")
    q1_list = []
    reward_list = []
    for i in range(1, lenth-1):
        obs, action, reward, done = trj[i]

        q11, q12 = agent.critic.apply(
            {"params": agent.critic_state.params}, obs, action)
        # print(q11, q12)
        q1 = jnp.minimum(q11, q12)
        q1_list.append(q1)
        reward_list.append(reward*20)
        # print(f"{i}:{q1:.2f},\t{q2:.2f},\t{diff:.2f}")
    # print(len(q1_list))
    # print(reward_list)
    data = jnp.array([q1_list, reward_list])
    return data


def main(argv):
    config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
    FLAGS = flags.FLAGS
    configs = FLAGS.config
    configs.seed = 1

    env = gym.make("Hopper-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    RIQL_agent = RIQLAgent(obs_dim=obs_dim,
                           act_dim=act_dim,
                           max_action=max_action,
                           hidden_dims=configs.hidden_dims,
                           seed=configs.seed,
                           lr=configs.lr,
                           tau=configs.tau,
                           gamma=configs.gamma,
                           expectile=configs.expectile,
                           temperature=configs.temperature,
                           max_timesteps=configs.max_timesteps,
                           mle_alpha=configs.mle_alpha,
                           initializer=configs.initializer)
    RIQL_agent.load(
        # socre: 89.63
        "/home/jiey/RIQL/relaxed_iql/saved_models/hopper-medium-v2/riql_s0_20230211_061526", step=200)
    print("==================\nRIQL Agent Loaded\n==================")

    comfo_agent = COMFO3Agent(obs_dim=obs_dim,
                              act_dim=act_dim,
                              max_action=max_action,
                              hidden_dims=configs.hidden_dims,
                              seed=configs.seed,
                              lr=configs.lr,
                              tau=configs.tau,
                              gamma=configs.gamma,
                              expectile=configs.expectile,
                              temperature=configs.temperature,
                              max_timesteps=configs.max_timesteps,
                              mle_alpha=configs.mle_alpha,
                              initializer=configs.initializer)
    comfo_agent.load(
        # "/home/jiey/RIQL/relaxed_iql/saved_models/hopper-medium-v2/riql_s0_20230211_061526", step=200)
        # socre: 60.87
        "/home/jiey/RIQL/baselines/comfo3/saved_models/20230213/hopper-medium-v2/comfo3_s0_20230212_221317", step=200)
    # socre: 85.08
    # "/home/jiey/RIQL/baselines/comfo3/saved_models/20230213/hopper-medium-v2/comfo3_s0_20230212_190908", step=200)

    print("==================\nCOMFO Agent Loaded\n==================")

    def get_state(env, agent, step):
        obs, _ = env.reset(), False
        for _ in range(step):
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, _, _, _ = env.step(action)
        saved_state = env.sim.get_state()
        return saved_state, obs

    state_store, obs = get_state(env, comfo_agent, COMFO_STEP)
    logdir = f"state_{COMFO_STEP}"
    os.makedirs(f"./{logdir}", exist_ok=True)

    with open(f'./{logdir}/mixed_s{configs.seed}_step{COMFO_STEP}_q_reward.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        agents = [comfo_agent,RIQL_agent]
        for agent in agents:
            for i in range(1):
                agent.seed = i
                agent.rng = jax.random.PRNGKey(agent.seed)
                env.reset()
                env.sim.set_state(state_store)
                trj, _ = get_trj(agent, env, obs)
                print(f"==================\ntraj {i} got")
                data = analysis_trj(trj, agent)
                for row in data:
                    writer.writerow(row)
                print("==================")
            # np.savetxt(
            #     f"./csv/{logdir}/mixed_seed{i}.csv", data, delimiter=",")


if __name__ == '__main__':
    app.run(main)
