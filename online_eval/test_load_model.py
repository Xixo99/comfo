import os
import time
from typing import Tuple

import gym
import jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags
from models import COMFO3Agent, RIQLAgent
from utils import ReplayBuffer

import d4rl

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


def main(argv):
    config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
    FLAGS = flags.FLAGS
    configs = FLAGS.config
    configs.seed = 1

    env = gym.make("hopper-medium-v2")
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

    eval_reward, eval_time = eval_policy(
        RIQL_agent, env, configs.eval_episodes)
    print(
        f"#############\neval_reward:{eval_reward},eval_time:{eval_time}\n#############")

    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    batch = replay_buffer.sample(configs.batch_size)

    q11, q12 = RIQL_agent.critic.apply(
        {"params": RIQL_agent.critic_state.params}, batch.observations, batch.actions)
    q1 = jnp.minimum(q11, q12)
    print(q1.mean())
    print("#############")

    value = RIQL_agent.value.apply(
        {"params": RIQL_agent.value_state.params}, batch.observations)
    print(value.mean())
    print("#############")

def eval_policy(agent, env: gym.Env, eval_episodes: int = 100) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        env.set_state()
        while not done:
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


if __name__ == '__main__':
    app.run(main)
