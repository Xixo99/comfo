from typing import Any, Callable, Dict, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import distrax
import functools
import jax
from jax import random
import jax.numpy as jnp
import optax
from utils import target_update, Batch


###################
# Utils Functions #
###################
LOG_STD_MAX = 2.
LOG_STD_MIN = -5.


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


def atanh(x: jnp.ndarray):
    one_plus_x = jnp.clip(1 + x, a_min=1e-6)
    one_minus_x = jnp.clip(1 - x, a_min=1e-6)
    return 0.5 * jnp.log(one_plus_x / one_minus_x)


#######################
# Actor-Critic Models #
#######################
class MLP(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    init_fn: Callable = nn.initializers.glorot_uniform()
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.init_fn)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(
            self.initializer), activate_final=True)
        self.out_layer = nn.Dense(
            1, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)

    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.critic1 = Critic(self.hidden_dims, initializer=self.initializer)
        self.critic2 = Critic(self.hidden_dims, initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def Q1(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        return q1


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(
            self.initializer), activate_final=True)
        self.out_layer = nn.Dense(
            1, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        v = self.out_layer(x)
        return v.squeeze(-1)


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    temperature: float = 3.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(
            self.initializer), activate_final=True)
        self.mu_layer = nn.Dense(
            self.act_dim, kernel_init=init_fn(self.initializer, 5/3))
        self.log_std = self.param(
            'log_std', nn.initializers.zeros, (self.act_dim,))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        mean_action = nn.tanh(x) * self.max_action
        return mean_action

    # without tanh
    def get_log_prob(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(x) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(
            mean_action, std*self.temperature)
        log_prob = action_distribution.log_prob(actions)
        return log_prob

    def get_log_prob_tanh(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        log_prob = action_distribution.log_prob(raw_action).sum(-1)
        log_prob -= 2*(jnp.log(2) - raw_action -
                       jax.nn.softplus(-2*raw_action)).sum(-1)
        return log_prob

    def sample(self, observations: jnp.ndarray, rng: Any) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(x, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_actions, log_probs = action_distribution.sample_and_log_prob(
            seed=rng)
        return sampled_actions * self.max_action, log_probs


#############
# New Actor #
#############
# class Sac_Actor(nn.Module):
#     act_dim: int
#     max_action: float = 1.0
#     hidden_dims: Sequence[int] = (256, 256, 256)
#     initializer: str = "orthogonal"

#     def setup(self):
#         self.net = MLP(self.hidden_dims, init_fn=init_fn(
#             self.initializer), activate_final=True)
#         self.mu_layer = nn.Dense(
#             self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))
#         self.std_layer = nn.Dense(
#             self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))

#     def __call__(self, rng: Any, observation: jnp.ndarray) -> jnp.ndarray:
#         x = self.net(observation)
#         mu = self.mu_layer(x)
#         log_std = self.std_layer(x)
#         log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = jnp.exp(log_std)

#         mean_action = nn.tanh(mu)
#         action_distribution = distrax.Transformed(
#             distrax.MultivariateNormalDiag(mu, std),
#             distrax.Block(distrax.Tanh(), ndims=1))
#         sampled_action, logp = action_distribution.sample_and_log_prob(
#             seed=rng)
#         return mean_action*self.max_action, sampled_action*self.max_action, logp

#     def get_logp(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
#         x = self.net(observation)
#         mu = self.mu_layer(x)
#         log_std = self.std_layer(x)
#         log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = jnp.exp(log_std)
#         action_distribution = distrax.Normal(mu, std)
#         raw_action = atanh(action)
#         logp = action_distribution.log_prob(raw_action).sum(-1)
#         logp -= 2*(jnp.log(2) - raw_action -
#                    jax.nn.softplus(-2*raw_action)).sum(-1)
#         return logp

#     def encode(self, observations):
#         embeddings = self.net(observations)
#         return embeddings


###############
# COMFO Agent #
###############
class COMFOAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float,
                 hidden_dims: Sequence[int],
                 seed: int,
                 lr: float,
                 tau: float,
                 gamma: float,
                 expectile: float,
                 temperature: float,
                 max_timesteps: int,
                 conservative_weight: float = 1,
                 noise_scale: float = 0.001,
                 num_random: int = 10,
                 initializer: str = "orthogonal"):

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.conservative_weight = conservative_weight
        self.num_random = num_random

        rng = jax.random.PRNGKey(seed)
        actor_key, critic_key, value_key, self.key_seed = jax.random.split(
            rng, 4)
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           temperature=temperature,
                           hidden_dims=hidden_dims,
                           initializer=initializer)
        actor_params = self.actor.init(actor_key, dummy_obs)["params"]
        schedule_fn = optax.cosine_decay_schedule(-lr, max_timesteps)
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)))

        self.critic = DoubleCritic(
            hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(
            critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=DoubleCritic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr))

        self.value = ValueCritic(hidden_dims, initializer=initializer)
        value_params = self.value.init(value_key, dummy_obs)["params"]
        self.value_state = train_state.TrainState.create(
            apply_fn=ValueCritic.apply,
            params=value_params,
            tx=optax.adam(learning_rate=lr))

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray) -> jnp.ndarray:
        sampled_action = self.actor.apply({"params": params}, observation)
        return sampled_action

    def value_train_step(self,
                         batch: Batch,
                         value_state: train_state.TrainState,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        # update V
        q1, q2 = self.critic.apply(
            {"params": critic_target_params}, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)

        def loss_fn(params: FrozenDict):
            v = self.value.apply({"params": params}, batch.observations)
            weight = jnp.where(q-v > 0, self.expectile, 1-self.expectile)
            value_loss = weight * jnp.square(q-v)
            avg_value_loss = value_loss.mean()
            return avg_value_loss, {
                "value_loss": avg_value_loss, "max_value_loss": value_loss.max(), "min_value_loss": value_loss.min(),
                "weight": weight.mean(), "max_weight": weight.max(), "min_weight": weight.min(),
                "v": v.mean(), "max_v": v.max(), "min_v": v.min()
            }
        (_, value_info), value_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(value_state.params)
        value_state = value_state.apply_gradients(grads=value_grads)
        return value_info, value_state

    def actor_train_step(self,
                         batch: Batch,
                         actor_state: train_state.TrainState,
                         value_params: FrozenDict,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        # note: update \pi with perterbation observations
        def loss_fn(params):
            self.key_seed, new_key = jax.random.split(self.key_seed, 2)
            perturbation = jax.random.uniform(
                new_key, jnp.shape(batch.observations), minval=-self.noise_scale, maxval=self.noise_scale)
            actions = self.actor.apply({"params": params},
                                        batch.observations)
            q1, q2 = self.critic.apply({"params": critic_target_params},
                                       batch.observations+perturbation, actions)
            q = jnp.minimum(q1, q2)
            avg_actor_loss = -(q).mean()
            return avg_actor_loss, {
                "actor_loss": avg_actor_loss,
                "adv": (q).mean(),
                "max_adv": (q).max(),
                "min_adv": (q).min(),
            }
        (_, actor_info), actor_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_state: train_state.TrainState,
                          value_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        # update q
        next_v = self.value.apply(
            {"params": value_params}, batch.next_observations)
        target_q = batch.rewards + self.gamma * batch.discounts * next_v

        def loss_fn(params: FrozenDict):
            q1, q2 = self.critic.apply(
                {"params": params}, batch.observations, batch.actions)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            avg_critic_loss = critic_loss.mean()

            # note: add conservative loss as extra critic loss
            avg_conservative_loss = 0
            for _ in range(self.num_random):
                self.key_seed, new_key = jax.random.split(self.key_seed, 2)
                perturbation = jax.random.uniform(
                    new_key, jnp.shape(batch.observations), minval=-self.noise_scale, maxval=self.noise_scale)
                noise_q11, noise_q12 = self.critic.apply(
                    {"params": params}, batch.observations+perturbation, batch.actions)
                noise_q21, noise_q22 = self.critic.apply(
                    {"params": params}, batch.observations+perturbation*10, batch.actions)

                id_noise_q = jnp.minimum(noise_q11, noise_q12)
                ood_noise_q = jnp.minimum(noise_q21, noise_q22)
                conservative_loss = ood_noise_q - id_noise_q
                avg_conservative_loss += conservative_loss.mean()
            avg_conservative_loss /= self.num_random

            return avg_critic_loss+self.conservative_weight*avg_conservative_loss, {
                "critic_loss": avg_critic_loss,
                "max_critic_loss": critic_loss.max(),
                "min_critic_loss": critic_loss.min(),
                "q1": q1.mean(), "max_q1": q1.max(), "min_q1": q1.min(),
                "q2": q2.mean(), "max_q2": q2.max(), "min_q2": q2.min(),
                "avg_conservative_loss": avg_conservative_loss,
                "target_q": target_q.mean(), "max_target_q": target_q.max(), "min_target_q": target_q.min(),
            }
        (_, critic_info), critic_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   actor_state: train_state.TrainState,
                   value_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        value_info, new_value_state = self.value_train_step(
            batch, value_state, critic_target_params)
        actor_info, new_actor_state = self.actor_train_step(
            batch, actor_state, new_value_state.params, critic_target_params)
        critic_info, new_critic_state = self.critic_train_step(
            batch, critic_state, new_value_state.params)
        new_critic_target_params = target_update(
            new_critic_state.params, critic_target_params, self.tau)
        return new_actor_state, new_value_state, new_critic_state, new_critic_target_params, {
            **actor_info, **value_info, **critic_info}

    def update(self, batch: Batch):
        (self.actor_state, self.value_state, self.critic_state,
         self.critic_target_params, log_info) = self.train_step(batch, self.actor_state, self.value_state,
                                                                self.critic_state, self.critic_target_params)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(
            fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(
            fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(
            fname, self.value_state, cnt, prefix="value_", keep=20, overwrite=True)
