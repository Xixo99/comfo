from utils import ReplayBuffer, get_logger
from typing import Any, Callable, Dict, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import distrax
import functools
import jax
import jax.numpy as jnp
import optax
from utils import target_update, Batch


###################
# Utils Functions #
###################
LOG_STD_MAX = 2.
LOG_STD_MIN = -5.


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


#############
# New Actor #
#############
class RIQL_Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(
            self.initializer), activate_final=True)
        self.mu_layer = nn.Dense(
            self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))
        self.std_layer = nn.Dense(
            self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))

    def get_logprob(self, observation, action):
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = distrax.Normal(mu, std)

        # Compute log_prob
        raw_action = atanh(action)
        log_prob = pi_distribution.log_prob(raw_action).sum(-1)
        log_prob -= (2*(jnp.log(2) - raw_action -
                     jax.nn.softplus(-2 * raw_action))).sum(-1)
        return log_prob

    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        mean_action = nn.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(
            seed=rng)
        return mean_action*self.max_action, sampled_action*self.max_action, logp


#############
# IQL Agent #
#############
# RIQL Actor with entropy
class RIQLAgent:
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
                 mle_alpha: float = 1.0,
                 initializer: str = "orthogonal"):

        self.act_dim = act_dim
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau = tau

        # RIQL MLE
        self.mle_alpha = mle_alpha

        # SAC Actor
        self.target_entropy = -act_dim

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, value_key = jax.random.split(
            self.rng, 4)
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = RIQL_Actor(act_dim=act_dim,
                                max_action=max_action,
                                hidden_dims=hidden_dims,
                                initializer=initializer)
        actor_params = self.actor.init(
            actor_key, actor_key, dummy_obs)["params"]
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

        # Entropy
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.adam(lr)
        )

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray) -> jnp.ndarray:
        sampled_action, _, _ = self.actor.apply(
            {"params": params}, self.rng, observation)
        return sampled_action

    def value_train_step(self,
                         batch: Batch,
                         value_state: train_state.TrainState,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
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

    def actor_alpha_train_step(self,
                               batch: Batch,
                               rng: Any,
                               alpha_state: train_state.TrainState,
                               actor_state: train_state.TrainState,
                               critic_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        def loss_fn(params, alpha_params):
            _, sampled_actions, logp = self.actor.apply(
                {"params": params}, rng, batch.observations)
            mle_prob = self.actor.apply({"params": params}, batch.observations, batch.actions,
                                        method=self.actor.get_logprob)

            # Alpha loss: stop gradient to avoid affecting Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            # alpha_loss: adapt sac temperature
            alpha_loss = -log_alpha * \
                jax.lax.stop_gradient(logp + self.target_entropy)

            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)
            # Actor loss
            q1, q2 = self.critic.apply({"params": critic_params},
                                       batch.observations, sampled_actions)
            q = jnp.minimum(q1, q2)
            sac_loss = -q+alpha*logp
            actor_loss = sac_loss - self.mle_alpha*mle_prob

            loss = actor_loss.mean() + alpha_loss.mean()
            return loss, {
                "actor_loss": actor_loss.mean(),
                "max_actor_loss": actor_loss.max(),
                "min_actor_loss": actor_loss.min(),
                "alpha": alpha,
                "alpha_loss": alpha_loss.mean(),
                "max_alpha_loss": alpha_loss.max(),
                "min_alpha_loss": alpha_loss.min(),
                "mle_prob": mle_prob.mean(),
                "max_mle_prob": mle_prob.max(),
                "min_mle_prob": mle_prob.min(),
            }

        (_, actor_info), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            actor_state.params, alpha_state.params)
        actor_grads, alpha_grads = grads
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        return actor_info, actor_state, alpha_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_state: train_state.TrainState,
                          value_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        next_v = self.value.apply(
            {"params": value_params}, batch.next_observations)
        target_q = batch.rewards + self.gamma * batch.discounts * next_v

        def loss_fn(params: FrozenDict):
            q1, q2 = self.critic.apply(
                {"params": params}, batch.observations, batch.actions)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            avg_critic_loss = critic_loss.mean()
            return avg_critic_loss, {
                "critic_loss": avg_critic_loss,
                "max_critic_loss": critic_loss.max(),
                "min_critic_loss": critic_loss.min(),
                "q1": q1.mean(), "max_q1": q1.max(), "min_q1": q1.min(),
                "q2": q2.mean(), "max_q2": q2.max(), "min_q2": q2.min(),
                "target_q": target_q.mean(), "max_target_q": target_q.max(), "min_target_q": target_q.min(),
            }
        (_, critic_info), critic_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   rng: Any,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   value_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        value_info, new_value_state = self.value_train_step(
            batch, value_state, critic_target_params)
        critic_info, new_critic_state = self.critic_train_step(
            batch, critic_state, new_value_state.params)
        actor_info, new_actor_state, new_alpha_state = self.actor_alpha_train_step(
            batch, rng, alpha_state, actor_state, critic_state.params)
        new_critic_target_params = target_update(
            new_critic_state.params, critic_target_params, self.tau)
        return new_alpha_state, new_actor_state, new_value_state, new_critic_state, new_critic_target_params, {
            **actor_info, **value_info, **critic_info}

    def update(self, batch: Batch):
        self.rng, update_rng = jax.random.split(self.rng, 2)
        (self.alpha_state, self.actor_state, self.value_state, self.critic_state,
         self.critic_target_params, log_info) = self.train_step(batch,
                                                                update_rng,
                                                                self.alpha_state,
                                                                self.actor_state,
                                                                self.value_state,
                                                                self.critic_state,
                                                                self.critic_target_params)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(
            fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(
            fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(
            fname, self.value_state, cnt, prefix="value_", keep=20, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.actor_state,
                                                          step=step, prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.critic_state,
                                                           step=step, prefix="critic_")
        self.value_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.value_state,
                                                          step=step, prefix="value_")


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()



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


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


#############
# New Actor #
#############
class COMFO3_Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(
            self.initializer), activate_final=True)
        self.mu_layer = nn.Dense(
            self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))
        self.std_layer = nn.Dense(
            self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))

    def get_logprob(self, observation, action):
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = distrax.Normal(mu, std)

        # Compute log_prob
        raw_action = atanh(action)
        log_prob = pi_distribution.log_prob(raw_action).sum(-1)
        log_prob -= (2*(jnp.log(2) - raw_action -
                     jax.nn.softplus(-2 * raw_action))).sum(-1)
        return log_prob

    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        mean_action = nn.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(
            seed=rng)
        return mean_action*self.max_action, sampled_action*self.max_action, logp




# COMFO3 Actor with entropy
class COMFO3Agent:
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
                 mle_alpha: float = 0.1,
                 conservative_weight: float = 1.0,
                 noise_scale: float = 0.001,
                 initializer: str = "orthogonal"):

        self.act_dim = act_dim
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau = tau

        # MLE
        self.mle_alpha = mle_alpha

        # SAC Actor
        self.target_entropy = -act_dim

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, value_key, key_seed = jax.random.split(
            self.rng, 5)
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = COMFO3_Actor(act_dim=act_dim,
                                  max_action=max_action,
                                  hidden_dims=hidden_dims,
                                  initializer=initializer)
        actor_params = self.actor.init(
            actor_key, actor_key, dummy_obs)["params"]
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

        # Entropy
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.adam(lr)
        )

        # new elements
        self.conservative_weight = conservative_weight
        self.noise_scale = noise_scale
        self.num_random = 10
        self.key_seed = key_seed

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray) -> jnp.ndarray:
        sampled_action, _, _ = self.actor.apply(
            {"params": params}, self.rng, observation)
        return sampled_action

    def value_train_step(self,
                         batch: Batch,
                         value_state: train_state.TrainState,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
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

    def actor_alpha_train_step(self,
                               batch: Batch,
                               rng: Any,
                               alpha_state: train_state.TrainState,
                               actor_state: train_state.TrainState,
                               critic_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        def loss_fn(params, alpha_params):
            _, sampled_actions, logp = self.actor.apply(
                {"params": params}, rng, batch.observations)
            mle_prob = self.actor.apply({"params": params}, batch.observations, batch.actions,
                                        method=self.actor.get_logprob)

            # Alpha loss: stop gradient to avoid affecting Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * \
                jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)

            # Actor loss
            q1, q2 = self.critic.apply({"params": critic_params},
                                       batch.observations, sampled_actions)
            q = jnp.minimum(q1, q2)
            actor_loss = alpha*logp - self.mle_alpha*mle_prob - q

            loss = actor_loss.mean() + alpha_loss.mean()
            return loss, {
                "actor_loss": actor_loss.mean(),
                "max_actor_loss": actor_loss.max(),
                "min_actor_loss": actor_loss.min(),
                "alpha": alpha,
                "alpha_loss": alpha_loss.mean(),
                "max_alpha_loss": alpha_loss.max(),
                "min_alpha_loss": alpha_loss.min(),
                "mle_prob": mle_prob.mean(),
                "max_mle_prob": mle_prob.max(),
                "min_mle_prob": mle_prob.min(),
            }

        (_, actor_info), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            actor_state.params, alpha_state.params)
        actor_grads, alpha_grads = grads
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        return actor_info, actor_state, alpha_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_state: train_state.TrainState,
                          value_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
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
            key = self.rng
            for _ in range(self.num_random):
                key, sub_key = jax.random.split(key, 2)
                perturbation = jax.random.uniform(
                    sub_key, jnp.shape(batch.observations), minval=-self.noise_scale, maxval=self.noise_scale)
                noise_q11, noise_q12 = self.critic.apply(
                    {"params": params}, batch.observations+perturbation, batch.actions)
                noise_q21, noise_q22 = self.critic.apply(
                    {"params": params}, batch.observations+perturbation*10, batch.actions)

                id_noise_q = jnp.minimum(noise_q11, noise_q12)
                ood_noise_q = jnp.minimum(noise_q21, noise_q22)
                conservative_loss = ood_noise_q - id_noise_q
                avg_conservative_loss += conservative_loss.mean()
            avg_conservative_loss /= self.num_random
            # print("****************")
            # print(avg_conservative_loss)
            # print("****************")
            # if avg_conservative_loss < -10.0:
            #     self.conservative_weight /= 2

            return avg_critic_loss+self.conservative_weight*avg_conservative_loss, {
                "critic_loss": avg_critic_loss,
                "max_critic_loss": critic_loss.max(),
                "min_critic_loss": critic_loss.min(),
                "avg_conservative_loss": avg_conservative_loss,
                "conservative_weight": self.conservative_weight,
                "q1": q1.mean(), "max_q1": q1.max(), "min_q1": q1.min(),
                "q2": q2.mean(), "max_q2": q2.max(), "min_q2": q2.min(),
                "target_q": target_q.mean(), "max_target_q": target_q.max(), "min_target_q": target_q.min(),
            }

        (_, critic_info), critic_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        # if avg_conservative_loss < 10.0:
        #     self.conservative_weight /= 2
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   rng: Any,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   value_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        value_info, new_value_state = self.value_train_step(
            batch, value_state, critic_target_params)
        critic_info, new_critic_state = self.critic_train_step(
            batch, critic_state, new_value_state.params)
        actor_info, new_actor_state, new_alpha_state = self.actor_alpha_train_step(
            batch, rng, alpha_state, actor_state, critic_state.params)
        new_critic_target_params = target_update(
            new_critic_state.params, critic_target_params, self.tau)
        return new_alpha_state, new_actor_state, new_value_state, new_critic_state, new_critic_target_params, {
            **actor_info, **value_info, **critic_info}

    def update(self, batch: Batch):
        self.rng, update_rng = jax.random.split(self.rng, 2)
        (self.alpha_state, self.actor_state, self.value_state, self.critic_state,
         self.critic_target_params, log_info) = self.train_step(batch,
                                                                update_rng,
                                                                self.alpha_state,
                                                                self.actor_state,
                                                                self.value_state,
                                                                self.critic_state,
                                                                self.critic_target_params)
        if log_info['avg_conservative_loss'] < -10.0:
            # print("**avg_conservative_loss")
            self.conservative_weight /= 2
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(
            fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(
            fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(
            fname, self.value_state, cnt, prefix="value_", keep=20, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.actor_state,
                                                          step=step, prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.critic_state,
                                                           step=step, prefix="critic_")
        self.value_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.value_state,
                                                          step=step, prefix="value_")
