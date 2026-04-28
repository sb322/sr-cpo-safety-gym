"""Top-level SR-CPO training loop.

The local smoke path uses a tiny deterministic JAX toy dynamics model so the
whole algorithm can be exercised on CPU without safe-learning installed. The
losses, replay buffer, dual update, target-network update, and probe formatting
are the same code path used by the production runner.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from sr_cpo.dual_estimator import estimate_discounted_cost
from sr_cpo.env_wrappers import Transition, make_safe_learning_go_to_goal
from sr_cpo.goal_space import _assert_goal_shape, _goal_from_obs
from sr_cpo.losses import (
    actor_loss_fn,
    alpha_loss_fn,
    cost_critic_loss_fn,
    critic_loss_fn,
    sample_tanh_gaussian,
)
from sr_cpo.networks import Actor, CostCritic, GEncoder, SAEncoder
from sr_cpo.pid_lagrangian import PIDState, make_pid_state, update_pid_lagrangian
from sr_cpo.probes import (
    _first_one_idx,
    _grads_global_norm,
    _grads_have_nan,
    _params_have_nan,
)
from sr_cpo.replay_buffer import (
    ReplayBuffer,
    insert_trajectory,
    make_replay_buffer,
    sample_hindsight_transitions,
)

Array = jax.Array
PrintFn = Callable[[str], None]


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for the SR-CPO training loop."""

    seed: int = 0
    epochs: int = 1
    steps_per_epoch: int = 4
    num_envs: int = 1
    unroll_length: int = 8
    use_real_env: bool = False
    env_episode_length: int = 1000
    prefill_steps: int = 2
    sgd_steps: int = 2
    batch_size: int = 8
    buffer_capacity: int = 64
    observation_dim: int = 6
    action_dim: int = 2
    goal_start: int = 0
    goal_dim: int = 3
    width: int = 64
    num_blocks: int = 2
    latent_dim: int = 32
    use_residual: bool = False
    learning_rate: float = 3e-4
    grad_clip_norm: float = 10.0
    tau: float = 0.1
    rho: float = 0.1
    gamma_c: float = 0.99
    target_update_rate: float = 0.005
    nu_f: float = 1.0
    nu_c: float = 1.0
    alpha_max: float = 1.0
    cost_limit: float = 0.0001
    pid_kp: float = 5.0
    pid_ki: float = 0.1
    pid_kd: float = 0.0
    pid_integral_min: float = -10.0
    pid_integral_max: float = 10.0


@dataclass(frozen=True)
class TrainingObjects:
    """Non-pytree modules and optimizers closed over by JITted updates."""

    actor: Actor
    sa_encoder: SAEncoder
    g_encoder: GEncoder
    cost_critic: CostCritic
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    cost_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    action_dim: int
    env_adapter: Any | None = None


@struct.dataclass
class ToyEnvState:
    """Vectorized state for the CPU smoke dynamics."""

    obs: Array


@struct.dataclass
class TrainState:
    """JAX pytree state carried by ``training_epoch``."""

    key: Array
    step: Array
    env_state: Any
    replay: ReplayBuffer
    actor_params: Any
    actor_opt_state: Any
    critic_params: Any
    critic_opt_state: Any
    cost_critic_params: Any
    cost_critic_target_params: Any
    cost_opt_state: Any
    log_alpha: Array
    log_alpha_opt_state: Any
    pid_state: PIDState


def _pad_action(action: Array, observation_dim: int) -> Array:
    pad = observation_dim - action.shape[-1]
    return jnp.pad(action, [(0, 0)] * (action.ndim - 1) + [(0, pad)])


def _toy_step(
    env_state: ToyEnvState,
    action: Array,
    key: Array,
    config: TrainConfig,
) -> tuple[ToyEnvState, Transition]:
    obs = env_state.obs
    noise = 0.01 * jax.random.normal(key, shape=obs.shape, dtype=jnp.float32)
    action_pad = _pad_action(action, config.observation_dim)
    next_obs = 0.98 * obs + 0.05 * action_pad + noise

    hazard_xy = jnp.asarray([0.35, -0.25], dtype=jnp.float32)
    dist_to_hazard = jnp.linalg.norm(next_obs[..., :2] - hazard_xy, axis=-1)
    cost = jnp.maximum(0.0, 0.20 - dist_to_hazard)
    d_wall = 1.0 - jnp.max(jnp.abs(next_obs[..., :2]), axis=-1)
    goal_error = jnp.linalg.norm(
        _goal_from_obs(next_obs, config.goal_start, config.goal_dim), axis=-1
    )
    goal_reached = (goal_error <= 0.05).astype(jnp.float32)
    reward = -goal_error
    discount = jnp.ones_like(reward, dtype=jnp.float32)
    next_state = ToyEnvState(obs=next_obs)
    transition = Transition(
        observation=obs,
        action=action,
        reward=reward,
        discount=discount,
        extras={
            "state": obs,
            "next_state": next_obs,
            "cost": cost.astype(jnp.float32),
            "goal_dist": goal_error.astype(jnp.float32),
            "goal_reached": goal_reached,
            "d_wall": d_wall.astype(jnp.float32),
            "hard_violation": (cost > 0.0).astype(jnp.float32),
        },
    )
    return next_state, transition


def _collect_trajectory(
    train_state: TrainState,
    objects: TrainingObjects,
    config: TrainConfig,
) -> tuple[TrainState, Mapping[str, Array]]:
    if objects.env_adapter is not None:
        return _collect_real_trajectory(train_state, objects, config)
    return _collect_toy_trajectory(train_state, objects, config)


def _collect_toy_trajectory(
    train_state: TrainState,
    objects: TrainingObjects,
    config: TrainConfig,
) -> tuple[TrainState, Mapping[str, Array]]:
    key, scan_key = jax.random.split(train_state.key)

    def collect_step(
        carry: tuple[Array, ToyEnvState], _: Array
    ) -> tuple[tuple[Array, ToyEnvState], Transition]:
        step_key, env_state = carry
        step_key, actor_key, env_key = jax.random.split(step_key, 3)
        goal = jnp.zeros((config.num_envs, config.goal_dim), dtype=jnp.float32)
        _assert_goal_shape(goal, config.goal_dim, context="toy actor rollout")
        sample = sample_tanh_gaussian(
            objects.actor,
            train_state.actor_params,
            env_state.obs,
            goal,
            actor_key,
        )
        next_env_state, transition = _toy_step(
            env_state, sample.action, env_key, config
        )
        return (step_key, next_env_state), transition

    (next_key, next_env_state), transitions = jax.lax.scan(
        collect_step,
        (scan_key, train_state.env_state),
        jnp.arange(config.unroll_length),
    )
    del next_key

    observations = jnp.concatenate(
        [
            train_state.env_state.obs[None, ...],
            transitions.extras["next_state"],
        ],
        axis=0,
    )
    rollout_goals = jnp.zeros(
        (config.unroll_length, config.num_envs, config.goal_dim),
        dtype=jnp.float32,
    )
    replay = _insert_vector_trajectories(
        train_state.replay,
        observations=observations,
        actions=transitions.action,
        rewards=transitions.reward,
        discounts=transitions.discount,
        costs=transitions.extras["cost"],
        d_wall=transitions.extras["d_wall"],
        hard_violations=transitions.extras["hard_violation"],
    )
    metrics = {
        "reward": jnp.mean(transitions.reward),
        "cost": jnp.mean(transitions.extras["cost"]),
        "hard_viol": jnp.mean(transitions.extras["hard_violation"]),
        "goal_dist": jnp.mean(transitions.extras["goal_dist"]),
        "goal_reached": jnp.mean(transitions.extras["goal_reached"]),
        "goal_slice_mean": jnp.mean(rollout_goals),
        "goal_slice_std": jnp.std(rollout_goals),
        "goal_slice_min": jnp.min(rollout_goals),
        "goal_slice_max": jnp.max(rollout_goals),
    }
    next_state = train_state.replace(
        key=key,
        env_state=next_env_state,
        replay=replay,
        step=train_state.step
        + config.num_envs * config.unroll_length,
    )
    return next_state, metrics


def _collect_real_trajectory(
    train_state: TrainState,
    objects: TrainingObjects,
    config: TrainConfig,
) -> tuple[TrainState, Mapping[str, Array]]:
    key, scan_key = jax.random.split(train_state.key)
    env_adapter = objects.env_adapter
    if env_adapter is None:
        raise ValueError("real-env collection requires objects.env_adapter")

    def collect_step(
        carry: tuple[Array, Any], _: Array
    ) -> tuple[tuple[Array, Any], Transition]:
        step_key, env_state = carry
        step_key, actor_key = jax.random.split(step_key)
        goal = _goal_from_obs(env_state.obs, config.goal_start, config.goal_dim)
        _assert_goal_shape(goal, config.goal_dim, context="real actor rollout")
        sample = sample_tanh_gaussian(
            objects.actor,
            train_state.actor_params,
            env_state.obs,
            goal,
            actor_key,
        )
        next_env_state, transition = env_adapter.step(env_state, sample.action)
        return (step_key, next_env_state), transition

    (next_key, next_env_state), transitions = jax.lax.scan(
        collect_step,
        (scan_key, train_state.env_state),
        jnp.arange(config.unroll_length),
    )
    del next_key

    observations = jnp.concatenate(
        [
            train_state.env_state.obs[None, ...],
            transitions.extras["next_state"],
        ],
        axis=0,
    )
    rollout_goals = _goal_from_obs(
        observations[:-1], config.goal_start, config.goal_dim
    )
    _assert_goal_shape(
        rollout_goals, config.goal_dim, context="real actor rollout metrics"
    )
    replay = _insert_vector_trajectories(
        train_state.replay,
        observations=observations,
        actions=transitions.action,
        rewards=transitions.reward,
        discounts=transitions.discount,
        costs=transitions.extras["cost"],
        d_wall=transitions.extras["d_wall"],
        hard_violations=transitions.extras["hard_violation"],
    )
    metrics = {
        "reward": jnp.mean(transitions.reward),
        "cost": jnp.mean(transitions.extras["cost"]),
        "hard_viol": jnp.mean(transitions.extras["hard_violation"]),
        "goal_dist": jnp.mean(transitions.extras["goal_dist"]),
        "goal_reached": jnp.mean(transitions.extras["goal_reached"]),
        "goal_slice_mean": jnp.mean(rollout_goals),
        "goal_slice_std": jnp.std(rollout_goals),
        "goal_slice_min": jnp.min(rollout_goals),
        "goal_slice_max": jnp.max(rollout_goals),
    }
    next_state = train_state.replace(
        key=key,
        env_state=next_env_state,
        replay=replay,
        step=train_state.step + config.num_envs * config.unroll_length,
    )
    return next_state, metrics


def _insert_vector_trajectories(
    buffer: ReplayBuffer,
    *,
    observations: Array,
    actions: Array,
    rewards: Array,
    discounts: Array,
    costs: Array,
    d_wall: Array,
    hard_violations: Array,
) -> ReplayBuffer:
    obs_by_env = jnp.swapaxes(observations, 0, 1)
    act_by_env = jnp.swapaxes(actions, 0, 1)
    rew_by_env = jnp.swapaxes(rewards, 0, 1)
    discount_by_env = jnp.swapaxes(discounts, 0, 1)
    cost_by_env = jnp.swapaxes(costs, 0, 1)
    d_wall_by_env = jnp.swapaxes(d_wall, 0, 1)
    hard_by_env = jnp.swapaxes(hard_violations, 0, 1)

    def insert_one(
        buf: ReplayBuffer, traj: tuple[Array, ...]
    ) -> tuple[ReplayBuffer, None]:
        obs, action, reward, discount, cost, wall, hard = traj
        buf = insert_trajectory(
            buf,
            observations=obs,
            actions=action,
            rewards=reward,
            discounts=discount,
            costs=cost,
            d_wall=wall,
            hard_violations=hard,
        )
        return buf, None

    buffer, _ = jax.lax.scan(
        insert_one,
        buffer,
        (
            obs_by_env,
            act_by_env,
            rew_by_env,
            discount_by_env,
            cost_by_env,
            d_wall_by_env,
            hard_by_env,
        ),
    )
    return buffer


def _sgd_step(
    train_state: TrainState,
    objects: TrainingObjects,
    config: TrainConfig,
) -> tuple[TrainState, dict[str, Array]]:
    key, sample_key, actor_key, cost_key, alpha_key, dual_key = jax.random.split(
        train_state.key, 6
    )
    batch = sample_hindsight_transitions(
        train_state.replay,
        sample_key,
        batch_size=config.batch_size,
        goal_start=config.goal_start,
        goal_end=config.goal_start + config.goal_dim,
    )
    _assert_goal_shape(
        batch.extras["goal"], config.goal_dim, context="hindsight critic"
    )

    def critic_objective(params: Any) -> tuple[Array, dict[str, Array]]:
        return critic_loss_fn(
            params,
            batch,
            sa_encoder=objects.sa_encoder,
            g_encoder=objects.g_encoder,
            tau=config.tau,
            rho=config.rho,
        )

    (c_loss, c_aux), c_grads = jax.value_and_grad(
        critic_objective, has_aux=True
    )(train_state.critic_params)
    c_updates, critic_opt_state = objects.critic_optimizer.update(
        c_grads, train_state.critic_opt_state, train_state.critic_params
    )
    critic_params = optax.apply_updates(train_state.critic_params, c_updates)

    def actor_objective(params: Any) -> tuple[Array, dict[str, Array]]:
        return actor_loss_fn(
            params,
            critic_params,
            train_state.cost_critic_params,
            batch,
            actor_key,
            actor=objects.actor,
            sa_encoder=objects.sa_encoder,
            g_encoder=objects.g_encoder,
            cost_critic=objects.cost_critic,
            log_alpha=train_state.log_alpha,
            lambda_tilde=train_state.pid_state.lambda_tilde,
            tau=config.tau,
            nu_f=config.nu_f,
            nu_c=config.nu_c,
        )

    (a_loss, a_aux), a_grads = jax.value_and_grad(
        actor_objective, has_aux=True
    )(train_state.actor_params)
    a_updates, actor_opt_state = objects.actor_optimizer.update(
        a_grads, train_state.actor_opt_state, train_state.actor_params
    )
    actor_params = optax.apply_updates(train_state.actor_params, a_updates)

    def cost_objective(params: Any) -> tuple[Array, dict[str, Array]]:
        return cost_critic_loss_fn(
            params,
            train_state.cost_critic_target_params,
            actor_params,
            batch,
            cost_key,
            actor=objects.actor,
            cost_critic=objects.cost_critic,
            gamma_c=config.gamma_c,
        )

    (cc_loss, cc_aux), cc_grads = jax.value_and_grad(
        cost_objective, has_aux=True
    )(train_state.cost_critic_params)
    cc_updates, cost_opt_state = objects.cost_optimizer.update(
        cc_grads,
        train_state.cost_opt_state,
        train_state.cost_critic_params,
    )
    cost_critic_params = optax.apply_updates(
        train_state.cost_critic_params, cc_updates
    )
    cost_critic_target_params = optax.incremental_update(
        cost_critic_params,
        train_state.cost_critic_target_params,
        config.target_update_rate,
    )

    sample = sample_tanh_gaussian(
        objects.actor,
        actor_params,
        batch.observation,
        batch.extras["goal"],
        alpha_key,
    )
    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        train_state.log_alpha, sample.log_prob, objects.action_dim
    )
    alpha_updates, log_alpha_opt_state = objects.alpha_optimizer.update(
        alpha_grad, train_state.log_alpha_opt_state, train_state.log_alpha
    )
    log_alpha = optax.apply_updates(train_state.log_alpha, alpha_updates)
    log_alpha_cap = jnp.log(jnp.asarray(config.alpha_max, dtype=jnp.float32))
    log_alpha = jnp.minimum(log_alpha, log_alpha_cap)

    jc_hat, dual_aux = estimate_discounted_cost(
        cost_critic=objects.cost_critic,
        cost_critic_params=cost_critic_params,
        actor=objects.actor,
        actor_params=actor_params,
        initial_states=batch.observation,
        goals=batch.extras["goal"],
        key=dual_key,
        gamma_c=config.gamma_c,
        num_action_samples=2,
    )
    pid_state = update_pid_lagrangian(
        train_state.pid_state,
        estimated_cost=jc_hat,
        budget=config.cost_limit,
        kp=config.pid_kp,
        ki=config.pid_ki,
        kd=config.pid_kd,
        integral_min=config.pid_integral_min,
        integral_max=config.pid_integral_max,
    )
    pid_error = jc_hat - jnp.asarray(config.cost_limit, dtype=jnp.float32)
    pid_derivative = pid_error - train_state.pid_state.previous_error
    pid_raw_lambda = (
        config.pid_kp * pid_error
        + config.pid_ki * pid_state.integral
        + config.pid_kd * pid_derivative
    )

    c_grad_nan = _grads_have_nan(c_grads)
    a_grad_nan = _grads_have_nan(a_grads)
    cc_grad_nan = _grads_have_nan(cc_grads)
    # Static-diff markers required by the Wulver launch scripts:
    # nan_obs_critic, nan_sa_critic, nan_g_critic, nan_logits_critic
    # alpha_logprob_actor, sat_correction_actor, log_std_mean_actor
    metrics = {
        "c_loss": c_loss,
        "c_accuracy": c_aux["accuracy"],
        "a_loss": a_loss,
        "cc_loss": cc_loss,
        "alpha_loss": alpha_loss,
        "nan_obs_critic": c_aux["nan_obs"],
        "nan_sa_critic": c_aux["nan_sa"],
        "nan_g_critic": c_aux["nan_g"],
        "nan_logits_critic": c_aux["nan_logits"],
        "sa_norm_min_critic": c_aux["sa_norm_min"],
        "g_norm_min_critic": c_aux["g_norm_min"],
        "nan_sa_actor": a_aux["nan_sa"],
        "nan_g_actor": a_aux["nan_g"],
        "nan_action_actor": a_aux["nan_action"],
        "nan_f_actor": a_aux["nan_f"],
        "sa_norm_min_actor": a_aux["sa_norm_min"],
        "g_norm_min_actor": a_aux["g_norm_min"],
        "action_abs_max": a_aux["action_abs_max"],
        "alpha_actor": a_aux["alpha"],
        "log_prob_actor": a_aux["log_prob_mean"],
        "alpha_logprob_actor": a_aux["alpha_logprob_mean"],
        "gaussian_logp_actor": a_aux["gaussian_logp_mean"],
        "sat_correction_actor": a_aux["sat_correction_mean"],
        "log_std_mean_actor": a_aux["log_std_mean"],
        "f_term_actor": a_aux["f_term_mean"],
        "qc_actor": a_aux["qc_actor_mean"],
        "lambda_qc_actor": a_aux["constraint_term_mean"],
        "nu_c": jnp.asarray(config.nu_c, dtype=jnp.float32),
        "alpha_clip": jnp.minimum(jnp.exp(log_alpha) / config.alpha_max, 1.0),
        "cost": cc_aux["mean_cost"],
        "qc": cc_aux["mean_qc"],
        "jc_hat": jc_hat,
        "dual_qc_mean": dual_aux["dual_qc_mean"],
        "cost_limit": jnp.asarray(config.cost_limit, dtype=jnp.float32),
        "pid_error": pid_error,
        "pid_integral": pid_state.integral,
        "pid_raw_lambda": pid_raw_lambda.astype(jnp.float32),
        "lambda_tilde": pid_state.lambda_tilde,
        "c_grad_nan": c_grad_nan,
        "a_grad_nan": a_grad_nan,
        "cc_grad_nan": cc_grad_nan,
        "c_grad_norm": _grads_global_norm(c_grads),
        "a_grad_norm": _grads_global_norm(a_grads),
        "cc_grad_norm": _grads_global_norm(cc_grads),
        "c_params_nan": _params_have_nan(critic_params),
        "a_params_nan": _params_have_nan(actor_params),
        "cc_params_nan": _params_have_nan(cost_critic_params),
    }
    next_state = train_state.replace(
        key=key,
        actor_params=actor_params,
        actor_opt_state=actor_opt_state,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        cost_critic_params=cost_critic_params,
        cost_critic_target_params=cost_critic_target_params,
        cost_opt_state=cost_opt_state,
        log_alpha=log_alpha,
        log_alpha_opt_state=log_alpha_opt_state,
        pid_state=pid_state,
    )
    return next_state, metrics


def _mean_metrics(metrics: Mapping[str, Array]) -> dict[str, Array]:
    return {name: jnp.mean(value, axis=0) for name, value in metrics.items()}


def make_training_epoch(
    objects: TrainingObjects,
    config: TrainConfig,
) -> Callable[[TrainState], tuple[TrainState, Mapping[str, Array]]]:
    """Builds a JITted epoch function with nested training and SGD scans."""

    def training_step(
        state: TrainState, _: Array
    ) -> tuple[TrainState, Mapping[str, Array]]:
        state, collect_metrics = _collect_trajectory(state, objects, config)

        def scan_sgd(
            sgd_state: TrainState, __: Array
        ) -> tuple[TrainState, Mapping[str, Array]]:
            return _sgd_step(sgd_state, objects, config)

        state, sgd_metrics = jax.lax.scan(
            scan_sgd, state, jnp.arange(config.sgd_steps)
        )
        metrics = _mean_metrics(sgd_metrics)
        metrics["hard_viol"] = collect_metrics["hard_viol"]
        metrics["rollout_cost"] = collect_metrics["cost"]
        metrics["rollout_reward"] = collect_metrics["reward"]
        metrics["goal_dist"] = collect_metrics["goal_dist"]
        metrics["goal_reached"] = collect_metrics["goal_reached"]
        metrics["goal_start"] = jnp.asarray(config.goal_start, dtype=jnp.float32)
        metrics["goal_dim"] = jnp.asarray(config.goal_dim, dtype=jnp.float32)
        metrics["goal_slice_mean"] = collect_metrics["goal_slice_mean"]
        metrics["goal_slice_std"] = collect_metrics["goal_slice_std"]
        metrics["goal_slice_min"] = collect_metrics["goal_slice_min"]
        metrics["goal_slice_max"] = collect_metrics["goal_slice_max"]
        return state, metrics

    @jax.jit
    def training_epoch(state: TrainState) -> tuple[TrainState, Mapping[str, Array]]:
        return jax.lax.scan(
            training_step, state, jnp.arange(config.steps_per_epoch)
        )

    return training_epoch


def initialize_training(
    config: TrainConfig, env_adapter: Any | None = None
) -> tuple[TrainState, TrainingObjects]:
    """Initializes modules, optimizers, toy env state, and replay buffer."""

    key = jax.random.PRNGKey(config.seed)
    (
        key,
        actor_key,
        sa_key,
        g_key,
        cc_key,
        obs_key,
        env_key,
    ) = jax.random.split(key, 7)
    runtime_observation_dim = config.observation_dim
    runtime_action_dim = config.action_dim
    if config.use_real_env:
        env_adapter = env_adapter or make_safe_learning_go_to_goal(
            num_envs=config.num_envs,
            episode_length=config.env_episode_length,
        )
        env_state, reset_transition = env_adapter.reset(env_key)
        runtime_observation_dim = int(reset_transition.observation.shape[-1])
        runtime_action_dim = int(env_adapter.action_size)
    else:
        env_state = ToyEnvState(
            obs=0.1
            * jax.random.normal(
                obs_key,
                (config.num_envs, runtime_observation_dim),
                dtype=jnp.float32,
            )
        )
    if config.goal_start < 0:
        raise ValueError("goal_start must be non-negative")
    if config.goal_dim <= 0:
        raise ValueError("goal_dim must be positive")
    if config.goal_start + config.goal_dim > runtime_observation_dim:
        raise ValueError(
            "goal_start + goal_dim must fit inside the runtime observation dimension"
        )

    actor = Actor(
        action_size=runtime_action_dim,
        width=config.width,
        num_blocks=config.num_blocks,
        use_residual=config.use_residual,
    )
    sa_encoder = SAEncoder(
        width=config.width,
        num_blocks=config.num_blocks,
        latent_dim=config.latent_dim,
        use_residual=config.use_residual,
    )
    g_encoder = GEncoder(
        width=config.width,
        num_blocks=config.num_blocks,
        latent_dim=config.latent_dim,
        use_residual=config.use_residual,
    )
    cost_critic = CostCritic(
        width=config.width,
        num_blocks=config.num_blocks,
        use_residual=config.use_residual,
    )
    dummy_state = jnp.zeros((1, runtime_observation_dim), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, runtime_action_dim), dtype=jnp.float32)
    dummy_goal = jnp.zeros((1, config.goal_dim), dtype=jnp.float32)
    actor_params = actor.init(actor_key, dummy_state, dummy_goal)
    critic_params = {
        "sa_encoder": sa_encoder.init(sa_key, dummy_state, dummy_action),
        "g_encoder": g_encoder.init(g_key, dummy_goal),
    }
    cost_critic_params = cost_critic.init(cc_key, dummy_state, dummy_action, dummy_goal)

    actor_optimizer = _make_optimizer(config)
    critic_optimizer = _make_optimizer(config)
    cost_optimizer = _make_optimizer(config)
    alpha_optimizer = optax.adam(config.learning_rate)
    replay = make_replay_buffer(
        capacity=config.buffer_capacity,
        episode_length=config.unroll_length,
        observation_dim=runtime_observation_dim,
        action_dim=runtime_action_dim,
    )
    state = TrainState(
        key=key,
        step=jnp.asarray(0, dtype=jnp.int32),
        env_state=env_state,
        replay=replay,
        actor_params=actor_params,
        actor_opt_state=actor_optimizer.init(actor_params),
        critic_params=critic_params,
        critic_opt_state=critic_optimizer.init(critic_params),
        cost_critic_params=cost_critic_params,
        cost_critic_target_params=cost_critic_params,
        cost_opt_state=cost_optimizer.init(cost_critic_params),
        log_alpha=jnp.asarray(0.0, dtype=jnp.float32),
        log_alpha_opt_state=alpha_optimizer.init(jnp.asarray(0.0, dtype=jnp.float32)),
        pid_state=make_pid_state(),
    )
    objects = TrainingObjects(
        actor=actor,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        cost_critic=cost_critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        cost_optimizer=cost_optimizer,
        alpha_optimizer=alpha_optimizer,
        action_dim=runtime_action_dim,
        env_adapter=env_adapter,
    )
    return state, objects


def _make_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    if config.grad_clip_norm <= 0.0:
        return optax.adam(config.learning_rate)
    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.learning_rate),
    )


def prefill_buffer(
    state: TrainState,
    objects: TrainingObjects,
    config: TrainConfig,
) -> TrainState:
    """Collects initial trajectories before the first SGD update."""

    def prefill_step(carry: TrainState, _: Array) -> tuple[TrainState, None]:
        carry, _ = _collect_trajectory(carry, objects, config)
        return carry, None

    @jax.jit
    def run_prefill(carry: TrainState) -> TrainState:
        carry, _ = jax.lax.scan(
            prefill_step, carry, jnp.arange(config.prefill_steps)
        )
        return carry

    return run_prefill(state)


def _mean_float(metrics: Mapping[str, Array], key: str) -> float:
    return float(jnp.mean(metrics[key]))


def _max_flag(metrics: Mapping[str, Array], key: str) -> int:
    return int(jnp.max(metrics[key]))


def format_epoch_metrics(
    epoch: int,
    total_epochs: int,
    metrics: Mapping[str, Array],
    *,
    steps: int,
    elapsed: float,
) -> str:
    """Formats the grep-compatible per-epoch probe block."""

    return "\n".join(
        [
            (
                f"[{epoch + 1:2d}/{total_epochs:2d}] steps={steps:,} | "
                f"c_loss={_mean_float(metrics, 'c_loss'):.4f} "
                f"acc={_mean_float(metrics, 'c_accuracy'):.3f} | "
                f"a_loss={_mean_float(metrics, 'a_loss'):+.4f} | "
                f"t={elapsed:.1f}s"
            ),
            (
                "         "
                f"hard_viol={_mean_float(metrics, 'hard_viol'):.4f} "
                f"cost={_mean_float(metrics, 'cost'):.4f} "
                f"rew={_mean_float(metrics, 'rollout_reward'):.4f} "
                f"gdist={_mean_float(metrics, 'goal_dist'):.4f} "
                f"reached={_mean_float(metrics, 'goal_reached'):.4f} "
                f"gstart={_mean_float(metrics, 'goal_start'):.0f} "
                f"gdim={_mean_float(metrics, 'goal_dim'):.0f} "
                f"gmean={_mean_float(metrics, 'goal_slice_mean'):.3f} "
                f"gstd={_mean_float(metrics, 'goal_slice_std'):.3f} "
                f"gmin={_mean_float(metrics, 'goal_slice_min'):.3f} "
                f"gmax={_mean_float(metrics, 'goal_slice_max'):.3f} "
                f"λ̃={_mean_float(metrics, 'lambda_tilde'):.4f} "
                f"Ĵ_c={_mean_float(metrics, 'jc_hat'):.4f} "
                f"Qc={_mean_float(metrics, 'qc'):.4f} "
                f"limit={_mean_float(metrics, 'cost_limit'):.2e} "
                f"pid_err={_mean_float(metrics, 'pid_error'):.2e} "
                f"S={_mean_float(metrics, 'pid_integral'):.2e} "
                f"λraw={_mean_float(metrics, 'pid_raw_lambda'):.2e} "
                f"λQc_a={_mean_float(metrics, 'lambda_qc_actor'):.2e} "
                f"nu_c={_mean_float(metrics, 'nu_c'):.1e}"
            ),
            (
                "         "
                f"nan[obs_c={_max_flag(metrics, 'nan_obs_critic')} "
                f"sa_c={_max_flag(metrics, 'nan_sa_critic')} "
                f"g_c={_max_flag(metrics, 'nan_g_critic')} "
                f"lg_c={_max_flag(metrics, 'nan_logits_critic')} "
                f"sa_a={_max_flag(metrics, 'nan_sa_actor')} "
                f"g_a={_max_flag(metrics, 'nan_g_actor')} "
                f"act_a={_max_flag(metrics, 'nan_action_actor')} "
                f"f_a={_max_flag(metrics, 'nan_f_actor')}] "
                f"‖φ‖min_c={_mean_float(metrics, 'sa_norm_min_critic'):.3f} "
                f"‖ψ‖min_c={_mean_float(metrics, 'g_norm_min_critic'):.3f} "
                f"‖φ‖min_a={_mean_float(metrics, 'sa_norm_min_actor'):.3f} "
                f"‖ψ‖min_a={_mean_float(metrics, 'g_norm_min_actor'):.3f} "
                f"|a|max={_mean_float(metrics, 'action_abs_max'):.3f}"
            ),
            (
                "         "
                f"grad[c={_max_flag(metrics, 'c_grad_nan')}/"
                f"{_mean_float(metrics, 'c_grad_norm'):.2e} "
                f"a={_max_flag(metrics, 'a_grad_nan')}/"
                f"{_mean_float(metrics, 'a_grad_norm'):.2e} "
                f"cc={_max_flag(metrics, 'cc_grad_nan')}/"
                f"{_mean_float(metrics, 'cc_grad_norm'):.2e}] "
                f"params[c={_max_flag(metrics, 'c_params_nan')} "
                f"a={_max_flag(metrics, 'a_params_nan')} "
                f"cc={_max_flag(metrics, 'cc_params_nan')}]"
            ),
            (
                "         "
                f"actor[α={_mean_float(metrics, 'alpha_actor'):.4f} "
                f"log_p={_mean_float(metrics, 'log_prob_actor'):.3f} "
                f"α·log_p={_mean_float(metrics, 'alpha_logprob_actor'):.3f} "
                f"gauss_lp={_mean_float(metrics, 'gaussian_logp_actor'):.3f} "
                f"sat_corr={_mean_float(metrics, 'sat_correction_actor'):.3f} "
                f"log_std={_mean_float(metrics, 'log_std_mean_actor'):.3f} "
                f"f_term={_mean_float(metrics, 'f_term_actor'):.3f} "
                f"α_clip={_mean_float(metrics, 'alpha_clip'):.2f}]"
            ),
        ]
    )


def print_prefill_probe(state: TrainState, print_fn: PrintFn = print) -> None:
    buffer_nan = any(
        bool(jnp.any(~jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(state.replay)
    )
    env_nan = bool(jnp.any(~jnp.isfinite(state.env_state.obs)))
    print_fn(
        "[prefill probe] "
        f"buffer.data shape={state.replay.observations.shape} "
        f"buffer NaN anywhere={int(buffer_nan)} "
        f"env_state.obs NaN={int(env_nan)}"
    )


def print_epoch1_forensics(
    metrics: Mapping[str, Array], print_fn: PrintFn = print
) -> None:
    print_fn("[epoch1 forensics]")
    for name in (
        "nan_obs_critic",
        "nan_sa_critic",
        "nan_g_critic",
        "nan_logits_critic",
        "nan_sa_actor",
        "nan_g_actor",
        "nan_action_actor",
        "nan_f_actor",
        "c_grad_nan",
        "a_grad_nan",
        "cc_grad_nan",
    ):
        print_fn(f"  first {name}={int(_first_one_idx(metrics[name]))}")
    for name in ("c_grad_norm", "a_grad_norm", "cc_grad_norm"):
        values = [float(x) for x in jnp.ravel(metrics[name])[:5]]
        rendered = ", ".join(f"{value:.3e}" for value in values)
        print_fn(f"  {name} first5=[{rendered}]")
    for name in ("c_params_nan", "a_params_nan", "cc_params_nan"):
        values = [int(x) for x in jnp.ravel(metrics[name])[:5]]
        rendered = ", ".join(str(value) for value in values)
        print_fn(f"  {name} first5=[{rendered}]")


def run_training(
    config: TrainConfig | None = None,
    *,
    print_fn: PrintFn = print,
) -> dict[str, Any]:
    """Runs SR-CPO training and prints the required probe blocks."""

    config = TrainConfig() if config is None else config
    state, objects = initialize_training(config)
    state = prefill_buffer(state, objects, config)
    print_prefill_probe(state, print_fn)
    training_epoch = make_training_epoch(objects, config)

    last_metrics: Mapping[str, Array] | None = None
    for epoch in range(config.epochs):
        start = time.perf_counter()
        state, metrics = training_epoch(state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        elapsed = time.perf_counter() - start
        steps = int(state.step)
        print_fn(
            format_epoch_metrics(
                epoch,
                config.epochs,
                metrics,
                steps=steps,
                elapsed=elapsed,
            )
        )
        if epoch == 0:
            print_epoch1_forensics(metrics, print_fn)
        last_metrics = metrics

    return {
        "state": state,
        "metrics": last_metrics,
        "epochs": config.epochs,
    }
