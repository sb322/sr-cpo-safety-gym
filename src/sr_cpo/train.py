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
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import serialization, struct

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
    replay_risky_available_fraction,
    sample_hindsight_transitions,
    sample_risk_biased_hindsight_transitions,
)

Array = jax.Array
PrintFn = Callable[[str], None]
XY_GOAL_MODES = {"xy", "relative_xy"}
GOAL_DISTANCE_METRIC_KEYS = (
    "goal_dist",
    "goal_dist_p10",
    "goal_dist_p50",
    "goal_dist_p90",
    "goal_dist_lt_0_5",
    "goal_dist_lt_1_0",
    "goal_dist_lt_2_0",
)
EVAL_PRIMARY_GOAL_RADII = (0.31, 0.5, 1.0, 2.0)


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
    goal_mode: str = "obs_slice"
    goal_start: int = 0
    goal_dim: int = 3
    mask_goal_in_state: bool = False
    mask_native_goal_lidar: bool = False
    probe_counterfactual_costs: bool = False
    width: int = 64
    num_blocks: int = 2
    latent_dim: int = 32
    use_residual: bool = False
    learning_rate: float = 3e-4
    grad_clip_norm: float = 10.0
    tau: float = 0.1
    rho: float = 0.1
    critic_score_mode: str = "cosine"
    gamma_c: float = 0.99
    cost_return_loss_weight: float = 0.0
    cost_risk_replay_ratio: float = 0.0
    cost_risk_hazard_lidar_thresh: float = 0.5
    cost_risk_min_fraction_available: float = 0.0
    target_update_rate: float = 0.005
    nu_f: float = 1.0
    nu_c: float = 1.0
    entropy_param: float = 0.5
    alpha_max: float = 1.0
    cost_limit: float = 0.0001
    pid_kp: float = 5.0
    pid_ki: float = 0.1
    pid_kd: float = 0.0
    pid_integral_min: float = -10.0
    pid_integral_max: float = 10.0
    pid_integral_decay: float = 1.0
    eval_freeze_goal_after_success: bool = False
    eval_action_std_scales: str = "0.0"
    checkpoint_output: str = ""


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


def _mask_goal_in_state(obs: Array, config: TrainConfig) -> Array:
    """Optionally removes the external goal slice from state inputs."""

    if config.goal_mode in XY_GOAL_MODES:
        return obs
    if not config.mask_goal_in_state:
        return obs
    return obs.at[..., config.goal_start : config.goal_start + config.goal_dim].set(
        0.0
    )


def _mask_transition_state_inputs(
    transitions: Transition, config: TrainConfig
) -> Transition:
    """Masks state channels seen by losses while preserving replay goals."""

    if not config.mask_goal_in_state:
        return transitions
    extras = dict(transitions.extras)
    if "next_state" in extras:
        extras["next_state"] = _mask_goal_in_state(extras["next_state"], config)
    return transitions.replace(
        observation=_mask_goal_in_state(transitions.observation, config),
        extras=extras,
    )


def _real_rollout_goal(env_adapter: Any, env_state: Any, config: TrainConfig) -> Array:
    if config.goal_mode in XY_GOAL_MODES:
        goal = env_adapter.desired_goal(env_state)
    else:
        goal = _goal_from_obs(env_state.obs, config.goal_start, config.goal_dim)
    _assert_goal_shape(goal, config.goal_dim, context="real actor rollout")
    return goal


def _real_state_observation(env_adapter: Any, env_state: Any) -> Array:
    if hasattr(env_adapter, "_state_observation"):
        return env_adapter._state_observation(env_state)
    return jnp.asarray(env_state.obs, dtype=jnp.float32)


def _real_robot_xy(env_adapter: Any, env_state: Any) -> Array:
    if hasattr(env_adapter, "achieved_goal"):
        try:
            return env_adapter.achieved_goal(env_state)
        except ValueError:
            pass
    data = getattr(env_state, "data", None)
    if data is not None and hasattr(data, "xpos") and hasattr(env_adapter, "base_env"):
        return jnp.asarray(
            data.xpos[..., env_adapter.base_env._robot_body_id, :2],
            dtype=jnp.float32,
        )
    obs = jnp.asarray(env_state.obs, dtype=jnp.float32)
    return jnp.zeros((*obs.shape[:-1], 2), dtype=jnp.float32)


def _real_goal_xy(env_adapter: Any, env_state: Any) -> Array:
    if hasattr(env_adapter, "goal_xy"):
        return jnp.asarray(env_adapter.goal_xy(env_state), dtype=jnp.float32)
    data = getattr(env_state, "data", None)
    if data is not None and hasattr(data, "mocap_pos") and hasattr(
        env_adapter, "base_env"
    ):
        return jnp.asarray(
            data.mocap_pos[..., env_adapter.base_env._goal_mocap_id, :2],
            dtype=jnp.float32,
        )
    if hasattr(env_adapter, "desired_goal"):
        robot_xy = _real_robot_xy(env_adapter, env_state)
        desired_goal = jnp.asarray(
            env_adapter.desired_goal(env_state), dtype=jnp.float32
        )
        if desired_goal.shape[-1] == 2:
            return robot_xy + desired_goal
    raise AttributeError("real goal XY is unavailable for this environment adapter")


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
            "hazard_violation": (dist_to_hazard <= 0.20).astype(jnp.float32),
            "robot_vase_contact": jnp.zeros_like(cost, dtype=jnp.float32),
            "point_vase_contact": jnp.zeros_like(cost, dtype=jnp.float32),
            "vase_contact": jnp.zeros_like(cost, dtype=jnp.float32),
            "contact_valid": jnp.zeros_like(cost, dtype=jnp.float32),
            "vase_body_displaced": jnp.zeros_like(cost, dtype=jnp.float32),
            "vase_body_displacement_valid": jnp.zeros_like(cost, dtype=jnp.float32),
            "vase_qpos_displaced": jnp.zeros_like(cost, dtype=jnp.float32),
            "vase_qpos_displacement_valid": jnp.zeros_like(cost, dtype=jnp.float32),
            "vase_displaced": jnp.zeros_like(cost, dtype=jnp.float32),
            "vase_displacement_valid": jnp.zeros_like(cost, dtype=jnp.float32),
            "cost_residual_violation": jnp.zeros_like(cost, dtype=jnp.float32),
            "min_hazard_dist": dist_to_hazard.astype(jnp.float32),
            "min_vase_dist": jnp.full_like(cost, -1.0, dtype=jnp.float32),
            "min_obstacle_dist": dist_to_hazard.astype(jnp.float32),
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


def _mean_transition_extra(
    extras: Mapping[str, Array], key: str, reference: Array
) -> Array:
    return jnp.mean(jnp.asarray(extras.get(key, jnp.zeros_like(reference))))


def _goal_distance_metrics(goal_dist: Array) -> dict[str, Array]:
    flat = jnp.sort(jnp.ravel(goal_dist))
    max_index = flat.size - 1

    def quantile(q: float) -> Array:
        index = jnp.asarray(q * max_index, dtype=jnp.float32)
        index = jnp.floor(index).astype(jnp.int32)
        index = jnp.clip(index, 0, max_index)
        return flat[index]

    return {
        "goal_dist": jnp.mean(goal_dist),
        "goal_dist_p10": quantile(0.10),
        "goal_dist_p50": quantile(0.50),
        "goal_dist_p90": quantile(0.90),
        "goal_dist_lt_0_5": jnp.mean((goal_dist < 0.5).astype(jnp.float32)),
        "goal_dist_lt_1_0": jnp.mean((goal_dist < 1.0).astype(jnp.float32)),
        "goal_dist_lt_2_0": jnp.mean((goal_dist < 2.0).astype(jnp.float32)),
    }


def _deterministic_action(
    actor: Actor,
    actor_params: Any,
    obs: Array,
    goal: Array,
    config: TrainConfig,
) -> Array:
    mean, _ = actor.apply(actor_params, _mask_goal_in_state(obs, config), goal)
    return jnp.tanh(mean)


def _eval_action(
    actor: Actor,
    actor_params: Any,
    obs: Array,
    goal: Array,
    key: Array,
    config: TrainConfig,
    *,
    std_scale: float,
) -> Array:
    mean, log_std = actor.apply(actor_params, _mask_goal_in_state(obs, config), goal)
    if std_scale <= 0.0:
        return jnp.tanh(mean)
    noise = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    return jnp.tanh(mean + std_scale * jnp.exp(log_std) * noise)


def _eval_radius_label(radius: float) -> str:
    return str(radius)


def _parse_eval_std_scales(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in raw.replace(",", " ").split():
        value = float(part)
        if value < 0.0:
            raise ValueError("eval action std scales must be non-negative")
        if value not in values:
            values.append(value)
    return tuple(values) if values else (0.0,)


def _eval_scale_suffix(std_scale: float) -> str:
    if std_scale <= 0.0:
        return ""
    label = f"{std_scale:g}".replace(".", "_")
    return f"_std{label}"


def _suffix_eval_metrics(
    metrics: Mapping[str, Array],
    *,
    std_scale: float,
) -> dict[str, Array]:
    suffix = _eval_scale_suffix(std_scale)
    if not suffix:
        return dict(metrics)
    return {f"{key}{suffix}": value for key, value in metrics.items()}


def _eval_metrics(
    *,
    commanded_goal_dist: Array,
    initial_goal_dist: Array,
    resampled_goal_dist: Array,
    goal_reached: Array,
    cost: Array,
    frozen_goal_dist: Array | None = None,
) -> dict[str, Array]:
    commanded_goal_dist = jnp.asarray(commanded_goal_dist, dtype=jnp.float32)
    initial_goal_dist = jnp.asarray(initial_goal_dist, dtype=jnp.float32)
    resampled_goal_dist = jnp.asarray(resampled_goal_dist, dtype=jnp.float32)
    goal_reached = jnp.asarray(goal_reached, dtype=jnp.float32)
    cost = jnp.asarray(cost, dtype=jnp.float32)
    reached_bool = goal_reached > 0.5
    ever_reached = jnp.max(goal_reached, axis=0)
    first_hit_index = jnp.argmax(reached_bool, axis=0).astype(jnp.float32)
    episode_length = jnp.asarray(goal_reached.shape[0], dtype=jnp.float32)
    first_hit_time = jnp.where(ever_reached > 0.5, first_hit_index, episode_length)
    metrics = {
        "eval_ever_reached": jnp.mean(ever_reached),
        "eval_first_hit_time": jnp.mean(first_hit_time),
        "eval_min_goal_dist_initial_goal": jnp.mean(
            jnp.min(initial_goal_dist, axis=0)
        ),
        "eval_success_count": jnp.mean(jnp.sum(goal_reached, axis=0)),
        "eval_cost_return": jnp.mean(jnp.sum(cost, axis=0)),
        "eval_time_at_goal_resampled": jnp.mean(goal_reached),
        "eval_final_goal_dist_resampled": jnp.mean(resampled_goal_dist[-1]),
    }
    for radius in EVAL_PRIMARY_GOAL_RADII:
        within = (commanded_goal_dist <= radius).astype(jnp.float32)
        label = _eval_radius_label(radius)
        metrics[f"eval_ever_within_{label}"] = jnp.mean(jnp.max(within, axis=0))
    if frozen_goal_dist is not None:
        frozen_goal_dist = jnp.asarray(frozen_goal_dist, dtype=jnp.float32)
        for radius in (0.31, 0.5):
            within = (frozen_goal_dist <= radius).astype(jnp.float32)
            metrics[f"eval_frozen_time_within_{_eval_radius_label(radius)}"] = (
                jnp.mean(within)
            )
        metrics["eval_frozen_final_dist"] = jnp.mean(frozen_goal_dist[-1])
    return metrics


def make_policy_evaluator(
    objects: TrainingObjects, config: TrainConfig, *, std_scale: float = 0.0
) -> Callable[[Any, Array], Mapping[str, Array]]:
    """Builds a full-episode evaluator for deterministic or scaled-noise actions."""

    env_adapter = objects.env_adapter

    def evaluate_real(actor_params: Any, key: Array) -> Mapping[str, Array]:
        if env_adapter is None:
            raise ValueError("real-env evaluation requires objects.env_adapter")
        key, reset_key = jax.random.split(key)
        eval_state, _ = env_adapter.reset(reset_key)
        initial_goal_xy = _real_goal_xy(env_adapter, eval_state)
        initial_frozen_goal_xy = initial_goal_xy
        initial_has_hit = jnp.zeros((config.num_envs,), dtype=bool)

        def eval_step(
            carry: tuple[Any, Array, Array, Array, Array], _: Array
        ) -> tuple[tuple[Any, Array, Array, Array, Array], Mapping[str, Array]]:
            (
                env_state,
                step_key,
                first_goal_xy,
                frozen_goal_xy,
                has_hit,
            ) = carry
            step_key, action_key = jax.random.split(step_key)
            obs = _real_state_observation(env_adapter, env_state)
            goal = _real_rollout_goal(env_adapter, env_state, config)
            goal_xy_before = _real_goal_xy(env_adapter, env_state)
            action = _eval_action(
                objects.actor,
                actor_params,
                obs,
                goal,
                action_key,
                config,
                std_scale=std_scale,
            )
            next_env_state, transition = env_adapter.step(env_state, action)
            robot_xy_after = _real_robot_xy(env_adapter, next_env_state)
            reference = jnp.zeros((config.num_envs,), dtype=jnp.float32)
            cost = jnp.asarray(
                transition.extras.get("cost", reference), dtype=jnp.float32
            )
            goal_reached = jnp.asarray(
                transition.extras.get("goal_reached", reference),
                dtype=jnp.float32,
            )
            new_hit = (goal_reached > 0.5) & ~has_hit
            frozen_goal_xy = jnp.where(
                new_hit[..., None], goal_xy_before, frozen_goal_xy
            )
            has_hit = has_hit | (goal_reached > 0.5)
            frozen_measure_goal_xy = jnp.where(
                has_hit[..., None], frozen_goal_xy, goal_xy_before
            )
            commanded_goal_dist = jnp.linalg.norm(
                robot_xy_after - goal_xy_before, axis=-1
            )
            initial_goal_dist = jnp.linalg.norm(
                robot_xy_after - first_goal_xy, axis=-1
            )
            frozen_goal_dist = jnp.linalg.norm(
                robot_xy_after - frozen_measure_goal_xy, axis=-1
            )
            return (
                next_env_state,
                step_key,
                first_goal_xy,
                frozen_goal_xy,
                has_hit,
            ), {
                "commanded_goal_dist": commanded_goal_dist,
                "initial_goal_dist": initial_goal_dist,
                "resampled_goal_dist": jnp.asarray(
                    transition.extras.get("goal_dist", reference), dtype=jnp.float32
                ),
                "goal_reached": goal_reached,
                "cost": cost,
                "frozen_goal_dist": frozen_goal_dist,
            }

        _, trajectory = jax.lax.scan(
            eval_step,
            (
                eval_state,
                key,
                initial_goal_xy,
                initial_frozen_goal_xy,
                initial_has_hit,
            ),
            jnp.arange(config.env_episode_length),
        )
        return _eval_metrics(
            commanded_goal_dist=trajectory["commanded_goal_dist"],
            initial_goal_dist=trajectory["initial_goal_dist"],
            resampled_goal_dist=trajectory["resampled_goal_dist"],
            goal_reached=trajectory["goal_reached"],
            cost=trajectory["cost"],
            frozen_goal_dist=(
                trajectory["frozen_goal_dist"]
                if config.eval_freeze_goal_after_success
                else None
            ),
        )

    def evaluate_toy(actor_params: Any, key: Array) -> Mapping[str, Array]:
        key, reset_key = jax.random.split(key)
        eval_state = ToyEnvState(
            obs=0.1
            * jax.random.normal(
                reset_key,
                (config.num_envs, config.observation_dim),
                dtype=jnp.float32,
            )
        )

        def eval_step(
            carry: tuple[Array, ToyEnvState], _: Array
        ) -> tuple[tuple[Array, ToyEnvState], Mapping[str, Array]]:
            step_key, env_state = carry
            step_key, env_key, action_key = jax.random.split(step_key, 3)
            goal = jnp.zeros((config.num_envs, config.goal_dim), dtype=jnp.float32)
            action = _eval_action(
                objects.actor,
                actor_params,
                env_state.obs,
                goal,
                action_key,
                config,
                std_scale=std_scale,
            )
            next_env_state, transition = _toy_step(
                env_state, action, env_key, config
            )
            return (step_key, next_env_state), {
                "commanded_goal_dist": transition.extras["goal_dist"],
                "initial_goal_dist": transition.extras["goal_dist"],
                "resampled_goal_dist": transition.extras["goal_dist"],
                "goal_reached": transition.extras["goal_reached"],
                "cost": transition.extras["cost"],
                "frozen_goal_dist": transition.extras["goal_dist"],
            }

        _, trajectory = jax.lax.scan(
            eval_step, (key, eval_state), jnp.arange(config.env_episode_length)
        )
        return _eval_metrics(
            commanded_goal_dist=trajectory["commanded_goal_dist"],
            initial_goal_dist=trajectory["initial_goal_dist"],
            resampled_goal_dist=trajectory["resampled_goal_dist"],
            goal_reached=trajectory["goal_reached"],
            cost=trajectory["cost"],
            frozen_goal_dist=(
                trajectory["frozen_goal_dist"]
                if config.eval_freeze_goal_after_success
                else None
            ),
        )

    if env_adapter is not None:
        return jax.jit(evaluate_real)
    return jax.jit(evaluate_toy)


def make_deterministic_evaluator(
    objects: TrainingObjects, config: TrainConfig
) -> Callable[[Any, Array], Mapping[str, Array]]:
    """Builds a full-episode deterministic actor-mean evaluator."""

    return make_policy_evaluator(objects, config, std_scale=0.0)


def trace_deterministic_eval_goal_resampling(
    objects: TrainingObjects,
    config: TrainConfig,
    actor_params: Any,
    key: Array,
) -> Mapping[str, Array]:
    """Traces goal/robot positions during a deterministic real-env eval episode."""

    env_adapter = objects.env_adapter
    if env_adapter is None:
        raise ValueError("goal-resampling trace requires a real env adapter")
    eval_state, _ = env_adapter.reset(key)

    def trace_step(env_state: Any, t: Array) -> tuple[Any, Mapping[str, Array]]:
        robot_xy_before = _real_robot_xy(env_adapter, env_state)
        goal_xy_before = _real_goal_xy(env_adapter, env_state)
        obs = _real_state_observation(env_adapter, env_state)
        goal = _real_rollout_goal(env_adapter, env_state, config)
        action = _deterministic_action(objects.actor, actor_params, obs, goal, config)
        next_env_state, transition = env_adapter.step(env_state, action)

        robot_xy = _real_robot_xy(env_adapter, next_env_state)
        goal_xy = _real_goal_xy(env_adapter, next_env_state)
        reference = jnp.zeros((config.num_envs,), dtype=jnp.float32)
        goal_dist = jnp.asarray(
            transition.extras.get("goal_dist", reference), dtype=jnp.float32
        )
        goal_reached = jnp.asarray(
            transition.extras.get("goal_reached", reference), dtype=jnp.float32
        )
        old_goal_dist_after_step = jnp.linalg.norm(robot_xy - goal_xy_before, axis=-1)
        goal_shift = jnp.linalg.norm(goal_xy - goal_xy_before, axis=-1)
        return next_env_state, {
            "t": jnp.full((config.num_envs,), t, dtype=jnp.int32),
            "robot_xy": robot_xy,
            "goal_xy": goal_xy,
            "goal_dist": goal_dist,
            "goal_reached": goal_reached,
            "goal_xy_before": goal_xy_before,
            "robot_xy_before": robot_xy_before,
            "old_goal_dist_after_step": old_goal_dist_after_step,
            "goal_shift": goal_shift,
        }

    _, trace = jax.lax.scan(
        trace_step, eval_state, jnp.arange(config.env_episode_length)
    )
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), trace)


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
            _mask_goal_in_state(env_state.obs, config),
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
    goal_metrics = _goal_distance_metrics(transitions.extras["goal_dist"])
    metrics = {
        "reward": jnp.mean(transitions.reward),
        "cost": jnp.mean(transitions.extras["cost"]),
        "hard_viol": jnp.mean(transitions.extras["hard_violation"]),
        "hazard_viol": _mean_transition_extra(
            transitions.extras, "hazard_violation", transitions.extras["cost"]
        ),
        "robot_vase_contact": _mean_transition_extra(
            transitions.extras, "robot_vase_contact", transitions.extras["cost"]
        ),
        "point_vase_contact": _mean_transition_extra(
            transitions.extras, "point_vase_contact", transitions.extras["cost"]
        ),
        "vase_contact": _mean_transition_extra(
            transitions.extras, "vase_contact", transitions.extras["cost"]
        ),
        "contact_valid": _mean_transition_extra(
            transitions.extras, "contact_valid", transitions.extras["cost"]
        ),
        "vase_body_displaced": _mean_transition_extra(
            transitions.extras, "vase_body_displaced", transitions.extras["cost"]
        ),
        "vase_body_valid": _mean_transition_extra(
            transitions.extras,
            "vase_body_displacement_valid",
            transitions.extras["cost"],
        ),
        "vase_qpos_displaced": _mean_transition_extra(
            transitions.extras, "vase_qpos_displaced", transitions.extras["cost"]
        ),
        "vase_qpos_valid": _mean_transition_extra(
            transitions.extras,
            "vase_qpos_displacement_valid",
            transitions.extras["cost"],
        ),
        "vase_displaced": _mean_transition_extra(
            transitions.extras, "vase_displaced", transitions.extras["cost"]
        ),
        "vase_disp_valid": _mean_transition_extra(
            transitions.extras, "vase_displacement_valid", transitions.extras["cost"]
        ),
        "cost_residual_viol": _mean_transition_extra(
            transitions.extras, "cost_residual_violation", transitions.extras["cost"]
        ),
        "min_hazard_dist": _mean_transition_extra(
            transitions.extras, "min_hazard_dist", transitions.extras["cost"]
        ),
        "min_vase_dist": _mean_transition_extra(
            transitions.extras, "min_vase_dist", transitions.extras["cost"]
        ),
        "min_obstacle_dist": _mean_transition_extra(
            transitions.extras, "min_obstacle_dist", transitions.extras["cost"]
        ),
        **goal_metrics,
        "goal_reached": jnp.mean(transitions.extras["goal_reached"]),
        "goal_slice_mean": jnp.mean(rollout_goals),
        "goal_slice_std": jnp.std(rollout_goals),
        "goal_slice_min": jnp.min(rollout_goals),
        "goal_slice_max": jnp.max(rollout_goals),
        "cost_zero_action": jnp.asarray(0.0, dtype=jnp.float32),
        "cost_neg_action": jnp.asarray(0.0, dtype=jnp.float32),
        "cost_action_minus_zero": jnp.asarray(0.0, dtype=jnp.float32),
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
        goal = _real_rollout_goal(env_adapter, env_state, config)
        sample = sample_tanh_gaussian(
            objects.actor,
            train_state.actor_params,
            _mask_goal_in_state(
                _real_state_observation(env_adapter, env_state), config
            ),
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

    initial_obs = _real_state_observation(env_adapter, train_state.env_state)
    observations = jnp.concatenate(
        [
            initial_obs[None, ...],
            transitions.extras["next_state"],
        ],
        axis=0,
    )
    if config.goal_mode in XY_GOAL_MODES:
        rollout_goals = transitions.extras["desired_goal"]
    else:
        rollout_goals = _goal_from_obs(
            observations[:-1], config.goal_start, config.goal_dim
        )
    _assert_goal_shape(
        rollout_goals, config.goal_dim, context="real actor rollout metrics"
    )
    if config.probe_counterfactual_costs:
        cost_zero_action = transitions.extras["cost_zero_action"]
        cost_neg_action = transitions.extras["cost_neg_action"]
    else:
        cost_zero_action = jnp.zeros_like(transitions.extras["cost"])
        cost_neg_action = jnp.zeros_like(transitions.extras["cost"])
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
    goal_metrics = _goal_distance_metrics(transitions.extras["goal_dist"])
    metrics = {
        "reward": jnp.mean(transitions.reward),
        "cost": jnp.mean(transitions.extras["cost"]),
        "hard_viol": jnp.mean(transitions.extras["hard_violation"]),
        "hazard_viol": _mean_transition_extra(
            transitions.extras, "hazard_violation", transitions.extras["cost"]
        ),
        "robot_vase_contact": _mean_transition_extra(
            transitions.extras, "robot_vase_contact", transitions.extras["cost"]
        ),
        "point_vase_contact": _mean_transition_extra(
            transitions.extras, "point_vase_contact", transitions.extras["cost"]
        ),
        "vase_contact": _mean_transition_extra(
            transitions.extras, "vase_contact", transitions.extras["cost"]
        ),
        "contact_valid": _mean_transition_extra(
            transitions.extras, "contact_valid", transitions.extras["cost"]
        ),
        "vase_body_displaced": _mean_transition_extra(
            transitions.extras, "vase_body_displaced", transitions.extras["cost"]
        ),
        "vase_body_valid": _mean_transition_extra(
            transitions.extras,
            "vase_body_displacement_valid",
            transitions.extras["cost"],
        ),
        "vase_qpos_displaced": _mean_transition_extra(
            transitions.extras, "vase_qpos_displaced", transitions.extras["cost"]
        ),
        "vase_qpos_valid": _mean_transition_extra(
            transitions.extras,
            "vase_qpos_displacement_valid",
            transitions.extras["cost"],
        ),
        "vase_displaced": _mean_transition_extra(
            transitions.extras, "vase_displaced", transitions.extras["cost"]
        ),
        "vase_disp_valid": _mean_transition_extra(
            transitions.extras, "vase_displacement_valid", transitions.extras["cost"]
        ),
        "cost_residual_viol": _mean_transition_extra(
            transitions.extras, "cost_residual_violation", transitions.extras["cost"]
        ),
        "min_hazard_dist": _mean_transition_extra(
            transitions.extras, "min_hazard_dist", transitions.extras["cost"]
        ),
        "min_vase_dist": _mean_transition_extra(
            transitions.extras, "min_vase_dist", transitions.extras["cost"]
        ),
        "min_obstacle_dist": _mean_transition_extra(
            transitions.extras, "min_obstacle_dist", transitions.extras["cost"]
        ),
        **goal_metrics,
        "goal_reached": jnp.mean(transitions.extras["goal_reached"]),
        "goal_slice_mean": jnp.mean(rollout_goals),
        "goal_slice_std": jnp.std(rollout_goals),
        "goal_slice_min": jnp.min(rollout_goals),
        "goal_slice_max": jnp.max(rollout_goals),
        "cost_zero_action": jnp.mean(cost_zero_action),
        "cost_neg_action": jnp.mean(cost_neg_action),
        "cost_action_minus_zero": jnp.mean(
            transitions.extras["cost"] - cost_zero_action
        ),
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
    (
        key,
        sample_key,
        cost_sample_key,
        actor_key,
        cost_key,
        alpha_key,
        dual_key,
    ) = jax.random.split(train_state.key, 7)
    batch = sample_hindsight_transitions(
        train_state.replay,
        sample_key,
        batch_size=config.batch_size,
        goal_start=config.goal_start,
        goal_end=config.goal_start + config.goal_dim,
        relative_goal=config.goal_mode == "relative_xy",
        cost_return_gamma=config.gamma_c,
    )
    _assert_goal_shape(
        batch.extras["goal"], config.goal_dim, context="hindsight critic"
    )
    batch = _mask_transition_state_inputs(batch, config)
    if config.cost_risk_replay_ratio > 0.0:
        cost_batch, cost_risk_aux = sample_risk_biased_hindsight_transitions(
            train_state.replay,
            cost_sample_key,
            batch_size=config.batch_size,
            risk_ratio=config.cost_risk_replay_ratio,
            hazard_lidar_threshold=config.cost_risk_hazard_lidar_thresh,
            min_fraction_available=config.cost_risk_min_fraction_available,
            goal_start=config.goal_start,
            goal_end=config.goal_start + config.goal_dim,
            relative_goal=config.goal_mode == "relative_xy",
            cost_return_gamma=config.gamma_c,
        )
        _assert_goal_shape(
            cost_batch.extras["goal"], config.goal_dim, context="cost critic"
        )
        cost_batch = _mask_transition_state_inputs(cost_batch, config)
    else:
        batch_hard = jnp.asarray(batch.extras["hard_violation"], dtype=jnp.float32)
        batch_cost = jnp.asarray(batch.extras["cost"], dtype=jnp.float32)
        cost_batch = batch
        cost_risk_aux = {
            "cost_risk_replay_ratio_actual": jnp.asarray(0.0, dtype=jnp.float32),
            "cost_risky_batch_frac": jnp.mean(
                ((batch_hard > 0.5) | (batch_cost > 0.0)).astype(jnp.float32)
            ),
            "cost_risky_available_frac": replay_risky_available_fraction(
                train_state.replay,
                hazard_lidar_threshold=config.cost_risk_hazard_lidar_thresh,
            ),
            "cost_risky_batch_mean_cost": jnp.asarray(0.0, dtype=jnp.float32),
            "cost_uniform_batch_mean_cost": jnp.mean(batch_cost),
        }

    def critic_objective(params: Any) -> tuple[Array, dict[str, Array]]:
        return critic_loss_fn(
            params,
            batch,
            sa_encoder=objects.sa_encoder,
            g_encoder=objects.g_encoder,
            tau=config.tau,
            rho=config.rho,
            score_mode=config.critic_score_mode,
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
            score_mode=config.critic_score_mode,
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
            cost_batch,
            cost_key,
            actor=objects.actor,
            cost_critic=objects.cost_critic,
            gamma_c=config.gamma_c,
            cost_return_loss_weight=config.cost_return_loss_weight,
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
        train_state.log_alpha,
        sample.log_prob,
        objects.action_dim,
        config.entropy_param,
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
        integral_decay=config.pid_integral_decay,
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
        "cc_td_loss": cc_aux["cost_critic_td_loss"],
        "cc_return_loss": cc_aux["cost_return_loss"],
        "cost_risk_replay_ratio_actual": cost_risk_aux[
            "cost_risk_replay_ratio_actual"
        ],
        "cost_risky_batch_frac": cost_risk_aux["cost_risky_batch_frac"],
        "cost_risky_available_frac": cost_risk_aux["cost_risky_available_frac"],
        "cost_risky_batch_mean_cost": cost_risk_aux[
            "cost_risky_batch_mean_cost"
        ],
        "cost_uniform_batch_mean_cost": cost_risk_aux[
            "cost_uniform_batch_mean_cost"
        ],
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
        "reward_actor_term": a_aux["reward_actor_term_mean"],
        "qc_actor": a_aux["qc_actor_mean"],
        "qc_zero_action_actor": a_aux["qc_zero_action_mean"],
        "qc_neg_action_actor": a_aux["qc_neg_action_mean"],
        "qc_action_delta_actor": a_aux["qc_action_delta_mean"],
        "qc_action_gap_actor": a_aux["qc_action_gap_mean"],
        "qc_action_delta_frac_pos_actor": a_aux["qc_action_delta_frac_pos"],
        "qc_actor_std": a_aux["qc_actor_std"],
        "lambda_qc_actor": a_aux["constraint_term_mean"],
        "grad_norm_qr_actor": a_aux["grad_norm_qr_a"],
        "grad_norm_qc_actor": a_aux["grad_norm_qc_a"],
        "lambda_grad_norm_qc_actor": a_aux["lambda_grad_norm_qc_a"],
        "grad_ratio_cost_reward_actor": a_aux["grad_ratio_cost_reward"],
        "cosine_grad_qr_qc_actor": a_aux["cosine_grad_qr_qc"],
        "risk_condition_frac_actor": a_aux["risk_condition_fraction"],
        "qc_actor_risky": a_aux["qc_actor_risky_mean"],
        "qc_action_delta_risky_actor": a_aux["qc_action_delta_risky_mean"],
        "grad_ratio_cost_reward_risky_actor": a_aux[
            "grad_ratio_cost_reward_risky"
        ],
        "actor_qc_rank_mean": a_aux["actor_qc_rank_mean"],
        "actor_qc_percentile": a_aux["actor_qc_percentile"],
        "q_c_action_spread": a_aux["q_c_action_spread"],
        "best_qc_action_is_actor_frac": a_aux["best_qc_action_is_actor_frac"],
        "action_rank_hazard_available_frac": a_aux[
            "action_rank_hazard_available_frac"
        ],
        "actor_qc_rank_mean_risk1": a_aux["actor_qc_rank_mean_risk1"],
        "actor_qc_percentile_risk1": a_aux["actor_qc_percentile_risk1"],
        "q_c_action_spread_risk1": a_aux["q_c_action_spread_risk1"],
        "best_qc_action_is_actor_frac_risk1": a_aux[
            "best_qc_action_is_actor_frac_risk1"
        ],
        "actor_qc_rank_mean_risk05": a_aux["actor_qc_rank_mean_risk05"],
        "actor_qc_percentile_risk05": a_aux["actor_qc_percentile_risk05"],
        "q_c_action_spread_risk05": a_aux["q_c_action_spread_risk05"],
        "best_qc_action_is_actor_frac_risk05": a_aux[
            "best_qc_action_is_actor_frac_risk05"
        ],
        "actor_qc_rank_mean_risk025": a_aux["actor_qc_rank_mean_risk025"],
        "actor_qc_percentile_risk025": a_aux["actor_qc_percentile_risk025"],
        "q_c_action_spread_risk025": a_aux["q_c_action_spread_risk025"],
        "best_qc_action_is_actor_frac_risk025": a_aux[
            "best_qc_action_is_actor_frac_risk025"
        ],
        "nu_c": jnp.asarray(config.nu_c, dtype=jnp.float32),
        "entropy_param": jnp.asarray(config.entropy_param, dtype=jnp.float32),
        "target_entropy": -jnp.asarray(
            config.entropy_param * objects.action_dim, dtype=jnp.float32
        ),
        "score_mode_l2": jnp.asarray(
            config.critic_score_mode == "l2", dtype=jnp.float32
        ),
        "alpha_clip": jnp.minimum(jnp.exp(log_alpha) / config.alpha_max, 1.0),
        "cost": cc_aux["mean_cost"],
        "qc": cc_aux["mean_qc"],
        "td_target": cc_aux["mean_target"],
        "cost_return": cc_aux["mean_cost_return"],
        "qc_return_error": cc_aux["qc_return_error"],
        "cost_return_loss_weight": jnp.asarray(
            config.cost_return_loss_weight, dtype=jnp.float32
        ),
        "jc_hat": jc_hat,
        "dual_qc_mean": dual_aux["dual_qc_mean"],
        "cost_limit": jnp.asarray(config.cost_limit, dtype=jnp.float32),
        "state_goal_masked": jnp.asarray(config.mask_goal_in_state, dtype=jnp.float32),
        "pid_error": pid_error,
        "pid_integral": pid_state.integral,
        "pid_integral_decay": jnp.asarray(
            config.pid_integral_decay, dtype=jnp.float32
        ),
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
        metrics["hazard_viol"] = collect_metrics["hazard_viol"]
        metrics["robot_vase_contact"] = collect_metrics["robot_vase_contact"]
        metrics["point_vase_contact"] = collect_metrics["point_vase_contact"]
        metrics["vase_contact"] = collect_metrics["vase_contact"]
        metrics["contact_valid"] = collect_metrics["contact_valid"]
        metrics["vase_body_displaced"] = collect_metrics["vase_body_displaced"]
        metrics["vase_body_valid"] = collect_metrics["vase_body_valid"]
        metrics["vase_qpos_displaced"] = collect_metrics["vase_qpos_displaced"]
        metrics["vase_qpos_valid"] = collect_metrics["vase_qpos_valid"]
        metrics["vase_displaced"] = collect_metrics["vase_displaced"]
        metrics["vase_disp_valid"] = collect_metrics["vase_disp_valid"]
        metrics["cost_residual_viol"] = collect_metrics["cost_residual_viol"]
        metrics["min_hazard_dist"] = collect_metrics["min_hazard_dist"]
        metrics["min_vase_dist"] = collect_metrics["min_vase_dist"]
        metrics["min_obstacle_dist"] = collect_metrics["min_obstacle_dist"]
        for key in GOAL_DISTANCE_METRIC_KEYS:
            metrics[key] = collect_metrics[key]
        metrics["goal_reached"] = collect_metrics["goal_reached"]
        metrics["goal_start"] = jnp.asarray(config.goal_start, dtype=jnp.float32)
        metrics["goal_dim"] = jnp.asarray(config.goal_dim, dtype=jnp.float32)
        metrics["goal_mode_xy"] = jnp.asarray(
            config.goal_mode in XY_GOAL_MODES, dtype=jnp.float32
        )
        metrics["goal_mode_relative"] = jnp.asarray(
            config.goal_mode == "relative_xy", dtype=jnp.float32
        )
        metrics["native_goal_lidar_masked"] = jnp.asarray(
            config.mask_native_goal_lidar, dtype=jnp.float32
        )
        metrics["score_mode_l2"] = jnp.asarray(
            config.critic_score_mode == "l2", dtype=jnp.float32
        )
        metrics["goal_slice_mean"] = collect_metrics["goal_slice_mean"]
        metrics["goal_slice_std"] = collect_metrics["goal_slice_std"]
        metrics["goal_slice_min"] = collect_metrics["goal_slice_min"]
        metrics["goal_slice_max"] = collect_metrics["goal_slice_max"]
        metrics["cost_zero_action"] = collect_metrics["cost_zero_action"]
        metrics["cost_neg_action"] = collect_metrics["cost_neg_action"]
        metrics["cost_action_minus_zero"] = collect_metrics["cost_action_minus_zero"]
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

    if config.goal_mode not in {"obs_slice", *XY_GOAL_MODES}:
        raise ValueError("goal_mode must be 'obs_slice', 'xy', or 'relative_xy'")
    if config.goal_mode in XY_GOAL_MODES and config.goal_dim != 2:
        raise ValueError("xy goal modes require goal_dim=2")
    if config.critic_score_mode not in {"cosine", "l2"}:
        raise ValueError("critic_score_mode must be 'cosine' or 'l2'")
    if not 0.0 <= config.cost_risk_replay_ratio <= 1.0:
        raise ValueError("cost_risk_replay_ratio must be in [0, 1]")
    if config.cost_risk_hazard_lidar_thresh <= 0.0:
        raise ValueError("cost_risk_hazard_lidar_thresh must be positive")
    if config.cost_risk_min_fraction_available < 0.0:
        raise ValueError("cost_risk_min_fraction_available must be non-negative")
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
            goal_mode=config.goal_mode,
            mask_native_goal_lidar=config.mask_native_goal_lidar,
            probe_counterfactual_costs=config.probe_counterfactual_costs,
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


def save_actor_checkpoint(path: str | Path, state: TrainState) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialization.to_bytes(state.actor_params))
    return output


def load_actor_checkpoint(path: str | Path, target_actor_params: Any) -> Any:
    return serialization.from_bytes(target_actor_params, Path(path).read_bytes())


def _mean_float(metrics: Mapping[str, Array], key: str) -> float:
    return float(jnp.mean(metrics[key]))


def _max_flag(metrics: Mapping[str, Array], key: str) -> int:
    return int(jnp.max(metrics[key]))


def _format_eval_metrics_line(metrics: Mapping[str, Array]) -> str:
    line = (
        "         "
        f"eval_ever_reached={_mean_float(metrics, 'eval_ever_reached'):.4f} "
        f"eval_first_hit_time={_mean_float(metrics, 'eval_first_hit_time'):.2f} "
        f"eval_min_goal_dist_initial_goal="
        f"{_mean_float(metrics, 'eval_min_goal_dist_initial_goal'):.4f} "
        f"eval_ever_within_0.31="
        f"{_mean_float(metrics, 'eval_ever_within_0.31'):.4f} "
        f"eval_ever_within_0.5="
        f"{_mean_float(metrics, 'eval_ever_within_0.5'):.4f} "
        f"eval_ever_within_1.0="
        f"{_mean_float(metrics, 'eval_ever_within_1.0'):.4f} "
        f"eval_ever_within_2.0="
        f"{_mean_float(metrics, 'eval_ever_within_2.0'):.4f} "
        f"eval_success_count={_mean_float(metrics, 'eval_success_count'):.4f} "
        f"eval_cost_return={_mean_float(metrics, 'eval_cost_return'):.4f} "
        f"eval_time_at_goal_resampled="
        f"{_mean_float(metrics, 'eval_time_at_goal_resampled'):.4f} "
        f"eval_final_goal_dist_resampled="
        f"{_mean_float(metrics, 'eval_final_goal_dist_resampled'):.4f}"
    )
    if "eval_frozen_final_dist" in metrics:
        line += (
            " "
            f"eval_frozen_time_within_0.31="
            f"{_mean_float(metrics, 'eval_frozen_time_within_0.31'):.4f} "
            f"eval_frozen_time_within_0.5="
            f"{_mean_float(metrics, 'eval_frozen_time_within_0.5'):.4f} "
            f"eval_frozen_final_dist="
            f"{_mean_float(metrics, 'eval_frozen_final_dist'):.4f}"
        )
    mode_lines: list[str] = []
    mode_suffixes = sorted(
        key.removeprefix("eval_ever_reached")
        for key in metrics
        if key.startswith("eval_ever_reached_std")
    )
    for suffix in mode_suffixes:
        mode_lines.append(
            "         "
            f"eval_ever_reached{suffix}="
            f"{_mean_float(metrics, f'eval_ever_reached{suffix}'):.4f} "
            f"eval_first_hit_time{suffix}="
            f"{_mean_float(metrics, f'eval_first_hit_time{suffix}'):.2f} "
            f"eval_min_goal_dist_initial_goal{suffix}="
            f"{_mean_float(metrics, f'eval_min_goal_dist_initial_goal{suffix}'):.4f} "
            f"eval_ever_within_0.31{suffix}="
            f"{_mean_float(metrics, f'eval_ever_within_0.31{suffix}'):.4f} "
            f"eval_success_count{suffix}="
            f"{_mean_float(metrics, f'eval_success_count{suffix}'):.4f} "
            f"eval_cost_return{suffix}="
            f"{_mean_float(metrics, f'eval_cost_return{suffix}'):.4f}"
        )
    if mode_lines:
        line = "\n".join([line, *mode_lines])
    return line


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
                f"hazard={_mean_float(metrics, 'hazard_viol'):.4f} "
                f"vase_contact={_mean_float(metrics, 'vase_contact'):.4f} "
                f"robot_vase={_mean_float(metrics, 'robot_vase_contact'):.4f} "
                f"point_vase={_mean_float(metrics, 'point_vase_contact'):.4f} "
                f"contact_valid={_mean_float(metrics, 'contact_valid'):.0f} "
                f"vase_body={_mean_float(metrics, 'vase_body_displaced'):.4f} "
                f"body_valid={_mean_float(metrics, 'vase_body_valid'):.0f} "
                f"vase_qpos={_mean_float(metrics, 'vase_qpos_displaced'):.4f} "
                f"qpos_valid={_mean_float(metrics, 'vase_qpos_valid'):.0f} "
                f"vase_disp={_mean_float(metrics, 'vase_displaced'):.4f} "
                f"vase_valid={_mean_float(metrics, 'vase_disp_valid'):.0f} "
                f"cost_resid={_mean_float(metrics, 'cost_residual_viol'):.4f} "
                f"min_haz={_mean_float(metrics, 'min_hazard_dist'):.3f} "
                f"min_vase={_mean_float(metrics, 'min_vase_dist'):.3f} "
                f"min_obs={_mean_float(metrics, 'min_obstacle_dist'):.3f} "
                f"cost0={_mean_float(metrics, 'cost_zero_action'):.4f} "
                f"cost-={_mean_float(metrics, 'cost_neg_action'):.4f} "
                f"cost-cost0={_mean_float(metrics, 'cost_action_minus_zero'):.4f} "
                f"rew={_mean_float(metrics, 'rollout_reward'):.4f} "
                f"gdist={_mean_float(metrics, 'goal_dist'):.4f} "
                f"g_p10={_mean_float(metrics, 'goal_dist_p10'):.4f} "
                f"g_p50={_mean_float(metrics, 'goal_dist_p50'):.4f} "
                f"g_p90={_mean_float(metrics, 'goal_dist_p90'):.4f} "
                f"g_lt0_5={_mean_float(metrics, 'goal_dist_lt_0_5'):.4f} "
                f"g_lt1={_mean_float(metrics, 'goal_dist_lt_1_0'):.4f} "
                f"g_lt2={_mean_float(metrics, 'goal_dist_lt_2_0'):.4f} "
                f"reached={_mean_float(metrics, 'goal_reached'):.4f} "
                f"gstart={_mean_float(metrics, 'goal_start'):.0f} "
                f"gdim={_mean_float(metrics, 'goal_dim'):.0f} "
                f"gxy={_mean_float(metrics, 'goal_mode_xy'):.0f} "
                f"grel={_mean_float(metrics, 'goal_mode_relative'):.0f} "
                f"glmask={_mean_float(metrics, 'native_goal_lidar_masked'):.0f} "
                f"score_l2={_mean_float(metrics, 'score_mode_l2'):.0f} "
                f"gmean={_mean_float(metrics, 'goal_slice_mean'):.3f} "
                f"gstd={_mean_float(metrics, 'goal_slice_std'):.3f} "
                f"gmin={_mean_float(metrics, 'goal_slice_min'):.3f} "
                f"gmax={_mean_float(metrics, 'goal_slice_max'):.3f} "
                f"gmask={_mean_float(metrics, 'state_goal_masked'):.0f} "
                f"λ̃={_mean_float(metrics, 'lambda_tilde'):.4f} "
                f"Ĵ_c={_mean_float(metrics, 'jc_hat'):.4f} "
                f"Qc={_mean_float(metrics, 'qc'):.4f} "
                f"TD={_mean_float(metrics, 'td_target'):.4f} "
                f"limit={_mean_float(metrics, 'cost_limit'):.2e} "
                f"pid_err={_mean_float(metrics, 'pid_error'):.2e} "
                f"S={_mean_float(metrics, 'pid_integral'):.2e} "
                f"Sdecay={_mean_float(metrics, 'pid_integral_decay'):.2f} "
                f"λraw={_mean_float(metrics, 'pid_raw_lambda'):.2e} "
                f"Qc_a={_mean_float(metrics, 'qc_actor'):.4f} "
                f"Qc0={_mean_float(metrics, 'qc_zero_action_actor'):.4f} "
                f"Qc-={_mean_float(metrics, 'qc_neg_action_actor'):.4f} "
                f"Jc_mc={_mean_float(metrics, 'cost_return'):.4f} "
                f"Qc-Jc={_mean_float(metrics, 'qc_return_error'):.4f} "
                f"mcw={_mean_float(metrics, 'cost_return_loss_weight'):.1e} "
                f"ΔQc_a0={_mean_float(metrics, 'qc_action_delta_actor'):.2e} "
                f"dQc_a0={_mean_float(metrics, 'qc_action_delta_actor'):.2e} "
                f"abs_dQc_a0={_mean_float(metrics, 'qc_action_gap_actor'):.2e} "
                f"frac_dQc_pos="
                f"{_mean_float(metrics, 'qc_action_delta_frac_pos_actor'):.3f} "
                f"Qcstd={_mean_float(metrics, 'qc_actor_std'):.2e} "
                f"λQc_a={_mean_float(metrics, 'lambda_qc_actor'):.2e} "
                f"r_term={_mean_float(metrics, 'reward_actor_term'):.3f} "
                f"dQr_da={_mean_float(metrics, 'grad_norm_qr_actor'):.2e} "
                f"dQc_da={_mean_float(metrics, 'grad_norm_qc_actor'):.2e} "
                f"lambda_dQc_da="
                f"{_mean_float(metrics, 'lambda_grad_norm_qc_actor'):.2e} "
                f"grad_ratio="
                f"{_mean_float(metrics, 'grad_ratio_cost_reward_actor'):.2e} "
                f"cos_qr_qc="
                f"{_mean_float(metrics, 'cosine_grad_qr_qc_actor'):.3f} "
                f"risk_frac={_mean_float(metrics, 'risk_condition_frac_actor'):.3f} "
                f"Qc_risk={_mean_float(metrics, 'qc_actor_risky'):.4f} "
                f"dQc_risk="
                f"{_mean_float(metrics, 'qc_action_delta_risky_actor'):.2e} "
                f"grad_ratio_risk="
                f"{_mean_float(metrics, 'grad_ratio_cost_reward_risky_actor'):.2e} "
                f"nu_c={_mean_float(metrics, 'nu_c'):.1e}"
            ),
            (
                "         "
                f"action_rank[ actor_qc_rank_mean="
                f"{_mean_float(metrics, 'actor_qc_rank_mean'):.2f} "
                f"actor_qc_percentile="
                f"{_mean_float(metrics, 'actor_qc_percentile'):.3f} "
                f"q_c_action_spread="
                f"{_mean_float(metrics, 'q_c_action_spread'):.2e} "
                f"best_qc_action_is_actor_frac="
                f"{_mean_float(metrics, 'best_qc_action_is_actor_frac'):.3f} "
                f"hazard_rank_bins_available="
                f"{_mean_float(metrics, 'action_rank_hazard_available_frac'):.3f} "
                f"actor_qc_rank_mean_risk1="
                f"{_mean_float(metrics, 'actor_qc_rank_mean_risk1'):.2f} "
                f"actor_qc_percentile_risk1="
                f"{_mean_float(metrics, 'actor_qc_percentile_risk1'):.3f} "
                f"q_c_action_spread_risk1="
                f"{_mean_float(metrics, 'q_c_action_spread_risk1'):.2e} "
                f"actor_qc_rank_mean_risk05="
                f"{_mean_float(metrics, 'actor_qc_rank_mean_risk05'):.2f} "
                f"actor_qc_percentile_risk05="
                f"{_mean_float(metrics, 'actor_qc_percentile_risk05'):.3f} "
                f"q_c_action_spread_risk05="
                f"{_mean_float(metrics, 'q_c_action_spread_risk05'):.2e} "
                f"actor_qc_rank_mean_risk025="
                f"{_mean_float(metrics, 'actor_qc_rank_mean_risk025'):.2f} "
                f"actor_qc_percentile_risk025="
                f"{_mean_float(metrics, 'actor_qc_percentile_risk025'):.3f} "
                f"q_c_action_spread_risk025="
                f"{_mean_float(metrics, 'q_c_action_spread_risk025'):.2e}]"
            ),
            (
                "         "
                f"cost_replay[ cost_risk_replay_ratio_actual="
                f"{_mean_float(metrics, 'cost_risk_replay_ratio_actual'):.3f} "
                f"cost_risky_batch_frac="
                f"{_mean_float(metrics, 'cost_risky_batch_frac'):.3f} "
                f"cost_risky_available_frac="
                f"{_mean_float(metrics, 'cost_risky_available_frac'):.3f} "
                f"cost_risky_batch_mean_cost="
                f"{_mean_float(metrics, 'cost_risky_batch_mean_cost'):.4f} "
                f"cost_uniform_batch_mean_cost="
                f"{_mean_float(metrics, 'cost_uniform_batch_mean_cost'):.4f}]"
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
                f"H={-_mean_float(metrics, 'log_prob_actor'):.3f} "
                f"H_tgt={-_mean_float(metrics, 'target_entropy'):.3f} "
                f"α·log_p={_mean_float(metrics, 'alpha_logprob_actor'):.3f} "
                f"gauss_lp={_mean_float(metrics, 'gaussian_logp_actor'):.3f} "
                f"sat_corr={_mean_float(metrics, 'sat_correction_actor'):.3f} "
                f"log_std={_mean_float(metrics, 'log_std_mean_actor'):.3f} "
                f"f_term={_mean_float(metrics, 'f_term_actor'):.3f} "
                f"α_clip={_mean_float(metrics, 'alpha_clip'):.2f}]"
            ),
            _format_eval_metrics_line(metrics),
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
    eval_std_scales = _parse_eval_std_scales(config.eval_action_std_scales)
    policy_evaluators = tuple(
        (std_scale, make_policy_evaluator(objects, config, std_scale=std_scale))
        for std_scale in eval_std_scales
    )
    eval_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), 0x5EED)

    last_metrics: Mapping[str, Array] | None = None
    for epoch in range(config.epochs):
        start = time.perf_counter()
        state, metrics = training_epoch(state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        elapsed = time.perf_counter() - start
        merged_eval_metrics: dict[str, Array] = {}
        for std_scale, policy_evaluator in policy_evaluators:
            eval_key, epoch_eval_key = jax.random.split(eval_key)
            eval_metrics = policy_evaluator(state.actor_params, epoch_eval_key)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), eval_metrics)
            merged_eval_metrics.update(
                _suffix_eval_metrics(eval_metrics, std_scale=std_scale)
            )
        metrics = {**metrics, **merged_eval_metrics}
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

    if config.checkpoint_output:
        output_path = save_actor_checkpoint(config.checkpoint_output, state)
        print_fn(f"CHECKPOINT_OUTPUT={output_path}")

    return {
        "state": state,
        "metrics": last_metrics,
        "epochs": config.epochs,
    }
