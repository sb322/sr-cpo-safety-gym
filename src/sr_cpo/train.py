"""Top-level SR-CPO training loop.

The local smoke path uses a tiny deterministic JAX toy dynamics model so the
whole algorithm can be exercised on CPU without safe-learning installed. The
losses, replay buffer, dual update, target-network update, and probe formatting
are the same code path used by the production runner.
"""

from __future__ import annotations

import os
import time
import warnings
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
from sr_cpo.rank_buffer import (
    RankBuffer,
    insert_rank_examples,
    make_rank_buffer,
    sample_rank_batch,
    save_rank_buffer_npz,
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
    eval_counterfactual_action_probes: bool = False
    counterfactual_probe_random_actions: int = 16
    counterfactual_probe_perturb_actions: int = 16
    counterfactual_probe_perturb_std: float = 0.1
    enable_multistep_counterfactual_probes: bool = False
    counterfactual_probe_horizons: str = "5,10,20"
    counterfactual_probe_max_states: int = 0
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
    cost_mode: str = "sparse"
    cost_dense_prox_tau: float = 0.5
    cost_return_loss_weight: float = 0.0
    cost_rank_loss_weight: float = 0.0
    cost_rank_horizon: int = 50
    cost_rank_num_candidates: int = 8
    cost_rank_states_per_epoch: int = 8
    cost_rank_buffer_capacity: int = 256
    cost_rank_batch_size: int = 32
    # Rank candidates stay mostly actor-local so pairwise labels constrain
    # dQ_c/da near the actor; a few uniform anchors keep broad ordering signal.
    cost_rank_candidate_perturb_std: float = 0.075
    cost_rank_uniform_anchor_count: int = 2
    cost_rank_uniform_random_frac: float = -1.0
    cost_rank_label_epsilon: float = 0.0
    cost_rank_min_label_spread: float = 1e-4
    cost_rank_label_kind: str = "dense"
    cost_rank_done_mode: str = "extend"
    cost_rank_debug_dump: bool = False
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
    rank_buffer: RankBuffer
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


def _env_state_info_value(
    env_state: Any,
    key: str,
    batch_shape: tuple[int, ...],
    *,
    dtype: Any = jnp.float32,
) -> Array:
    info = getattr(env_state, "info", None)
    if isinstance(info, Mapping) and key in info:
        return jnp.asarray(info[key], dtype=dtype)
    return jnp.zeros(batch_shape, dtype=dtype)


def _env_state_done(env_state: Any, batch_shape: tuple[int, ...]) -> Array:
    done = getattr(env_state, "done", None)
    if done is None:
        return jnp.zeros(batch_shape, dtype=jnp.float32)
    return jnp.asarray(done, dtype=jnp.float32)


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


def _take_env_state(env_state: Any, indices: Array, batch_size: int) -> Any:
    def take_leaf(leaf: Any) -> Any:
        if (
            hasattr(leaf, "shape")
            and len(leaf.shape) > 0
            and leaf.shape[0] == batch_size
        ):
            return leaf[indices]
        return leaf

    return jax.tree_util.tree_map(take_leaf, env_state)


def _repeat_env_state(env_state: Any, repeats: int, batch_size: int) -> Any:
    def repeat_leaf(leaf: Any) -> Any:
        if (
            hasattr(leaf, "shape")
            and len(leaf.shape) > 0
            and leaf.shape[0] == batch_size
        ):
            return jnp.repeat(leaf, repeats, axis=0)
        return leaf

    return jax.tree_util.tree_map(repeat_leaf, env_state)


def _flatten_env_state_history(env_states: Any, time_size: int, env_size: int) -> Any:
    """Flattens scan-stacked env states from ``[T, B, ...]`` to ``[T*B, ...]``."""

    def flatten_leaf(leaf: Any) -> Any:
        if (
            hasattr(leaf, "shape")
            and len(leaf.shape) >= 2
            and leaf.shape[0] == time_size
            and leaf.shape[1] == env_size
        ):
            return leaf.reshape((time_size * env_size, *leaf.shape[2:]))
        return leaf

    return jax.tree_util.tree_map(flatten_leaf, env_states)


def _rank_state_history_mask(discounts: Array, horizon: int) -> Array:
    """Marks source rollout states with a full in-window actor-alive lookahead.

    The mask is diagnostic only: rank labels are generated by fresh env rollouts,
    not by reusing this source rollout.  ``discounts`` comes from
    ``Transition.discount`` and is derived from ``next_state.done`` in the real
    env adapter, not from ``info["truncation"]``.  A source state at index ``t``
    is counted only when transitions ``[t, t + horizon)`` are alive and the
    source history also contains the state at ``t + horizon``; for
    ``unroll_length=62, horizon=20`` this leaves 42 eligible starts when there
    are no dones.
    """

    alive = jnp.asarray(discounts, dtype=jnp.float32) > 0.5
    time_size = alive.shape[0]
    window_alive = jnp.ones_like(alive, dtype=bool)
    for offset in range(min(horizon, time_size)):
        shifted = jnp.concatenate(
            (
                alive[offset:],
                jnp.zeros((offset, *alive.shape[1:]), dtype=bool),
            ),
            axis=0,
        )
        window_alive = jnp.logical_and(window_alive, shifted)
    enough_future = jnp.arange(time_size)[:, None] < (time_size - horizon)
    return jnp.logical_and(window_alive, enough_future)


def _deterministic_actor_action(
    actor: Actor,
    actor_params: Any,
    observation: Array,
    goal: Array,
) -> Array:
    mean, _ = actor.apply(actor_params, observation, goal)
    return jnp.tanh(mean)


def _rank_candidate_actions(
    actor_action: Array,
    key: Array,
    config: TrainConfig,
    action_dim: int,
) -> Array:
    num_candidates = config.cost_rank_num_candidates
    if config.cost_rank_uniform_random_frac >= 0.0:
        warnings.warn(
            "cost_rank_uniform_random_frac is deprecated; use "
            "cost_rank_uniform_anchor_count instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        num_uniform = int(round(num_candidates * config.cost_rank_uniform_random_frac))
    else:
        num_uniform = config.cost_rank_uniform_anchor_count
    num_uniform = min(max(num_uniform, 0), num_candidates)
    num_perturb = num_candidates - num_uniform
    perturb_key, uniform_key = jax.random.split(key)
    perturb = (
        actor_action[:, None, :]
        + config.cost_rank_candidate_perturb_std
        * jax.random.normal(
            perturb_key,
            (actor_action.shape[0], num_perturb, action_dim),
            dtype=jnp.float32,
        )
    )
    uniform = jax.random.uniform(
        uniform_key,
        (actor_action.shape[0], num_uniform, action_dim),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    actions = jnp.concatenate((perturb, uniform), axis=1)
    return jnp.clip(actions, -1.0, 1.0)


def _collect_rank_labels(
    train_state: TrainState,
    objects: TrainingObjects,
    config: TrainConfig,
    env_state: Any,
    state_mask: Array,
    state_pool_size: int,
    key: Array,
) -> tuple[RankBuffer, Mapping[str, Array]]:
    env_adapter = objects.env_adapter
    if env_adapter is None:
        raise ValueError("rank-label collection requires objects.env_adapter")

    select_key, action_key = jax.random.split(key)
    valid_state_count = jnp.sum(jnp.asarray(state_mask, dtype=bool))
    state_indices = jax.random.randint(
        select_key,
        (config.cost_rank_states_per_epoch,),
        0,
        state_pool_size,
    )
    selected_state_valid = jnp.take(
        jnp.asarray(state_mask, dtype=bool).reshape(-1),
        state_indices,
        mode="clip",
    )
    selected_env_state = _take_env_state(env_state, state_indices, state_pool_size)
    selected_batch_shape = (config.cost_rank_states_per_epoch,)
    state_info_steps = _env_state_info_value(
        selected_env_state, "steps", selected_batch_shape
    )
    state_done = _env_state_done(selected_env_state, selected_batch_shape)
    unroll_index_at_collection = state_indices // config.num_envs
    selected_obs = _real_state_observation(env_adapter, selected_env_state)
    selected_model_obs = _mask_goal_in_state(selected_obs, config)
    selected_goal = _real_rollout_goal(env_adapter, selected_env_state, config)
    actor_action = _deterministic_actor_action(
        objects.actor,
        train_state.actor_params,
        selected_model_obs,
        selected_goal,
    )
    candidate_actions = _rank_candidate_actions(
        actor_action, action_key, config, objects.action_dim
    )
    num_states = config.cost_rank_states_per_epoch
    num_candidates = config.cost_rank_num_candidates
    flat_actions = candidate_actions.reshape(
        num_states * num_candidates, objects.action_dim
    )
    flat_env_state = _repeat_env_state(selected_env_state, num_candidates, num_states)

    def rank_step(
        carry: tuple[Any, Array, Array, Array], step_idx: Array
    ) -> tuple[tuple[Any, Array, Array, Array], tuple[Array, Array, Array, Array]]:
        step_env_state, dense_return, sparse_return, alive = carry
        alive_before = alive
        obs = _real_state_observation(env_adapter, step_env_state)
        goal = _real_rollout_goal(env_adapter, step_env_state, config)
        follow_action = _deterministic_actor_action(
            objects.actor,
            train_state.actor_params,
            _mask_goal_in_state(obs, config),
            goal,
        )
        action = jax.lax.cond(
            step_idx == 0,
            lambda _: flat_actions,
            lambda _: follow_action,
            operand=None,
        )
        next_env_state, transition = env_adapter.step(step_env_state, action)
        dense_cost = _transition_dense_cost(transition.extras).reshape(
            num_states, num_candidates
        )
        sparse_cost = _transition_sparse_cost(transition.extras).reshape(
            num_states, num_candidates
        )
        gamma_t = jnp.power(jnp.asarray(config.gamma_c, dtype=jnp.float32), step_idx)
        dense_return = dense_return + alive * gamma_t * dense_cost
        sparse_return = sparse_return + alive * gamma_t * sparse_cost
        done = (1.0 - jnp.asarray(transition.discount, dtype=jnp.float32)).reshape(
            num_states, num_candidates
        )
        alive = alive * (1.0 - done)
        return (
            (next_env_state, dense_return, sparse_return, alive),
            (dense_cost, sparse_cost, done, alive_before),
        )

    zeros = jnp.zeros((num_states, num_candidates), dtype=jnp.float32)
    alive = jnp.ones((num_states, num_candidates), dtype=jnp.float32)
    (_, dense_labels, sparse_labels, alive_final), rank_trace = jax.lax.scan(
        rank_step,
        (flat_env_state, zeros, zeros, alive),
        jnp.arange(config.cost_rank_horizon),
    )
    dense_cost_trace, sparse_cost_trace, done_trace, alive_trace = rank_trace
    terminal_free = jnp.all(alive_final > 0.5, axis=-1)
    label = dense_labels if config.cost_rank_label_kind == "dense" else sparse_labels
    label_spread = jnp.max(label, axis=-1) - jnp.min(label, axis=-1)
    example_valid = label_spread > jnp.asarray(
        config.cost_rank_min_label_spread, dtype=jnp.float32
    )
    # Guard against stale or near-terminal source states.  The rank rollout uses
    # done-extend labels, so terminal states would otherwise enter as nearly
    # flat examples and silently drain the effective RankBuffer size.
    safe_step_limit = jnp.maximum(
        jnp.asarray(
            config.env_episode_length - config.cost_rank_horizon - 5,
            dtype=jnp.float32,
        ),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    step_window_valid = state_info_steps < safe_step_limit
    example_valid = example_valid & step_window_valid & (state_done < 0.5)
    rank_buffer = insert_rank_examples(
        train_state.rank_buffer,
        states=selected_model_obs,
        candidate_actions=candidate_actions,
        goals=selected_goal,
        dense_labels=dense_labels,
        sparse_labels=sparse_labels,
        state_info_steps=state_info_steps,
        state_done=state_done,
        unroll_index_at_collection=unroll_index_at_collection,
        valid=example_valid,
    )
    centered = label - jnp.mean(label, axis=-1, keepdims=True)
    within_var = jnp.mean(jnp.var(label, axis=-1))
    between_var = jnp.var(jnp.mean(label, axis=-1))
    upper_pairs = jnp.triu(
        jnp.ones((config.cost_rank_num_candidates, config.cost_rank_num_candidates)),
        k=1,
    )
    label_pair_mask = (
        jnp.abs(centered[:, :, None] - centered[:, None, :])
        > config.cost_rank_label_epsilon
    ).astype(jnp.float32) * upper_pairs[None, :, :]
    aux = {
        "rank_label_within_between": within_var / jnp.maximum(between_var, 1e-8),
        "rank_label_mean_spread": jnp.mean(
            jnp.max(label, axis=-1) - jnp.min(label, axis=-1)
        ),
        "rank_label_pair_frac_epoch": jnp.sum(label_pair_mask) / jnp.maximum(
            config.cost_rank_states_per_epoch
            * config.cost_rank_num_candidates
            * (config.cost_rank_num_candidates - 1)
            / 2.0,
            1.0,
        ),
        "rank_rollout_alive_frac": jnp.mean(alive_final),
        "rank_example_valid_frac": jnp.mean(example_valid.astype(jnp.float32)),
        "rank_terminal_free_frac": jnp.mean(terminal_free.astype(jnp.float32)),
        "rank_state_pool_valid_frac": valid_state_count.astype(jnp.float32)
        / jnp.maximum(jnp.asarray(state_pool_size, dtype=jnp.float32), 1.0),
        "rank_selected_state_valid_frac": jnp.mean(
            selected_state_valid.astype(jnp.float32)
        ),
    }
    if config.cost_rank_debug_dump:
        first_done = jnp.argmax(done_trace[:, 0, :] > 0.5, axis=0)
        any_done = jnp.any(done_trace[:, 0, :] > 0.5, axis=0)
        first_done = jnp.where(
            any_done,
            first_done,
            jnp.asarray(config.cost_rank_horizon, dtype=first_done.dtype),
        )

        def print_dump(_: None) -> Array:
            jax.debug.print(
                (
                    "RANK_DEBUG step={step} seed={seed} state_index={state_index} "
                    "horizon={horizon} gamma={gamma} tau={tau}\n"
                    "RANK_DEBUG obs_head={obs_head}\n"
                    "RANK_DEBUG goal={goal}\n"
                    "RANK_DEBUG actor_action={actor_action}\n"
                    "RANK_DEBUG candidate_actions={candidate_actions}\n"
                    "RANK_DEBUG dense_labels={dense_labels}\n"
                    "RANK_DEBUG sparse_labels={sparse_labels}\n"
                    "RANK_DEBUG dense_spread={dense_spread} "
                    "sparse_spread={sparse_spread} within_var={within_var} "
                    "between_var={between_var} wb={wb}\n"
                    "RANK_DEBUG first_done_step={first_done_step}\n"
                    "RANK_DEBUG alive_final={alive_final}\n"
                    "RANK_DEBUG alive_trace={alive_trace}\n"
                    "RANK_DEBUG done_trace={done_trace}\n"
                    "RANK_DEBUG dense_cost_trace={dense_cost_trace}\n"
                    "RANK_DEBUG sparse_cost_trace={sparse_cost_trace}"
                ),
                step=train_state.step,
                seed=jnp.asarray(config.seed, dtype=jnp.int32),
                state_index=state_indices[0],
                horizon=jnp.asarray(config.cost_rank_horizon, dtype=jnp.int32),
                gamma=jnp.asarray(config.gamma_c, dtype=jnp.float32),
                tau=jnp.asarray(config.cost_dense_prox_tau, dtype=jnp.float32),
                obs_head=selected_model_obs[0, :10],
                goal=selected_goal[0],
                actor_action=actor_action[0],
                candidate_actions=candidate_actions[0],
                dense_labels=dense_labels[0],
                sparse_labels=sparse_labels[0],
                dense_spread=jnp.max(dense_labels[0]) - jnp.min(dense_labels[0]),
                sparse_spread=jnp.max(sparse_labels[0]) - jnp.min(sparse_labels[0]),
                within_var=within_var,
                between_var=between_var,
                wb=aux["rank_label_within_between"],
                first_done_step=first_done,
                alive_final=alive_final[0],
                alive_trace=alive_trace[:, 0, :],
                done_trace=done_trace[:, 0, :],
                dense_cost_trace=dense_cost_trace[:, 0, :],
                sparse_cost_trace=sparse_cost_trace[:, 0, :],
            )
            return jnp.asarray(0, dtype=jnp.int32)

        _ = jax.lax.cond(
            train_state.step == 0,
            print_dump,
            lambda _: jnp.asarray(0, dtype=jnp.int32),
            operand=None,
        )
    return rank_buffer, aux


def _mean_transition_extra(
    extras: Mapping[str, Array], key: str, reference: Array
) -> Array:
    return jnp.mean(jnp.asarray(extras.get(key, jnp.zeros_like(reference))))


def _transition_sparse_cost(extras: Mapping[str, Array]) -> Array:
    cost = jnp.asarray(extras["cost"], dtype=jnp.float32)
    return jnp.asarray(extras.get("sparse_cost", cost), dtype=jnp.float32)


def _transition_dense_cost(extras: Mapping[str, Array]) -> Array:
    cost = jnp.asarray(extras["cost"], dtype=jnp.float32)
    return jnp.asarray(extras.get("dense_cost", cost), dtype=jnp.float32)


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


def _split_eval_params(params: Any) -> tuple[Any, Any | None]:
    if isinstance(params, tuple) and len(params) == 2:
        return params
    return params, None


def _safe_mean_masked(values: Array, mask: Array) -> Array:
    values = jnp.asarray(values, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    denom = jnp.sum(mask)
    return jnp.where(denom > 0.0, jnp.sum(values * mask) / denom, 0.0)


def _candidate_corr(x: Array, y: Array) -> Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    denom = jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y))
    return jnp.where(denom > 1e-8, jnp.sum(x * y) / denom, 0.0)


def _actor_percentile(values: Array) -> Array:
    values = jnp.asarray(values, dtype=jnp.float32)
    actor_value = values[0]
    less = jnp.sum((values < actor_value).astype(jnp.float32))
    equal = jnp.sum((values == actor_value).astype(jnp.float32))
    return (less + 0.5 * equal) / values.shape[0]


def _parse_counterfactual_horizons(value: str) -> tuple[int, ...]:
    horizons: list[int] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        horizon = int(piece)
        if horizon <= 0:
            raise ValueError("counterfactual probe horizons must be positive")
        horizons.append(horizon)
    return tuple(horizons)


def _counterfactual_candidate_summary(
    *,
    prefix: str,
    qcs: Array,
    true_cost: Array,
    true_sparse_cost: Array | None = None,
    true_dense_cost: Array | None = None,
    hard_violation: Array,
    hazard_violation: Array,
    current_cost: Array,
    current_sparse_cost: Array | None = None,
    current_dense_cost: Array | None = None,
    current_hard_violation: Array,
    current_min_hazard_dist: Array,
) -> dict[str, Array]:
    """Summarizes same-state candidate-action safety probes."""

    qcs = jnp.asarray(qcs, dtype=jnp.float32)
    true_cost = jnp.asarray(true_cost, dtype=jnp.float32)
    true_sparse_cost = true_cost if true_sparse_cost is None else true_sparse_cost
    true_dense_cost = true_cost if true_dense_cost is None else true_dense_cost
    true_sparse_cost = jnp.asarray(true_sparse_cost, dtype=jnp.float32)
    true_dense_cost = jnp.asarray(true_dense_cost, dtype=jnp.float32)
    hard_violation = jnp.asarray(hard_violation, dtype=jnp.float32)
    hazard_violation = jnp.asarray(hazard_violation, dtype=jnp.float32)
    current_cost = jnp.asarray(current_cost, dtype=jnp.float32)
    current_sparse_cost = current_cost if current_sparse_cost is None else current_sparse_cost
    current_dense_cost = current_cost if current_dense_cost is None else current_dense_cost
    current_sparse_cost = jnp.asarray(current_sparse_cost, dtype=jnp.float32)
    current_dense_cost = jnp.asarray(current_dense_cost, dtype=jnp.float32)
    current_hard_violation = jnp.asarray(current_hard_violation, dtype=jnp.float32)
    current_min_hazard_dist = jnp.asarray(current_min_hazard_dist, dtype=jnp.float32)

    true_cost_spread = jnp.max(true_cost, axis=0) - jnp.min(true_cost, axis=0)
    true_sparse_cost_spread = (
        jnp.max(true_sparse_cost, axis=0) - jnp.min(true_sparse_cost, axis=0)
    )
    true_dense_cost_spread = (
        jnp.max(true_dense_cost, axis=0) - jnp.min(true_dense_cost, axis=0)
    )
    hard_spread = jnp.max(hard_violation, axis=0) - jnp.min(hard_violation, axis=0)
    hazard_spread = (
        jnp.max(hazard_violation, axis=0) - jnp.min(hazard_violation, axis=0)
    )
    qc_spread = jnp.max(qcs, axis=0) - jnp.min(qcs, axis=0)
    corr_cost = jax.vmap(_candidate_corr, in_axes=(1, 1))(qcs, true_cost)
    corr_hazard = jax.vmap(_candidate_corr, in_axes=(1, 1))(qcs, hazard_violation)
    actor_true_cost_percentile = jax.vmap(_actor_percentile, in_axes=1)(true_cost)
    actor_qc_percentile = jax.vmap(_actor_percentile, in_axes=1)(qcs)
    actor_true_hazard_percentile = jax.vmap(_actor_percentile, in_axes=1)(
        hazard_violation
    )
    best_qc = jnp.argmin(qcs, axis=0)
    best_true_cost = jnp.argmin(true_cost, axis=0)
    best_true_hazard = jnp.argmin(hazard_violation, axis=0)
    match_cost = (best_qc == best_true_cost).astype(jnp.float32)
    match_hazard = (best_qc == best_true_hazard).astype(jnp.float32)
    nonzero_cost_spread = (true_cost_spread > 1e-6).astype(jnp.float32)
    nonzero_hard_spread = (hard_spread > 1e-6).astype(jnp.float32)
    nonzero_hazard_spread = (hazard_spread > 1e-6).astype(jnp.float32)

    per_state = {
        f"{prefix}true_cost_spread": true_cost_spread,
        f"{prefix}true_active_cost_spread": true_cost_spread,
        f"{prefix}true_sparse_cost_spread": true_sparse_cost_spread,
        f"{prefix}true_dense_cost_spread": true_dense_cost_spread,
        f"{prefix}true_hard_viol_spread": hard_spread,
        f"{prefix}true_hazard_spread": hazard_spread,
        f"{prefix}qc_action_spread": qc_spread,
        f"{prefix}corr_qc_true_cost": corr_cost,
        f"{prefix}corr_qc_true_hazard": corr_hazard,
        f"{prefix}actor_true_cost_percentile": actor_true_cost_percentile,
        f"{prefix}actor_qc_percentile": actor_qc_percentile,
        f"{prefix}actor_true_hazard_percentile": actor_true_hazard_percentile,
        f"{prefix}best_qc_matches_best_true_cost_frac": match_cost,
        f"{prefix}best_qc_matches_best_true_hazard_frac": match_hazard,
        f"{prefix}frac_nonzero_cost_spread": nonzero_cost_spread,
        f"{prefix}frac_nonzero_hard_viol_spread": nonzero_hard_spread,
        f"{prefix}frac_nonzero_hazard_spread": nonzero_hazard_spread,
    }
    metrics = {key: jnp.mean(value) for key, value in per_state.items()}

    hazard_dist_available = current_min_hazard_dist >= 0.0
    masks = {
        "costpos": current_cost > 0.0,
        "hardpos": current_hard_violation > 0.0,
        "haz1": hazard_dist_available & (current_min_hazard_dist < 1.0),
        "haz05": hazard_dist_available & (current_min_hazard_dist < 0.5),
        "haz025": hazard_dist_available & (current_min_hazard_dist < 0.25),
    }
    metrics[f"{prefix}hazard_dist_available_frac"] = jnp.mean(
        hazard_dist_available.astype(jnp.float32)
    )
    for suffix, mask in masks.items():
        mask_f = mask.astype(jnp.float32)
        metrics[f"{prefix}{suffix}_frac"] = jnp.mean(mask_f)
        for key, value in per_state.items():
            metrics[f"{key}_{suffix}"] = _safe_mean_masked(value, mask_f)
    return metrics


def _counterfactual_summary(
    *,
    qcs: Array,
    true_cost: Array,
    true_sparse_cost: Array | None = None,
    true_dense_cost: Array | None = None,
    hard_violation: Array,
    hazard_violation: Array,
    current_cost: Array,
    current_sparse_cost: Array | None = None,
    current_dense_cost: Array | None = None,
    current_hard_violation: Array,
    current_min_hazard_dist: Array,
) -> dict[str, Array]:
    """Summarizes same-state candidate-action safety probes."""
    metrics = _counterfactual_candidate_summary(
        prefix="one_step_",
        qcs=qcs,
        true_cost=true_cost,
        true_sparse_cost=true_sparse_cost,
        true_dense_cost=true_dense_cost,
        hard_violation=hard_violation,
        hazard_violation=hazard_violation,
        current_cost=current_cost,
        current_sparse_cost=current_sparse_cost,
        current_dense_cost=current_dense_cost,
        current_hard_violation=current_hard_violation,
        current_min_hazard_dist=current_min_hazard_dist,
    )
    legacy_key_map = {
        "true_action_cost_spread": "one_step_true_cost_spread",
        "true_action_active_cost_spread": "one_step_true_active_cost_spread",
        "true_action_sparse_cost_spread": "one_step_true_sparse_cost_spread",
        "true_action_dense_cost_spread": "one_step_true_dense_cost_spread",
        "true_action_hard_viol_spread": "one_step_true_hard_viol_spread",
        "true_action_hazard_spread": "one_step_true_hazard_spread",
        "qc_action_spread": "one_step_qc_action_spread",
        "corr_qc_true_cost": "one_step_corr_qc_true_cost",
        "corr_qc_true_hazard": "one_step_corr_qc_true_hazard",
        "actor_true_cost_percentile": "one_step_actor_true_cost_percentile",
        "actor_qc_percentile_cf": "one_step_actor_qc_percentile",
        "actor_true_hazard_percentile": "one_step_actor_true_hazard_percentile",
        "best_qc_matches_best_true_cost_frac": (
            "one_step_best_qc_matches_best_true_cost_frac"
        ),
        "best_qc_matches_best_true_hazard_frac": (
            "one_step_best_qc_matches_best_true_hazard_frac"
        ),
        "frac_states_with_nonzero_true_cost_spread": (
            "one_step_frac_nonzero_cost_spread"
        ),
        "frac_states_with_nonzero_hazard_spread": (
            "one_step_frac_nonzero_hazard_spread"
        ),
        "cf_hazard_dist_available_frac": "one_step_hazard_dist_available_frac",
        "cf_costpos_frac": "one_step_costpos_frac",
        "cf_hardpos_frac": "one_step_hardpos_frac",
        "cf_haz1_frac": "one_step_haz1_frac",
        "cf_haz05_frac": "one_step_haz05_frac",
        "cf_haz025_frac": "one_step_haz025_frac",
        "true_action_cost_spread_costpos": (
            "one_step_true_cost_spread_costpos"
        ),
        "corr_qc_true_cost_costpos": "one_step_corr_qc_true_cost_costpos",
        "true_action_cost_spread_hardpos": (
            "one_step_true_cost_spread_hardpos"
        ),
        "corr_qc_true_cost_hardpos": "one_step_corr_qc_true_cost_hardpos",
        "true_action_cost_spread_haz1": "one_step_true_cost_spread_haz1",
        "corr_qc_true_cost_haz1": "one_step_corr_qc_true_cost_haz1",
        "true_action_cost_spread_haz05": "one_step_true_cost_spread_haz05",
        "corr_qc_true_cost_haz05": "one_step_corr_qc_true_cost_haz05",
        "true_action_cost_spread_haz025": "one_step_true_cost_spread_haz025",
        "corr_qc_true_cost_haz025": "one_step_corr_qc_true_cost_haz025",
    }
    for legacy_key, new_key in legacy_key_map.items():
        metrics[legacy_key] = metrics[new_key]
    return metrics


def _multistep_counterfactual_summary(
    *,
    horizon: int,
    qcs: Array,
    cost_return: Array,
    sparse_cost_return: Array | None = None,
    dense_cost_return: Array | None = None,
    hard_return: Array,
    hazard_return: Array,
    current_cost: Array,
    current_sparse_cost: Array | None = None,
    current_dense_cost: Array | None = None,
    current_hard_violation: Array,
    current_min_hazard_dist: Array,
) -> dict[str, Array]:
    return _counterfactual_candidate_summary(
        prefix=f"cf_{horizon}_",
        qcs=qcs,
        true_cost=cost_return,
        true_sparse_cost=sparse_cost_return,
        true_dense_cost=dense_cost_return,
        hard_violation=hard_return,
        hazard_violation=hazard_return,
        current_cost=current_cost,
        current_sparse_cost=current_sparse_cost,
        current_dense_cost=current_dense_cost,
        current_hard_violation=current_hard_violation,
        current_min_hazard_dist=current_min_hazard_dist,
    )


def make_policy_evaluator(
    objects: TrainingObjects, config: TrainConfig, *, std_scale: float = 0.0
) -> Callable[[Any, Array], Mapping[str, Array]]:
    """Builds a full-episode evaluator for deterministic or scaled-noise actions."""

    env_adapter = objects.env_adapter
    multistep_horizons = _parse_counterfactual_horizons(
        config.counterfactual_probe_horizons
    )
    max_multistep_horizon = max(multistep_horizons, default=0)
    multistep_metric_prefixes = tuple(
        f"cf_{horizon}_" for horizon in multistep_horizons
    )

    def evaluate_real(params: Any, key: Array) -> Mapping[str, Array]:
        if env_adapter is None:
            raise ValueError("real-env evaluation requires objects.env_adapter")
        actor_params, cost_critic_params = _split_eval_params(params)
        key, reset_key = jax.random.split(key)
        eval_state, _ = env_adapter.reset(reset_key)
        initial_goal_xy = _real_goal_xy(env_adapter, eval_state)
        initial_frozen_goal_xy = initial_goal_xy
        initial_has_hit = jnp.zeros((config.num_envs,), dtype=bool)
        run_counterfactual_probe = (
            config.eval_counterfactual_action_probes
            and cost_critic_params is not None
        )
        run_multistep_counterfactual_probe = (
            config.enable_multistep_counterfactual_probes
            and run_counterfactual_probe
            and bool(multistep_horizons)
        )

        def eval_step(
            carry: tuple[Any, Array, Array, Array, Array], step_index: Array
        ) -> tuple[tuple[Any, Array, Array, Array, Array], Mapping[str, Array]]:
            (
                env_state,
                step_key,
                first_goal_xy,
                frozen_goal_xy,
                has_hit,
            ) = carry
            if run_counterfactual_probe:
                step_key, action_key, probe_key = jax.random.split(step_key, 3)
            else:
                step_key, action_key = jax.random.split(step_key)
                probe_key = action_key
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
            probe_metrics: dict[str, Array] = {}
            if run_counterfactual_probe:
                num_random = max(config.counterfactual_probe_random_actions, 0)
                num_perturb = max(config.counterfactual_probe_perturb_actions, 0)
                rand_key, perturb_key = jax.random.split(probe_key)
                random_actions = jax.random.uniform(
                    rand_key,
                    (num_random, config.num_envs, action.shape[-1]),
                    minval=-1.0,
                    maxval=1.0,
                    dtype=jnp.float32,
                )
                perturb_actions = jnp.clip(
                    action[None, ...]
                    + config.counterfactual_probe_perturb_std
                    * jax.random.normal(
                        perturb_key,
                        (num_perturb, config.num_envs, action.shape[-1]),
                        dtype=jnp.float32,
                    ),
                    -1.0,
                    1.0,
                )
                candidate_actions = jnp.concatenate(
                    [
                        action[None, ...],
                        jnp.zeros_like(action)[None, ...],
                        (-action)[None, ...],
                        random_actions,
                        perturb_actions,
                    ],
                    axis=0,
                )
                qcs = jax.vmap(
                    lambda candidate_action: objects.cost_critic.apply(
                        cost_critic_params, obs, candidate_action, goal
                    )
                )(candidate_actions)

                reference = jnp.zeros((config.num_envs,), dtype=jnp.float32)

                def probe_step(candidate_action: Array) -> tuple[Array, ...]:
                    _, candidate_transition = env_adapter.step(
                        env_state, candidate_action
                    )
                    extras = candidate_transition.extras
                    true_cost = jnp.asarray(
                        extras.get("cost", reference), dtype=jnp.float32
                    )
                    true_sparse_cost = jnp.asarray(
                        extras.get("sparse_cost", true_cost), dtype=jnp.float32
                    )
                    true_dense_cost = jnp.asarray(
                        extras.get("dense_cost", true_cost), dtype=jnp.float32
                    )
                    hard_violation = jnp.asarray(
                        extras.get("hard_violation", (true_cost > 0.0)),
                        dtype=jnp.float32,
                    )
                    hazard_violation = jnp.asarray(
                        extras.get("hazard_violation", hard_violation),
                        dtype=jnp.float32,
                    )
                    min_hazard_dist = jnp.asarray(
                        extras.get(
                            "min_hazard_dist",
                            jnp.full_like(true_cost, -1.0, dtype=jnp.float32),
                        ),
                        dtype=jnp.float32,
                    )
                    return (
                        true_cost,
                        true_sparse_cost,
                        true_dense_cost,
                        hard_violation,
                        hazard_violation,
                        min_hazard_dist,
                    )

                (
                    true_cost,
                    true_sparse_cost,
                    true_dense_cost,
                    hard_violation,
                    hazard_violation,
                    _candidate_min_hazard_dist,
                ) = jax.vmap(probe_step)(candidate_actions)
                current_cost = jnp.asarray(
                    transition.extras.get("cost", reference), dtype=jnp.float32
                )
                current_sparse_cost = jnp.asarray(
                    transition.extras.get("sparse_cost", current_cost),
                    dtype=jnp.float32,
                )
                current_dense_cost = jnp.asarray(
                    transition.extras.get("dense_cost", current_cost),
                    dtype=jnp.float32,
                )
                current_hard_violation = jnp.asarray(
                    transition.extras.get(
                        "hard_violation", (current_cost > 0.0).astype(jnp.float32)
                    ),
                    dtype=jnp.float32,
                )
                current_min_hazard_dist = jnp.asarray(
                    transition.extras.get(
                        "min_hazard_dist",
                        jnp.full_like(current_cost, -1.0, dtype=jnp.float32),
                    ),
                    dtype=jnp.float32,
                )
                probe_metrics = _counterfactual_summary(
                    qcs=qcs,
                    true_cost=true_cost,
                    true_sparse_cost=true_sparse_cost,
                    true_dense_cost=true_dense_cost,
                    hard_violation=hard_violation,
                    hazard_violation=hazard_violation,
                    current_cost=current_cost,
                    current_sparse_cost=current_sparse_cost,
                    current_dense_cost=current_dense_cost,
                    current_hard_violation=current_hard_violation,
                    current_min_hazard_dist=current_min_hazard_dist,
                )
                if run_multistep_counterfactual_probe:
                    max_states = config.counterfactual_probe_max_states
                    active = jnp.asarray(max_states <= 0) | (
                        step_index < max_states
                    )

                    def extract_step_safety(candidate_transition: Any) -> tuple[
                        Array, Array, Array, Array, Array, Array
                    ]:
                        extras = candidate_transition.extras
                        true_cost = jnp.asarray(
                            extras.get("cost", reference), dtype=jnp.float32
                        )
                        true_sparse_cost = jnp.asarray(
                            extras.get("sparse_cost", true_cost), dtype=jnp.float32
                        )
                        true_dense_cost = jnp.asarray(
                            extras.get("dense_cost", true_cost), dtype=jnp.float32
                        )
                        hard_violation = jnp.asarray(
                            extras.get("hard_violation", (true_cost > 0.0)),
                            dtype=jnp.float32,
                        )
                        hazard_violation = jnp.asarray(
                            extras.get("hazard_violation", hard_violation),
                            dtype=jnp.float32,
                        )
                        min_hazard_dist = jnp.asarray(
                            extras.get(
                                "min_hazard_dist",
                                jnp.full_like(
                                    true_cost, -1.0, dtype=jnp.float32
                                ),
                            ),
                            dtype=jnp.float32,
                        )
                        return (
                            true_cost,
                            true_sparse_cost,
                            true_dense_cost,
                            hard_violation,
                            hazard_violation,
                            min_hazard_dist,
                        )

                    def compute_multistep() -> dict[str, Array]:
                        def rollout_candidate(candidate_action: Array) -> tuple[
                            Array, Array, Array, Array, Array, Array
                        ]:
                            def rollout_step(
                                carry: tuple[
                                    Any, Array, Array, Array, Array, Array, Array
                                ],
                                horizon_index: Array,
                            ) -> tuple[
                                tuple[
                                    Any, Array, Array, Array, Array, Array, Array
                                ],
                                tuple[Array, Array, Array, Array, Array, Array],
                            ]:
                                (
                                    rollout_state,
                                    cost_acc,
                                    sparse_cost_acc,
                                    dense_cost_acc,
                                    hard_acc,
                                    hazard_acc,
                                    discount,
                                ) = carry
                                rollout_obs = _real_state_observation(
                                    env_adapter, rollout_state
                                )
                                rollout_goal = _real_rollout_goal(
                                    env_adapter, rollout_state, config
                                )
                                actor_follow_action = _deterministic_action(
                                    objects.actor,
                                    actor_params,
                                    rollout_obs,
                                    rollout_goal,
                                    config,
                                )
                                rollout_action = jnp.where(
                                    horizon_index == 0,
                                    candidate_action,
                                    actor_follow_action,
                                )
                                next_rollout_state, rollout_transition = (
                                    env_adapter.step(rollout_state, rollout_action)
                                )
                                (
                                    step_cost,
                                    step_sparse_cost,
                                    step_dense_cost,
                                    step_hard,
                                    step_hazard,
                                    step_min_hazard_dist,
                                ) = extract_step_safety(rollout_transition)
                                return (
                                    next_rollout_state,
                                    cost_acc + discount * step_cost,
                                    sparse_cost_acc + discount * step_sparse_cost,
                                    dense_cost_acc + discount * step_dense_cost,
                                    hard_acc + discount * step_hard,
                                    hazard_acc + discount * step_hazard,
                                    discount * config.gamma_c,
                                ), (
                                    cost_acc + discount * step_cost,
                                    sparse_cost_acc + discount * step_sparse_cost,
                                    dense_cost_acc + discount * step_dense_cost,
                                    hard_acc + discount * step_hard,
                                    hazard_acc + discount * step_hazard,
                                    step_min_hazard_dist,
                                )

                            init = (
                                env_state,
                                jnp.zeros((config.num_envs,), dtype=jnp.float32),
                                jnp.zeros((config.num_envs,), dtype=jnp.float32),
                                jnp.zeros((config.num_envs,), dtype=jnp.float32),
                                jnp.zeros((config.num_envs,), dtype=jnp.float32),
                                jnp.zeros((config.num_envs,), dtype=jnp.float32),
                                jnp.ones((config.num_envs,), dtype=jnp.float32),
                            )
                            _, history = jax.lax.scan(
                                rollout_step,
                                init,
                                jnp.arange(max_multistep_horizon),
                            )
                            (
                                cost_history,
                                sparse_cost_history,
                                dense_cost_history,
                                hard_history,
                                hazard_history,
                                min_hazard_history,
                            ) = history
                            return (
                                cost_history,
                                sparse_cost_history,
                                dense_cost_history,
                                hard_history,
                                hazard_history,
                                min_hazard_history,
                            )

                        (
                            cost_histories,
                            sparse_cost_histories,
                            dense_cost_histories,
                            hard_histories,
                            hazard_histories,
                            min_hazard_histories,
                        ) = jax.vmap(rollout_candidate)(candidate_actions)
                        metrics: dict[str, Array] = {}
                        for horizon in multistep_horizons:
                            horizon_index = horizon - 1
                            prefix = f"cf_{horizon}_"
                            min_hazard_over_h = jnp.min(
                                min_hazard_histories[:, :horizon], axis=1
                            )
                            final_hazard_dist = min_hazard_histories[
                                :, horizon_index
                            ]
                            metrics.update(
                                _multistep_counterfactual_summary(
                                    horizon=horizon,
                                    qcs=qcs,
                                    cost_return=cost_histories[:, horizon_index],
                                    sparse_cost_return=sparse_cost_histories[
                                        :, horizon_index
                                    ],
                                    dense_cost_return=dense_cost_histories[
                                        :, horizon_index
                                    ],
                                    hard_return=hard_histories[:, horizon_index],
                                    hazard_return=hazard_histories[:, horizon_index],
                                    current_cost=current_cost,
                                    current_sparse_cost=current_sparse_cost,
                                    current_dense_cost=current_dense_cost,
                                    current_hard_violation=current_hard_violation,
                                    current_min_hazard_dist=current_min_hazard_dist,
                                )
                            )
                            metrics[f"{prefix}min_hazard_dist_over_h_spread"] = (
                                jnp.mean(
                                    jnp.max(min_hazard_over_h, axis=0)
                                    - jnp.min(min_hazard_over_h, axis=0)
                                )
                            )
                            metrics[f"{prefix}final_hazard_dist_spread"] = jnp.mean(
                                jnp.max(final_hazard_dist, axis=0)
                                - jnp.min(final_hazard_dist, axis=0)
                            )
                        metrics["multistep_cf_active"] = jnp.asarray(
                            1.0, dtype=jnp.float32
                        )
                        return metrics

                    def zero_multistep() -> dict[str, Array]:
                        metrics: dict[str, Array] = {}
                        for horizon in multistep_horizons:
                            template = _multistep_counterfactual_summary(
                                horizon=horizon,
                                qcs=jnp.zeros_like(qcs),
                                cost_return=jnp.zeros_like(qcs),
                                sparse_cost_return=jnp.zeros_like(qcs),
                                dense_cost_return=jnp.zeros_like(qcs),
                                hard_return=jnp.zeros_like(qcs),
                                hazard_return=jnp.zeros_like(qcs),
                                current_cost=current_cost,
                                current_sparse_cost=current_sparse_cost,
                                current_dense_cost=current_dense_cost,
                                current_hard_violation=current_hard_violation,
                                current_min_hazard_dist=current_min_hazard_dist,
                            )
                            zero_metrics = {
                                name: jnp.zeros_like(value)
                                for name, value in template.items()
                            }
                            metrics.update(zero_metrics)
                            prefix = f"cf_{horizon}_"
                            metrics[f"{prefix}min_hazard_dist_over_h_spread"] = (
                                jnp.asarray(0.0, dtype=jnp.float32)
                            )
                            metrics[f"{prefix}final_hazard_dist_spread"] = (
                                jnp.asarray(0.0, dtype=jnp.float32)
                            )
                        metrics["multistep_cf_active"] = jnp.asarray(
                            0.0, dtype=jnp.float32
                        )
                        return metrics

                    probe_metrics = {
                        **probe_metrics,
                        **jax.lax.cond(active, compute_multistep, zero_multistep),
                    }
            robot_xy_after = _real_robot_xy(env_adapter, next_env_state)
            reference = jnp.zeros((config.num_envs,), dtype=jnp.float32)
            cost = jnp.asarray(
                transition.extras.get(
                    "sparse_cost", transition.extras.get("cost", reference)
                ),
                dtype=jnp.float32,
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
                **probe_metrics,
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
        metrics = _eval_metrics(
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
        if run_counterfactual_probe:
            for key, value in trajectory.items():
                if key in {
                    "commanded_goal_dist",
                    "initial_goal_dist",
                    "resampled_goal_dist",
                    "goal_reached",
                    "cost",
                    "frozen_goal_dist",
                }:
                    continue
                value = jnp.asarray(value, dtype=jnp.float32)
                if (
                    run_multistep_counterfactual_probe
                    and key.startswith(multistep_metric_prefixes)
                    and "multistep_cf_active" in trajectory
                ):
                    active = jnp.asarray(
                        trajectory["multistep_cf_active"], dtype=jnp.float32
                    )
                    denom = jnp.maximum(jnp.sum(active), 1.0)
                    metrics[key] = jnp.sum(value * active) / denom
                else:
                    metrics[key] = jnp.mean(value)
        return metrics

    def evaluate_toy(params: Any, key: Array) -> Mapping[str, Array]:
        actor_params, _ = _split_eval_params(params)
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
    sparse_cost = _transition_sparse_cost(transitions.extras)
    dense_cost = _transition_dense_cost(transitions.extras)
    goal_metrics = _goal_distance_metrics(transitions.extras["goal_dist"])
    metrics = {
        "reward": jnp.mean(transitions.reward),
        "cost": jnp.mean(sparse_cost),
        "dense_cost_mean": jnp.mean(dense_cost),
        "dense_cost_std": jnp.std(dense_cost),
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
        "rank_label_within_between": jnp.asarray(0.0, dtype=jnp.float32),
        "rank_label_mean_spread": jnp.asarray(0.0, dtype=jnp.float32),
        "rank_label_pair_frac_epoch": jnp.asarray(0.0, dtype=jnp.float32),
        "rank_rollout_alive_frac": jnp.asarray(0.0, dtype=jnp.float32),
        "rank_example_valid_frac": jnp.asarray(0.0, dtype=jnp.float32),
        "rank_terminal_free_frac": jnp.asarray(0.0, dtype=jnp.float32),
        "rank_state_pool_valid_frac": jnp.asarray(0.0, dtype=jnp.float32),
        "rank_selected_state_valid_frac": jnp.asarray(0.0, dtype=jnp.float32),
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
    key, scan_key, rank_key = jax.random.split(train_state.key, 3)
    env_adapter = objects.env_adapter
    if env_adapter is None:
        raise ValueError("real-env collection requires objects.env_adapter")

    def collect_step(
        carry: tuple[Array, Any], _: Array
    ) -> tuple[tuple[Array, Any], tuple[Any, Transition]]:
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
        return (step_key, next_env_state), (env_state, transition)

    (next_key, next_env_state), (rollout_env_states, transitions) = jax.lax.scan(
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
    sparse_cost = _transition_sparse_cost(transitions.extras)
    dense_cost = _transition_dense_cost(transitions.extras)
    if config.probe_counterfactual_costs:
        cost_zero_action = transitions.extras["cost_zero_action"]
        cost_neg_action = transitions.extras["cost_neg_action"]
        sparse_cost_zero_action = transitions.extras["sparse_cost_zero_action"]
        sparse_cost_neg_action = transitions.extras["sparse_cost_neg_action"]
        dense_cost_zero_action = transitions.extras["dense_cost_zero_action"]
        dense_cost_neg_action = transitions.extras["dense_cost_neg_action"]
    else:
        cost_zero_action = jnp.zeros_like(transitions.extras["cost"])
        cost_neg_action = jnp.zeros_like(transitions.extras["cost"])
        sparse_cost_zero_action = jnp.zeros_like(sparse_cost)
        sparse_cost_neg_action = jnp.zeros_like(sparse_cost)
        dense_cost_zero_action = jnp.zeros_like(dense_cost)
        dense_cost_neg_action = jnp.zeros_like(dense_cost)
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
        "cost": jnp.mean(sparse_cost),
        "dense_cost_mean": jnp.mean(dense_cost),
        "dense_cost_std": jnp.std(dense_cost),
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
        "sparse_cost_zero_action": jnp.mean(sparse_cost_zero_action),
        "sparse_cost_neg_action": jnp.mean(sparse_cost_neg_action),
        "sparse_cost_action_minus_zero": jnp.mean(sparse_cost - sparse_cost_zero_action),
        "dense_cost_zero_action": jnp.mean(dense_cost_zero_action),
        "dense_cost_neg_action": jnp.mean(dense_cost_neg_action),
        "dense_cost_action_minus_zero": jnp.mean(dense_cost - dense_cost_zero_action),
    }
    rank_buffer = train_state.rank_buffer
    if config.cost_rank_loss_weight > 0.0:
        rank_state_pool = _flatten_env_state_history(
            rollout_env_states,
            config.unroll_length,
            config.num_envs,
        )
        rank_state_mask = _rank_state_history_mask(
            transitions.discount,
            config.cost_rank_horizon,
        ).reshape((config.unroll_length * config.num_envs,))
        rank_buffer, rank_aux = _collect_rank_labels(
            train_state,
            objects,
            config,
            rank_state_pool,
            rank_state_mask,
            config.unroll_length * config.num_envs,
            rank_key,
        )
    else:
        rank_aux = {
            "rank_label_within_between": jnp.asarray(0.0, dtype=jnp.float32),
            "rank_label_mean_spread": jnp.asarray(0.0, dtype=jnp.float32),
            "rank_label_pair_frac_epoch": jnp.asarray(0.0, dtype=jnp.float32),
            "rank_rollout_alive_frac": jnp.asarray(0.0, dtype=jnp.float32),
            "rank_example_valid_frac": jnp.asarray(0.0, dtype=jnp.float32),
            "rank_terminal_free_frac": jnp.asarray(0.0, dtype=jnp.float32),
            "rank_state_pool_valid_frac": jnp.asarray(0.0, dtype=jnp.float32),
            "rank_selected_state_valid_frac": jnp.asarray(0.0, dtype=jnp.float32),
        }
    metrics.update(rank_aux)
    next_state = train_state.replace(
        key=key,
        env_state=next_env_state,
        replay=replay,
        rank_buffer=rank_buffer,
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
        rank_sample_key,
        actor_key,
        cost_key,
        alpha_key,
        dual_key,
    ) = jax.random.split(train_state.key, 8)
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
    rank_batch = (
        sample_rank_batch(
            train_state.rank_buffer,
            rank_sample_key,
            batch_size=config.cost_rank_batch_size,
        )
        if config.cost_rank_loss_weight > 0.0
        else None
    )

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
            rank_batch=rank_batch,
            cost_rank_loss_weight=config.cost_rank_loss_weight,
            cost_rank_label_kind=config.cost_rank_label_kind,
            cost_rank_label_epsilon=config.cost_rank_label_epsilon,
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
        "cost_rank_loss": cc_aux["cost_rank_loss"],
        "cost_rank_pair_frac": cc_aux["cost_rank_pair_frac"],
        "cost_rank_batch_frac": cc_aux["cost_rank_batch_frac"],
        "cost_rank_spearman": cc_aux["cost_rank_spearman"],
        "cost_rank_top1_match": cc_aux["cost_rank_top1_match"],
        "cost_rank_loss_weight": jnp.asarray(
            config.cost_rank_loss_weight, dtype=jnp.float32
        ),
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
        "cost_target": cc_aux["mean_cost"],
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
        metrics["cost"] = collect_metrics["cost"]
        metrics["dense_cost_mean"] = collect_metrics["dense_cost_mean"]
        metrics["dense_cost_std"] = collect_metrics["dense_cost_std"]
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
        metrics["rank_label_within_between"] = collect_metrics[
            "rank_label_within_between"
        ]
        metrics["rank_label_mean_spread"] = collect_metrics["rank_label_mean_spread"]
        metrics["rank_label_pair_frac_epoch"] = collect_metrics[
            "rank_label_pair_frac_epoch"
        ]
        metrics["rank_rollout_alive_frac"] = collect_metrics["rank_rollout_alive_frac"]
        metrics["rank_example_valid_frac"] = collect_metrics["rank_example_valid_frac"]
        metrics["rank_terminal_free_frac"] = collect_metrics["rank_terminal_free_frac"]
        metrics["rank_state_pool_valid_frac"] = collect_metrics[
            "rank_state_pool_valid_frac"
        ]
        metrics["rank_selected_state_valid_frac"] = collect_metrics[
            "rank_selected_state_valid_frac"
        ]
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
    if config.cost_mode not in {"sparse", "dense_proximity"}:
        raise ValueError("cost_mode must be 'sparse' or 'dense_proximity'")
    if config.cost_dense_prox_tau <= 0.0:
        raise ValueError("cost_dense_prox_tau must be positive")
    if config.cost_rank_loss_weight < 0.0:
        raise ValueError("cost_rank_loss_weight must be non-negative")
    if config.cost_rank_horizon <= 0:
        raise ValueError("cost_rank_horizon must be positive")
    if config.cost_rank_num_candidates < 2:
        raise ValueError("cost_rank_num_candidates must be at least 2")
    if config.cost_rank_states_per_epoch <= 0:
        raise ValueError("cost_rank_states_per_epoch must be positive")
    if config.cost_rank_buffer_capacity <= 0:
        raise ValueError("cost_rank_buffer_capacity must be positive")
    if config.cost_rank_batch_size <= 0:
        raise ValueError("cost_rank_batch_size must be positive")
    if config.cost_rank_uniform_anchor_count < 0:
        raise ValueError("cost_rank_uniform_anchor_count must be non-negative")
    if config.cost_rank_uniform_random_frac >= 0.0 and not (
        0.0 <= config.cost_rank_uniform_random_frac <= 1.0
    ):
        raise ValueError("cost_rank_uniform_random_frac must be -1 or in [0, 1]")
    if config.cost_rank_candidate_perturb_std < 0.0:
        raise ValueError("cost_rank_candidate_perturb_std must be non-negative")
    if config.cost_rank_label_epsilon < 0.0:
        raise ValueError("cost_rank_label_epsilon must be non-negative")
    if config.cost_rank_min_label_spread < 0.0:
        raise ValueError("cost_rank_min_label_spread must be non-negative")
    if config.cost_rank_label_kind not in {"dense", "sparse"}:
        raise ValueError("cost_rank_label_kind must be 'dense' or 'sparse'")
    if config.cost_rank_done_mode not in {"extend"}:
        raise ValueError("cost_rank_done_mode must be 'extend'")
    if not 0.0 <= config.cost_risk_replay_ratio <= 1.0:
        raise ValueError("cost_risk_replay_ratio must be in [0, 1]")
    if config.cost_risk_hazard_lidar_thresh <= 0.0:
        raise ValueError("cost_risk_hazard_lidar_thresh must be positive")
    if config.cost_risk_min_fraction_available < 0.0:
        raise ValueError("cost_risk_min_fraction_available must be non-negative")
    _parse_counterfactual_horizons(config.counterfactual_probe_horizons)
    if config.counterfactual_probe_max_states < 0:
        raise ValueError("counterfactual_probe_max_states must be non-negative")
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
            cost_mode=config.cost_mode,
            cost_dense_prox_tau=config.cost_dense_prox_tau,
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
    rank_buffer = make_rank_buffer(
        capacity=config.cost_rank_buffer_capacity,
        state_dim=runtime_observation_dim,
        action_dim=runtime_action_dim,
        goal_dim=config.goal_dim,
        num_candidates=config.cost_rank_num_candidates,
    )
    state = TrainState(
        key=key,
        step=jnp.asarray(0, dtype=jnp.int32),
        env_state=env_state,
        replay=replay,
        rank_buffer=rank_buffer,
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


def _format_counterfactual_probe_line(metrics: Mapping[str, Array]) -> str | None:
    if "true_action_cost_spread" not in metrics:
        return None
    return (
        "         "
        f"counterfactual[ true_action_active_cost_spread="
        f"{_mean_float(metrics, 'true_action_active_cost_spread'):.2e} "
        f"true_action_sparse_cost_spread="
        f"{_mean_float(metrics, 'true_action_sparse_cost_spread'):.2e} "
        f"true_action_dense_cost_spread="
        f"{_mean_float(metrics, 'true_action_dense_cost_spread'):.2e} "
        f"true_action_hard_viol_spread="
        f"{_mean_float(metrics, 'true_action_hard_viol_spread'):.2e} "
        f"true_action_hazard_spread="
        f"{_mean_float(metrics, 'true_action_hazard_spread'):.2e} "
        f"qc_action_spread={_mean_float(metrics, 'qc_action_spread'):.2e} "
        f"corr_qc_true_cost={_mean_float(metrics, 'corr_qc_true_cost'):.3f} "
        f"corr_qc_true_hazard={_mean_float(metrics, 'corr_qc_true_hazard'):.3f} "
        f"actor_true_cost_percentile="
        f"{_mean_float(metrics, 'actor_true_cost_percentile'):.3f} "
        f"actor_qc_percentile_cf="
        f"{_mean_float(metrics, 'actor_qc_percentile_cf'):.3f} "
        f"actor_true_hazard_percentile="
        f"{_mean_float(metrics, 'actor_true_hazard_percentile'):.3f} "
        f"best_qc_matches_best_true_cost_frac="
        f"{_mean_float(metrics, 'best_qc_matches_best_true_cost_frac'):.3f} "
        f"best_qc_matches_best_true_hazard_frac="
        f"{_mean_float(metrics, 'best_qc_matches_best_true_hazard_frac'):.3f} "
        f"frac_states_with_nonzero_true_cost_spread="
        f"{_mean_float(metrics, 'frac_states_with_nonzero_true_cost_spread'):.3f} "
        f"frac_states_with_nonzero_hazard_spread="
        f"{_mean_float(metrics, 'frac_states_with_nonzero_hazard_spread'):.3f} "
        f"cf_hazard_dist_available_frac="
        f"{_mean_float(metrics, 'cf_hazard_dist_available_frac'):.3f} "
        f"cf_costpos_frac={_mean_float(metrics, 'cf_costpos_frac'):.3f} "
        f"cf_hardpos_frac={_mean_float(metrics, 'cf_hardpos_frac'):.3f} "
        f"cf_haz1_frac={_mean_float(metrics, 'cf_haz1_frac'):.3f} "
        f"cf_haz05_frac={_mean_float(metrics, 'cf_haz05_frac'):.3f} "
        f"cf_haz025_frac={_mean_float(metrics, 'cf_haz025_frac'):.3f} "
        f"true_action_cost_spread_costpos="
        f"{_mean_float(metrics, 'true_action_cost_spread_costpos'):.2e} "
        f"corr_qc_true_cost_costpos="
        f"{_mean_float(metrics, 'corr_qc_true_cost_costpos'):.3f} "
        f"true_action_cost_spread_hardpos="
        f"{_mean_float(metrics, 'true_action_cost_spread_hardpos'):.2e} "
        f"corr_qc_true_cost_hardpos="
        f"{_mean_float(metrics, 'corr_qc_true_cost_hardpos'):.3f} "
        f"true_action_cost_spread_haz1="
        f"{_mean_float(metrics, 'true_action_cost_spread_haz1'):.2e} "
        f"corr_qc_true_cost_haz1="
        f"{_mean_float(metrics, 'corr_qc_true_cost_haz1'):.3f} "
        f"true_action_cost_spread_haz05="
        f"{_mean_float(metrics, 'true_action_cost_spread_haz05'):.2e} "
        f"corr_qc_true_cost_haz05="
        f"{_mean_float(metrics, 'corr_qc_true_cost_haz05'):.3f} "
        f"true_action_cost_spread_haz025="
        f"{_mean_float(metrics, 'true_action_cost_spread_haz025'):.2e} "
        f"corr_qc_true_cost_haz025="
        f"{_mean_float(metrics, 'corr_qc_true_cost_haz025'):.3f}]"
    )


def _format_multistep_counterfactual_probe_lines(
    metrics: Mapping[str, Array]
) -> list[str]:
    horizon_labels = sorted(
        {
            key.split("_", 2)[1]
            for key in metrics
            if key.startswith("cf_") and key.endswith("_true_cost_spread")
        },
        key=int,
    )
    lines: list[str] = []
    for horizon in horizon_labels:
        prefix = f"cf_{horizon}_"
        best_cost_frac = _mean_float(
            metrics, f"{prefix}best_qc_matches_best_true_cost_frac"
        )
        best_hazard_frac = _mean_float(
            metrics, f"{prefix}best_qc_matches_best_true_hazard_frac"
        )
        lines.append(
            "         "
            f"counterfactual_H{horizon}[ "
            f"{prefix}true_active_cost_spread="
            f"{_mean_float(metrics, f'{prefix}true_active_cost_spread'):.2e} "
            f"{prefix}true_sparse_cost_spread="
            f"{_mean_float(metrics, f'{prefix}true_sparse_cost_spread'):.2e} "
            f"{prefix}true_dense_cost_spread="
            f"{_mean_float(metrics, f'{prefix}true_dense_cost_spread'):.2e} "
            f"{prefix}true_hard_viol_spread="
            f"{_mean_float(metrics, f'{prefix}true_hard_viol_spread'):.2e} "
            f"{prefix}true_hazard_spread="
            f"{_mean_float(metrics, f'{prefix}true_hazard_spread'):.2e} "
            f"{prefix}frac_nonzero_cost_spread="
            f"{_mean_float(metrics, f'{prefix}frac_nonzero_cost_spread'):.3f} "
            f"{prefix}frac_nonzero_hard_viol_spread="
            f"{_mean_float(metrics, f'{prefix}frac_nonzero_hard_viol_spread'):.3f} "
            f"{prefix}frac_nonzero_hazard_spread="
            f"{_mean_float(metrics, f'{prefix}frac_nonzero_hazard_spread'):.3f} "
            f"{prefix}corr_qc_true_cost="
            f"{_mean_float(metrics, f'{prefix}corr_qc_true_cost'):.3f} "
            f"{prefix}corr_qc_true_hazard="
            f"{_mean_float(metrics, f'{prefix}corr_qc_true_hazard'):.3f} "
            f"{prefix}actor_true_cost_percentile="
            f"{_mean_float(metrics, f'{prefix}actor_true_cost_percentile'):.3f} "
            f"{prefix}actor_true_hazard_percentile="
            f"{_mean_float(metrics, f'{prefix}actor_true_hazard_percentile'):.3f} "
            f"{prefix}actor_qc_percentile="
            f"{_mean_float(metrics, f'{prefix}actor_qc_percentile'):.3f} "
            f"{prefix}best_qc_matches_best_true_cost_frac="
            f"{best_cost_frac:.3f} "
            f"{prefix}best_qc_matches_best_true_hazard_frac="
            f"{best_hazard_frac:.3f} "
            f"{prefix}qc_action_spread="
            f"{_mean_float(metrics, f'{prefix}qc_action_spread'):.2e} "
            f"{prefix}min_hazard_dist_over_h_spread="
            f"{_mean_float(metrics, f'{prefix}min_hazard_dist_over_h_spread'):.2e} "
            f"{prefix}final_hazard_dist_spread="
            f"{_mean_float(metrics, f'{prefix}final_hazard_dist_spread'):.2e} "
            f"{prefix}hazard_dist_available_frac="
            f"{_mean_float(metrics, f'{prefix}hazard_dist_available_frac'):.3f} "
            f"{prefix}costpos_frac={_mean_float(metrics, f'{prefix}costpos_frac'):.3f} "
            f"{prefix}hardpos_frac={_mean_float(metrics, f'{prefix}hardpos_frac'):.3f} "
            f"{prefix}haz1_frac={_mean_float(metrics, f'{prefix}haz1_frac'):.3f} "
            f"{prefix}haz05_frac={_mean_float(metrics, f'{prefix}haz05_frac'):.3f} "
            f"{prefix}haz025_frac={_mean_float(metrics, f'{prefix}haz025_frac'):.3f} "
            f"{prefix}true_cost_spread_costpos="
            f"{_mean_float(metrics, f'{prefix}true_cost_spread_costpos'):.2e} "
            f"{prefix}corr_qc_true_cost_costpos="
            f"{_mean_float(metrics, f'{prefix}corr_qc_true_cost_costpos'):.3f} "
            f"{prefix}true_cost_spread_hardpos="
            f"{_mean_float(metrics, f'{prefix}true_cost_spread_hardpos'):.2e} "
            f"{prefix}corr_qc_true_cost_hardpos="
            f"{_mean_float(metrics, f'{prefix}corr_qc_true_cost_hardpos'):.3f} "
            f"{prefix}true_cost_spread_haz1="
            f"{_mean_float(metrics, f'{prefix}true_cost_spread_haz1'):.2e} "
            f"{prefix}corr_qc_true_cost_haz1="
            f"{_mean_float(metrics, f'{prefix}corr_qc_true_cost_haz1'):.3f} "
            f"{prefix}true_cost_spread_haz05="
            f"{_mean_float(metrics, f'{prefix}true_cost_spread_haz05'):.2e} "
            f"{prefix}corr_qc_true_cost_haz05="
            f"{_mean_float(metrics, f'{prefix}corr_qc_true_cost_haz05'):.3f} "
            f"{prefix}true_cost_spread_haz025="
            f"{_mean_float(metrics, f'{prefix}true_cost_spread_haz025'):.2e} "
            f"{prefix}corr_qc_true_cost_haz025="
            f"{_mean_float(metrics, f'{prefix}corr_qc_true_cost_haz025'):.3f}]"
        )
    return lines


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
                f"dense_cost_mean={_mean_float(metrics, 'dense_cost_mean'):.4f} "
                f"dense_cost_std={_mean_float(metrics, 'dense_cost_std'):.4f} "
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
                f"c_target={_mean_float(metrics, 'cost_target'):.4f} "
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
                f"rank[ cost_rank_loss="
                f"{_mean_float(metrics, 'cost_rank_loss'):.4f} "
                f"cost_rank_loss_weight="
                f"{_mean_float(metrics, 'cost_rank_loss_weight'):.2e} "
                f"cost_rank_pair_frac="
                f"{_mean_float(metrics, 'cost_rank_pair_frac'):.3f} "
                f"cost_rank_batch_frac="
                f"{_mean_float(metrics, 'cost_rank_batch_frac'):.3f} "
                f"cost_rank_spearman="
                f"{_mean_float(metrics, 'cost_rank_spearman'):.3f} "
                f"cost_rank_top1_match="
                f"{_mean_float(metrics, 'cost_rank_top1_match'):.3f} "
                f"rank_label_within_between="
                f"{_mean_float(metrics, 'rank_label_within_between'):.2e} "
                f"rank_label_mean_spread="
                f"{_mean_float(metrics, 'rank_label_mean_spread'):.2e} "
                f"rank_label_pair_frac_epoch="
                f"{_mean_float(metrics, 'rank_label_pair_frac_epoch'):.3f} "
                f"rank_rollout_alive_frac="
                f"{_mean_float(metrics, 'rank_rollout_alive_frac'):.3f} "
                f"rank_example_valid_frac="
                f"{_mean_float(metrics, 'rank_example_valid_frac'):.3f} "
                f"rank_terminal_free_frac="
                f"{_mean_float(metrics, 'rank_terminal_free_frac'):.3f} "
                f"rank_state_pool_valid_frac="
                f"{_mean_float(metrics, 'rank_state_pool_valid_frac'):.3f} "
                f"rank_selected_state_valid_frac="
                f"{_mean_float(metrics, 'rank_selected_state_valid_frac'):.3f}]"
            ),
            *(
                [counterfactual_line]
                if (
                    counterfactual_line := _format_counterfactual_probe_line(metrics)
                )
                is not None
                else []
            ),
            *_format_multistep_counterfactual_probe_lines(metrics),
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


def _rank_buffer_dump_target(epoch: int) -> Path | None:
    raw_epoch = os.environ.get("SR_CPO_DUMP_RANK_BUFFER_AT_EPOCH", "")
    if not raw_epoch:
        return None
    try:
        dump_epoch = int(raw_epoch)
    except ValueError as exc:
        raise ValueError("SR_CPO_DUMP_RANK_BUFFER_AT_EPOCH must be an integer") from exc
    if dump_epoch <= 0 or epoch != dump_epoch:
        return None
    default_path = f"rank_buffer_dump_epoch{dump_epoch}.npz"
    return Path(os.environ.get("SR_CPO_DUMP_RANK_BUFFER_PATH", default_path))


def _maybe_dump_rank_buffer(
    *, state: TrainState, config: TrainConfig, epoch: int, print_fn: PrintFn
) -> None:
    dump_path = _rank_buffer_dump_target(epoch)
    if dump_path is None:
        return
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    save_rank_buffer_npz(
        str(dump_path),
        state.rank_buffer,
        cost_rank_horizon=config.cost_rank_horizon,
        cost_rank_num_candidates=config.cost_rank_num_candidates,
        cost_dense_prox_tau=config.cost_dense_prox_tau,
        epoch_dumped=epoch,
    )
    print_fn(f"RANK_BUFFER_DUMP={dump_path}")


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
        _maybe_dump_rank_buffer(
            state=state, config=config, epoch=epoch + 1, print_fn=print_fn
        )
        elapsed = time.perf_counter() - start
        merged_eval_metrics: dict[str, Array] = {}
        for std_scale, policy_evaluator in policy_evaluators:
            eval_key, epoch_eval_key = jax.random.split(eval_key)
            eval_params: Any
            if config.eval_counterfactual_action_probes:
                eval_params = (state.actor_params, state.cost_critic_params)
            else:
                eval_params = state.actor_params
            eval_metrics = policy_evaluator(eval_params, epoch_eval_key)
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
