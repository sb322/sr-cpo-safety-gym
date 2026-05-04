"""Trajectory replay buffer with hindsight goal relabeling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from sr_cpo.env_wrappers import Transition
from sr_cpo.goal_space import _goal_from_obs


@struct.dataclass
class ReplayBuffer:
    """Ring buffer of fixed-length trajectories."""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    costs: jax.Array
    d_wall: jax.Array
    hard_violations: jax.Array
    write_index: jax.Array
    size: jax.Array


def _valid_transition_mask(buffer: ReplayBuffer) -> jax.Array:
    valid_traj = jnp.arange(buffer.costs.shape[0]) < buffer.size
    return jnp.broadcast_to(valid_traj[:, None], buffer.costs.shape)


def _risk_transition_mask(
    buffer: ReplayBuffer,
    *,
    hazard_lidar_threshold: float = 0.5,
) -> jax.Array:
    del hazard_lidar_threshold
    valid = _valid_transition_mask(buffer)
    hard_risk = (buffer.hard_violations > 0.5) & valid
    cost_risk = (buffer.costs > 0.0) & valid
    return jnp.where(jnp.any(hard_risk), hard_risk, cost_risk)


def replay_risky_available_fraction(
    buffer: ReplayBuffer,
    *,
    hazard_lidar_threshold: float = 0.5,
) -> jax.Array:
    """Fraction of valid replay transitions currently considered risky."""

    valid = _valid_transition_mask(buffer)
    risk = _risk_transition_mask(
        buffer, hazard_lidar_threshold=hazard_lidar_threshold
    )
    valid_count = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    return jnp.sum(risk.astype(jnp.float32)) / valid_count


def make_replay_buffer(
    *,
    capacity: int,
    episode_length: int,
    observation_dim: int,
    action_dim: int,
) -> ReplayBuffer:
    """Creates an empty trajectory replay buffer."""

    obs_shape = (capacity, episode_length + 1, observation_dim)
    step_shape = (capacity, episode_length)
    return ReplayBuffer(
        observations=jnp.zeros(obs_shape, dtype=jnp.float32),
        actions=jnp.zeros((*step_shape, action_dim), dtype=jnp.float32),
        rewards=jnp.zeros(step_shape, dtype=jnp.float32),
        discounts=jnp.zeros(step_shape, dtype=jnp.float32),
        costs=jnp.zeros(step_shape, dtype=jnp.float32),
        d_wall=jnp.zeros(step_shape, dtype=jnp.float32),
        hard_violations=jnp.zeros(step_shape, dtype=jnp.float32),
        write_index=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def insert_trajectory(
    buffer: ReplayBuffer,
    *,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    discounts: jax.Array,
    costs: jax.Array,
    d_wall: jax.Array | None = None,
    hard_violations: jax.Array | None = None,
) -> ReplayBuffer:
    """Inserts one complete trajectory, overwriting oldest rows as needed."""

    capacity = buffer.observations.shape[0]
    idx = buffer.write_index
    zeros = jnp.zeros_like(costs, dtype=jnp.float32)
    d_wall_arr = zeros if d_wall is None else jnp.asarray(d_wall, dtype=jnp.float32)
    hard_arr = (
        (jnp.asarray(costs) > 0.0).astype(jnp.float32)
        if hard_violations is None
        else jnp.asarray(hard_violations, dtype=jnp.float32)
    )
    next_write_index = (idx + 1) % capacity
    next_size = jnp.minimum(buffer.size + 1, capacity)
    return buffer.replace(
        observations=buffer.observations.at[idx].set(
            jnp.asarray(observations, dtype=jnp.float32)
        ),
        actions=buffer.actions.at[idx].set(jnp.asarray(actions, dtype=jnp.float32)),
        rewards=buffer.rewards.at[idx].set(jnp.asarray(rewards, dtype=jnp.float32)),
        discounts=buffer.discounts.at[idx].set(
            jnp.asarray(discounts, dtype=jnp.float32)
        ),
        costs=buffer.costs.at[idx].set(jnp.asarray(costs, dtype=jnp.float32)),
        d_wall=buffer.d_wall.at[idx].set(d_wall_arr),
        hard_violations=buffer.hard_violations.at[idx].set(hard_arr),
        write_index=next_write_index.astype(jnp.int32),
        size=next_size.astype(jnp.int32),
    )


def _discounted_cost_returns(
    costs: jax.Array,
    discounts: jax.Array,
    gamma: float,
) -> jax.Array:
    costs_t = jnp.swapaxes(jnp.asarray(costs, dtype=jnp.float32), 0, 1)
    discounts_t = jnp.swapaxes(jnp.asarray(discounts, dtype=jnp.float32), 0, 1)
    gamma_arr = jnp.asarray(gamma, dtype=jnp.float32)

    def scan_step(
        carry: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        cost_t, discount_t = inputs
        ret_t = cost_t + gamma_arr * discount_t * carry
        return ret_t, ret_t

    _, returns_rev = jax.lax.scan(
        scan_step,
        jnp.zeros((costs_t.shape[1],), dtype=jnp.float32),
        (costs_t[::-1], discounts_t[::-1]),
    )
    return jnp.swapaxes(returns_rev[::-1], 0, 1)


def sample_hindsight_transitions(
    buffer: ReplayBuffer,
    key: jax.Array,
    *,
    batch_size: int,
    goal_start: int = 0,
    goal_end: int | None = None,
    relative_goal: bool = False,
    cost_return_gamma: float = 0.99,
) -> Transition:
    """Uniformly samples transitions and relabels goals from future states."""

    if goal_end is None:
        goal_end = buffer.observations.shape[-1]
    episode_length = buffer.actions.shape[1]
    key_traj, key_t, key_offset = jax.random.split(key, 3)
    traj_idx = jax.random.randint(key_traj, (batch_size,), 0, buffer.size)
    step_idx = jax.random.randint(key_t, (batch_size,), 0, episode_length)
    return _sample_hindsight_by_index(
        buffer,
        key_offset,
        traj_idx=traj_idx,
        step_idx=step_idx,
        goal_start=goal_start,
        goal_end=goal_end,
        relative_goal=relative_goal,
        cost_return_gamma=cost_return_gamma,
    )


def _sample_hindsight_by_index(
    buffer: ReplayBuffer,
    key: jax.Array,
    *,
    traj_idx: jax.Array,
    step_idx: jax.Array,
    goal_start: int,
    goal_end: int,
    relative_goal: bool = False,
    cost_return_gamma: float = 0.99,
) -> Transition:
    """Samples fixed replay indices and relabels goals from future states."""

    batch_size = traj_idx.shape[0]
    episode_length = buffer.actions.shape[1]
    max_offset = episode_length - step_idx
    offset = jax.random.randint(key, (batch_size,), 1, max_offset + 1)
    future_idx = step_idx + offset

    obs = buffer.observations[traj_idx, step_idx]
    next_obs = buffer.observations[traj_idx, step_idx + 1]
    future_state = buffer.observations[traj_idx, future_idx]
    goal = _goal_from_obs(future_state, goal_start, goal_end - goal_start)
    if relative_goal:
        goal = goal - _goal_from_obs(obs, goal_start, goal_end - goal_start)
    action = buffer.actions[traj_idx, step_idx]
    reward = buffer.rewards[traj_idx, step_idx]
    discount = buffer.discounts[traj_idx, step_idx]
    cost = buffer.costs[traj_idx, step_idx]
    selected_costs = buffer.costs[traj_idx]
    selected_discounts = buffer.discounts[traj_idx]
    cost_returns = _discounted_cost_returns(
        selected_costs, selected_discounts, cost_return_gamma
    )
    cost_return = cost_returns[jnp.arange(batch_size), step_idx]
    d_wall = buffer.d_wall[traj_idx, step_idx]
    hard = buffer.hard_violations[traj_idx, step_idx]

    return Transition(
        observation=obs,
        action=action,
        reward=reward,
        discount=discount,
        extras={
            "goal": goal,
            "future_state": future_state,
            "next_state": next_obs,
            "cost": cost,
            "cost_return": cost_return,
            "d_wall": d_wall,
            "hard_violation": hard,
            "trajectory_index": traj_idx,
            "step_index": step_idx,
            "future_index": future_idx,
        },
    )


def _sample_flat_indices_from_mask(
    mask: jax.Array,
    key: jax.Array,
    *,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    episode_length = mask.shape[1]
    logits = jnp.where(jnp.ravel(mask), 0.0, -jnp.inf)
    flat_idx = jax.random.categorical(key, logits, shape=(batch_size,))
    return flat_idx // episode_length, flat_idx % episode_length


def _concat_transitions(first: Transition, second: Transition) -> Transition:
    extras = {
        name: jnp.concatenate([first.extras[name], second.extras[name]], axis=0)
        for name in first.extras
    }
    return Transition(
        observation=jnp.concatenate(
            [first.observation, second.observation], axis=0
        ),
        action=jnp.concatenate([first.action, second.action], axis=0),
        reward=jnp.concatenate([first.reward, second.reward], axis=0),
        discount=jnp.concatenate([first.discount, second.discount], axis=0),
        extras=extras,
    )


def sample_risk_biased_hindsight_transitions(
    buffer: ReplayBuffer,
    key: jax.Array,
    *,
    batch_size: int,
    risk_ratio: float,
    hazard_lidar_threshold: float = 0.5,
    min_fraction_available: float = 0.0,
    goal_start: int = 0,
    goal_end: int | None = None,
    relative_goal: bool = False,
    cost_return_gamma: float = 0.99,
) -> tuple[Transition, dict[str, jax.Array]]:
    """Samples a uniform/risky mixture for cost-critic-only updates."""

    if goal_end is None:
        goal_end = buffer.observations.shape[-1]
    risk_ratio_clamped = min(max(float(risk_ratio), 0.0), 1.0)
    risk_batch_size = int(round(batch_size * risk_ratio_clamped))
    risk_batch_size = min(max(risk_batch_size, 0), batch_size)
    uniform_batch_size = batch_size - risk_batch_size
    key_uniform, key_risk, key_uniform_goal, key_risk_goal = jax.random.split(key, 4)

    valid = _valid_transition_mask(buffer)
    risk = _risk_transition_mask(
        buffer, hazard_lidar_threshold=hazard_lidar_threshold
    )
    available_frac = replay_risky_available_fraction(
        buffer, hazard_lidar_threshold=hazard_lidar_threshold
    )
    enough_risk = available_frac >= jnp.asarray(
        min_fraction_available, dtype=jnp.float32
    )
    use_risk = (jnp.sum(risk.astype(jnp.float32)) > 0.0) & enough_risk

    uniform_traj, uniform_step = _sample_flat_indices_from_mask(
        valid, key_uniform, batch_size=uniform_batch_size
    )
    risk_mask = jnp.where(use_risk, risk, valid)
    risk_traj, risk_step = _sample_flat_indices_from_mask(
        risk_mask, key_risk, batch_size=risk_batch_size
    )

    uniform_batch = _sample_hindsight_by_index(
        buffer,
        key_uniform_goal,
        traj_idx=uniform_traj,
        step_idx=uniform_step,
        goal_start=goal_start,
        goal_end=goal_end,
        relative_goal=relative_goal,
        cost_return_gamma=cost_return_gamma,
    )
    risk_batch = _sample_hindsight_by_index(
        buffer,
        key_risk_goal,
        traj_idx=risk_traj,
        step_idx=risk_step,
        goal_start=goal_start,
        goal_end=goal_end,
        relative_goal=relative_goal,
        cost_return_gamma=cost_return_gamma,
    )
    batch = _concat_transitions(uniform_batch, risk_batch)

    def mean_or_zero(value: jax.Array, size: int) -> jax.Array:
        if size == 0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        return jnp.mean(jnp.asarray(value, dtype=jnp.float32))

    batch_hard = jnp.asarray(batch.extras["hard_violation"], dtype=jnp.float32)
    batch_cost = jnp.asarray(batch.extras["cost"], dtype=jnp.float32)
    batch_risk = (batch_hard > 0.5) | (batch_cost > 0.0)
    risk_ratio_actual = jnp.where(
        use_risk,
        jnp.asarray(risk_batch_size / max(batch_size, 1), dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    probes = {
        "cost_risk_replay_ratio_actual": risk_ratio_actual,
        "cost_risky_batch_frac": jnp.mean(batch_risk.astype(jnp.float32)),
        "cost_risky_available_frac": available_frac,
        "cost_risky_batch_mean_cost": jnp.where(
            use_risk,
            mean_or_zero(risk_batch.extras["cost"], risk_batch_size),
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        "cost_uniform_batch_mean_cost": jnp.where(
            use_risk,
            mean_or_zero(uniform_batch.extras["cost"], uniform_batch_size),
            mean_or_zero(batch.extras["cost"], batch_size),
        ),
    }
    return batch, probes
