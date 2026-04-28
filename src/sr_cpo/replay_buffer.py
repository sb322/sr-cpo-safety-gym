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


def sample_hindsight_transitions(
    buffer: ReplayBuffer,
    key: jax.Array,
    *,
    batch_size: int,
    goal_start: int = 0,
    goal_end: int | None = None,
    relative_goal: bool = False,
) -> Transition:
    """Uniformly samples transitions and relabels goals from future states."""

    if goal_end is None:
        goal_end = buffer.observations.shape[-1]
    episode_length = buffer.actions.shape[1]
    key_traj, key_t, key_offset = jax.random.split(key, 3)
    traj_idx = jax.random.randint(key_traj, (batch_size,), 0, buffer.size)
    step_idx = jax.random.randint(key_t, (batch_size,), 0, episode_length)
    max_offset = episode_length - step_idx
    offset = jax.random.randint(key_offset, (batch_size,), 1, max_offset + 1)
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
            "d_wall": d_wall,
            "hard_violation": hard,
            "trajectory_index": traj_idx,
            "step_index": step_idx,
            "future_index": future_idx,
        },
    )
