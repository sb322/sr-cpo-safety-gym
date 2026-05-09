"""Small replay buffer for auxiliary cost-rank labels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class RankBuffer:
    """Ring buffer of per-state candidate-action cost labels."""

    states: jax.Array
    candidate_actions: jax.Array
    goals: jax.Array
    dense_labels: jax.Array
    sparse_labels: jax.Array
    state_info_steps: jax.Array
    state_done: jax.Array
    unroll_index_at_collection: jax.Array
    valid: jax.Array
    write_index: jax.Array
    size: jax.Array


@struct.dataclass
class RankBatch:
    """Mini-batch sampled from ``RankBuffer``."""

    states: jax.Array
    candidate_actions: jax.Array
    goals: jax.Array
    dense_labels: jax.Array
    sparse_labels: jax.Array
    valid: jax.Array


def make_rank_buffer(
    *,
    capacity: int,
    state_dim: int,
    action_dim: int,
    goal_dim: int,
    num_candidates: int,
) -> RankBuffer:
    """Creates an empty rank-label ring buffer."""

    return RankBuffer(
        states=jnp.zeros((capacity, state_dim), dtype=jnp.float32),
        candidate_actions=jnp.zeros(
            (capacity, num_candidates, action_dim), dtype=jnp.float32
        ),
        goals=jnp.zeros((capacity, goal_dim), dtype=jnp.float32),
        dense_labels=jnp.zeros((capacity, num_candidates), dtype=jnp.float32),
        sparse_labels=jnp.zeros((capacity, num_candidates), dtype=jnp.float32),
        state_info_steps=jnp.zeros((capacity,), dtype=jnp.float32),
        state_done=jnp.zeros((capacity,), dtype=jnp.float32),
        unroll_index_at_collection=jnp.zeros((capacity,), dtype=jnp.int32),
        valid=jnp.zeros((capacity,), dtype=bool),
        write_index=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def insert_rank_examples(
    buffer: RankBuffer,
    *,
    states: jax.Array,
    candidate_actions: jax.Array,
    goals: jax.Array,
    dense_labels: jax.Array,
    sparse_labels: jax.Array,
    state_info_steps: jax.Array | None = None,
    state_done: jax.Array | None = None,
    unroll_index_at_collection: jax.Array | None = None,
    valid: jax.Array | None = None,
) -> RankBuffer:
    """Inserts a batch of labeled candidate sets into the ring buffer."""

    capacity = buffer.states.shape[0]
    num_examples = states.shape[0]
    valid_arr = (
        jnp.ones((num_examples,), dtype=bool)
        if valid is None
        else jnp.asarray(valid, dtype=bool)
    )
    steps_arr = (
        jnp.zeros((num_examples,), dtype=jnp.float32)
        if state_info_steps is None
        else jnp.asarray(state_info_steps, dtype=jnp.float32)
    )
    done_arr = (
        jnp.zeros((num_examples,), dtype=jnp.float32)
        if state_done is None
        else jnp.asarray(state_done, dtype=jnp.float32)
    )
    unroll_arr = (
        jnp.zeros((num_examples,), dtype=jnp.int32)
        if unroll_index_at_collection is None
        else jnp.asarray(unroll_index_at_collection, dtype=jnp.int32)
    )
    offsets = jnp.arange(num_examples, dtype=jnp.int32)
    indices = (buffer.write_index + offsets) % capacity
    next_write_index = (buffer.write_index + num_examples) % capacity
    next_size = jnp.minimum(buffer.size + num_examples, capacity)
    return buffer.replace(
        states=buffer.states.at[indices].set(jnp.asarray(states, dtype=jnp.float32)),
        candidate_actions=buffer.candidate_actions.at[indices].set(
            jnp.asarray(candidate_actions, dtype=jnp.float32)
        ),
        goals=buffer.goals.at[indices].set(jnp.asarray(goals, dtype=jnp.float32)),
        dense_labels=buffer.dense_labels.at[indices].set(
            jnp.asarray(dense_labels, dtype=jnp.float32)
        ),
        sparse_labels=buffer.sparse_labels.at[indices].set(
            jnp.asarray(sparse_labels, dtype=jnp.float32)
        ),
        state_info_steps=buffer.state_info_steps.at[indices].set(steps_arr),
        state_done=buffer.state_done.at[indices].set(done_arr),
        unroll_index_at_collection=buffer.unroll_index_at_collection.at[indices].set(
            unroll_arr
        ),
        valid=buffer.valid.at[indices].set(valid_arr),
        write_index=next_write_index.astype(jnp.int32),
        size=next_size.astype(jnp.int32),
    )


def sample_rank_batch(
    buffer: RankBuffer,
    key: jax.Array,
    *,
    batch_size: int,
) -> RankBatch:
    """Uniformly samples rank examples; invalid rows are marked in the batch."""

    high = jnp.maximum(buffer.size, 1)
    indices = jax.random.randint(key, (batch_size,), 0, high)
    valid = (indices < buffer.size) & buffer.valid[indices]
    return RankBatch(
        states=buffer.states[indices],
        candidate_actions=buffer.candidate_actions[indices],
        goals=buffer.goals[indices],
        dense_labels=buffer.dense_labels[indices],
        sparse_labels=buffer.sparse_labels[indices],
        valid=valid,
    )


def save_rank_buffer_npz(
    path: str,
    buffer: RankBuffer,
    *,
    cost_rank_horizon: int,
    cost_rank_num_candidates: int,
    cost_dense_prox_tau: float,
    epoch_dumped: int,
) -> None:
    """Serializes a RankBuffer snapshot for offline oracle refitting."""

    np.savez(
        path,
        states=np.asarray(buffer.states),
        candidate_actions=np.asarray(buffer.candidate_actions),
        goals=np.asarray(buffer.goals),
        labels_dense_h50=np.asarray(buffer.dense_labels),
        labels_sparse_h50=np.asarray(buffer.sparse_labels),
        state_info_steps=np.asarray(buffer.state_info_steps),
        state_done=np.asarray(buffer.state_done),
        unroll_index_at_collection=np.asarray(buffer.unroll_index_at_collection),
        valid_mask=np.asarray(buffer.valid),
        cost_rank_horizon=np.asarray(cost_rank_horizon, dtype=np.int32),
        cost_rank_num_candidates=np.asarray(cost_rank_num_candidates, dtype=np.int32),
        cost_dense_prox_tau=np.asarray(cost_dense_prox_tau, dtype=np.float32),
        epoch_dumped=np.asarray(epoch_dumped, dtype=np.int32),
    )
