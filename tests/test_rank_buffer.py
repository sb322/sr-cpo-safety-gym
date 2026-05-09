import jax
import jax.numpy as jnp

from sr_cpo.rank_buffer import (
    insert_rank_examples,
    make_rank_buffer,
    sample_rank_batch,
)


def test_rank_buffer_insert_overwrite_wraps() -> None:
    buffer = make_rank_buffer(
        capacity=3,
        state_dim=2,
        action_dim=1,
        goal_dim=2,
        num_candidates=2,
    )
    states = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    actions = jnp.arange(8, dtype=jnp.float32).reshape(4, 2, 1)
    goals = states + 100.0
    dense = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    sparse = dense + 10.0

    buffer = insert_rank_examples(
        buffer,
        states=states,
        candidate_actions=actions,
        goals=goals,
        dense_labels=dense,
        sparse_labels=sparse,
        state_info_steps=jnp.asarray([10.0, 11.0, 12.0, 13.0]),
        state_done=jnp.asarray([0.0, 0.0, 1.0, 0.0]),
        unroll_index_at_collection=jnp.asarray([0, 0, 1, 1]),
    )

    assert int(buffer.size) == 3
    assert int(buffer.write_index) == 1
    assert bool(jnp.all(buffer.valid))
    assert bool(jnp.any(jnp.all(buffer.states == states[-1], axis=-1)))
    assert not bool(jnp.any(jnp.all(buffer.states == states[0], axis=-1)))
    assert bool(
        jnp.array_equal(
            jnp.sort(buffer.state_info_steps), jnp.asarray([11.0, 12.0, 13.0])
        )
    )
    assert bool(jnp.any(buffer.state_done == 1.0))
    assert bool(jnp.any(buffer.unroll_index_at_collection == 1))


def test_sample_rank_batch_marks_empty_rows_invalid() -> None:
    buffer = make_rank_buffer(
        capacity=4,
        state_dim=2,
        action_dim=1,
        goal_dim=2,
        num_candidates=2,
    )

    batch = sample_rank_batch(buffer, jax.random.PRNGKey(0), batch_size=5)

    assert batch.states.shape == (5, 2)
    assert not bool(jnp.any(batch.valid))


def test_rank_buffer_preserves_inserted_valid_mask() -> None:
    buffer = make_rank_buffer(
        capacity=3,
        state_dim=2,
        action_dim=1,
        goal_dim=2,
        num_candidates=2,
    )
    states = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    actions = jnp.zeros((3, 2, 1), dtype=jnp.float32)
    labels = jnp.ones((3, 2), dtype=jnp.float32)

    buffer = insert_rank_examples(
        buffer,
        states=states,
        candidate_actions=actions,
        goals=states,
        dense_labels=labels,
        sparse_labels=labels,
        valid=jnp.asarray([True, False, True]),
    )

    assert bool(jnp.array_equal(buffer.valid, jnp.asarray([True, False, True])))


def test_rank_buffer_preserves_source_state_metadata() -> None:
    buffer = make_rank_buffer(
        capacity=3,
        state_dim=2,
        action_dim=1,
        goal_dim=2,
        num_candidates=2,
    )
    states = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    actions = jnp.zeros((3, 2, 1), dtype=jnp.float32)
    labels = jnp.ones((3, 2), dtype=jnp.float32)

    buffer = insert_rank_examples(
        buffer,
        states=states,
        candidate_actions=actions,
        goals=states,
        dense_labels=labels,
        sparse_labels=labels,
        state_info_steps=jnp.asarray([3.0, 4.0, 5.0], dtype=jnp.float32),
        state_done=jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
        unroll_index_at_collection=jnp.asarray([1, 1, 2], dtype=jnp.int32),
    )

    assert bool(
        jnp.array_equal(buffer.state_info_steps, jnp.asarray([3.0, 4.0, 5.0]))
    )
    assert bool(jnp.array_equal(buffer.state_done, jnp.asarray([0.0, 1.0, 0.0])))
    assert bool(
        jnp.array_equal(buffer.unroll_index_at_collection, jnp.asarray([1, 1, 2]))
    )
