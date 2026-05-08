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
    )

    assert int(buffer.size) == 3
    assert int(buffer.write_index) == 1
    assert bool(jnp.all(buffer.valid))
    assert bool(jnp.any(jnp.all(buffer.states == states[-1], axis=-1)))
    assert not bool(jnp.any(jnp.all(buffer.states == states[0], axis=-1)))


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
