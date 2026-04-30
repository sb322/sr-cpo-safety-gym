import jax
import jax.numpy as jnp

from sr_cpo.replay_buffer import (
    insert_trajectory,
    make_replay_buffer,
    sample_hindsight_transitions,
)

TRAJECTORY_KEYS = (
    "observations",
    "actions",
    "rewards",
    "discounts",
    "costs",
    "d_wall",
    "hard_violations",
)


def _trajectory(offset: float) -> tuple[jax.Array, ...]:
    observations = jnp.arange(5 * 3, dtype=jnp.float32).reshape(5, 3) + offset
    actions = jnp.ones((4, 2), dtype=jnp.float32) * offset
    rewards = jnp.arange(4, dtype=jnp.float32)
    discounts = jnp.ones((4,), dtype=jnp.float32)
    costs = jnp.asarray([0.0, 1.0, 0.0, 0.5], dtype=jnp.float32)
    d_wall = jnp.linspace(1.0, 0.25, 4, dtype=jnp.float32)
    hard = (costs > 0.0).astype(jnp.float32)
    return observations, actions, rewards, discounts, costs, d_wall, hard


def test_replay_buffer_insert_wraps_and_tracks_size() -> None:
    buffer = make_replay_buffer(
        capacity=2, episode_length=4, observation_dim=3, action_dim=2
    )
    for i in range(3):
        buffer = insert_trajectory(
            buffer, **dict(zip(TRAJECTORY_KEYS, _trajectory(float(i)), strict=True))
        )

    assert int(buffer.size) == 2
    assert int(buffer.write_index) == 1
    assert bool(jnp.allclose(buffer.observations[0, 0, 0], 2.0))


def test_hindsight_sample_uses_configured_future_goal_slice() -> None:
    buffer = make_replay_buffer(
        capacity=3, episode_length=4, observation_dim=3, action_dim=2
    )
    for i in range(3):
        buffer = insert_trajectory(
            buffer, **dict(zip(TRAJECTORY_KEYS, _trajectory(float(i)), strict=True))
        )

    batch = sample_hindsight_transitions(
        buffer, jax.random.PRNGKey(0), batch_size=8, goal_start=1, goal_end=3
    )

    assert batch.observation.shape == (8, 3)
    assert batch.action.shape == (8, 2)
    assert batch.extras["goal"].shape == (8, 2)
    assert bool(jnp.all(batch.extras["future_index"] > batch.extras["step_index"]))
    assert bool(
        jnp.allclose(batch.extras["goal"], batch.extras["future_state"][:, 1:3])
    )
    assert bool(jnp.all(jnp.isfinite(batch.extras["cost"])))
    assert batch.extras["cost_return"].shape == (8,)
    assert bool(jnp.all(jnp.isfinite(batch.extras["cost_return"])))


def test_hindsight_sample_can_use_relative_future_goal_slice() -> None:
    buffer = make_replay_buffer(
        capacity=3, episode_length=4, observation_dim=3, action_dim=2
    )
    for i in range(3):
        buffer = insert_trajectory(
            buffer, **dict(zip(TRAJECTORY_KEYS, _trajectory(float(i)), strict=True))
        )

    batch = sample_hindsight_transitions(
        buffer,
        jax.random.PRNGKey(1),
        batch_size=8,
        goal_start=1,
        goal_end=3,
        relative_goal=True,
    )

    expected = batch.extras["future_state"][:, 1:3] - batch.observation[:, 1:3]
    assert batch.extras["goal"].shape == (8, 2)
    assert bool(jnp.allclose(batch.extras["goal"], expected))


def test_hindsight_sample_exposes_discounted_cost_return() -> None:
    buffer = make_replay_buffer(
        capacity=1, episode_length=4, observation_dim=3, action_dim=2
    )
    buffer = insert_trajectory(
        buffer, **dict(zip(TRAJECTORY_KEYS, _trajectory(0.0), strict=True))
    )

    batch = sample_hindsight_transitions(
        buffer,
        jax.random.PRNGKey(2),
        batch_size=8,
        goal_start=1,
        goal_end=3,
        cost_return_gamma=0.5,
    )

    costs = buffer.costs[batch.extras["trajectory_index"]]
    expected_returns = []
    for sample_idx, step_idx in enumerate(batch.extras["step_index"]):
        ret = 0.0
        weight = 1.0
        for cost in costs[sample_idx, int(step_idx) :]:
            ret += weight * float(cost)
            weight *= 0.5
        expected_returns.append(ret)

    expected = jnp.asarray(expected_returns, dtype=jnp.float32)
    assert bool(jnp.allclose(batch.extras["cost_return"], expected))
