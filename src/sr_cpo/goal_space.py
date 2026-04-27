"""Goal-space helpers shared by rollout and hindsight relabeling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def _goal_from_obs(obs: Array, goal_start: int, goal_dim: int) -> Array:
    """Extracts the configured goal representation from observations."""

    return jnp.asarray(
        obs[..., goal_start : goal_start + goal_dim], dtype=jnp.float32
    )


def _assert_goal_shape(goal: Array, goal_dim: int, *, context: str) -> None:
    """Fails early if actor and critic goals drift into different spaces."""

    if not goal.shape or goal.shape[-1] != goal_dim:
        raise ValueError(
            f"{context} goal dim {goal.shape[-1] if goal.shape else 'scalar'} "
            f"does not match TrainConfig.goal_dim={goal_dim}; actor rollout "
            "and hindsight critic goals must use the same representation."
        )
