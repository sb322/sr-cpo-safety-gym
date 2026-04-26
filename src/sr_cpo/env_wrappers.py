"""Brax/MJX safety-navigation environment adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training
from flax import struct


@struct.dataclass
class Transition:
    """Canonical SR-CPO transition emitted by environment adapters."""

    observation: jax.Array
    action: jax.Array
    reward: jax.Array
    discount: jax.Array
    extras: dict[str, Any]


def _load_safe_learning_go_to_goal(**env_kwargs: Any) -> Any:
    try:
        from ss2r.benchmark_suites.safety_gym.go_to_goal import GoToGoal
    except ImportError as exc:  # pragma: no cover - exercised only off-repo
        raise ImportError(
            "safe-learning's ss2r package is required for the real GoToGoal env. "
            "Install lasgroup/safe-learning or put it on PYTHONPATH."
        ) from exc
    return GoToGoal(**env_kwargs)


def _info_array(
    info: Mapping[str, Any], key: str, default: jax.Array | None = None
) -> jax.Array:
    if key in info:
        return jnp.asarray(info[key])
    if default is not None:
        return jnp.asarray(default)
    raise KeyError(f"safe-learning GoToGoal state.info is missing {key!r}")


def _cost_from_info(info: Mapping[str, Any]) -> jax.Array:
    # safe-learning sets info["cost"] after step at:
    # https://github.com/lasgroup/safe-learning/blob/aba4b94b91dbfebe48a45c3b371f9a6f8fbed606/ss2r/benchmark_suites/safety_gym/go_to_goal.py#L420-L424
    # This implies Bellman-B: cost is c(s_{t+1}), not c(s_t).
    # This differs from cost-at-state Bellman-A by a gamma scaling on J_c.
    # Cost-critic targets use transition.extras["cost"] as c(s_{t+1}):
    # y_t = c(s_{t+1}) + gamma * (1 - d) * Q_c_bar(s_{t+1}, a', g).
    return jnp.asarray(_info_array(info, "cost"), dtype=jnp.float32)


class SafeLearningGoToGoalAdapter:
    """Vectorized adapter around safe-learning's Brax/MJX GoToGoal task."""

    def __init__(
        self,
        env: Any | None = None,
        *,
        num_envs: int,
        episode_length: int = 1000,
        action_repeat: int = 1,
        **env_kwargs: Any,
    ) -> None:
        base_env = (
            env if env is not None else _load_safe_learning_go_to_goal(**env_kwargs)
        )
        episodic_env = training.EpisodeWrapper(
            base_env, episode_length=episode_length, action_repeat=action_repeat
        )
        self.env = training.VmapWrapper(episodic_env, batch_size=num_envs)
        self.num_envs = num_envs
        self.episode_length = episode_length

    @property
    def action_size(self) -> int:
        return int(self.env.action_size)

    @property
    def observation_size(self) -> Any:
        return self.env.observation_size

    def reset(self, rng: jax.Array | int) -> tuple[Any, Transition]:
        key = jax.random.PRNGKey(rng) if isinstance(rng, int) else rng
        state = self.env.reset(key)
        return state, self._transition_from_reset(state)

    def step(self, state: Any, action: jax.Array) -> tuple[Any, Transition]:
        action = jnp.asarray(action, dtype=jnp.float32)
        next_state = self.env.step(state, action)
        transition = self._transition_from_step(state, action, next_state)
        return next_state, transition

    def _transition_from_reset(self, state: Any) -> Transition:
        obs = jnp.asarray(state.obs, dtype=jnp.float32)
        action = jnp.zeros((*obs.shape[:-1], self.action_size), dtype=jnp.float32)
        reward = jnp.asarray(state.reward, dtype=jnp.float32)
        discount = jnp.ones_like(reward, dtype=jnp.float32)
        cost = _cost_from_info(state.info)
        extras = self._extras(
            state_info=state.info,
            state_obs=obs,
            next_obs=obs,
            cost=cost,
        )
        return Transition(obs, action, reward, discount, extras)

    def _transition_from_step(
        self, state: Any, action: jax.Array, next_state: Any
    ) -> Transition:
        obs = jnp.asarray(state.obs, dtype=jnp.float32)
        next_obs = jnp.asarray(next_state.obs, dtype=jnp.float32)
        reward = jnp.asarray(next_state.reward, dtype=jnp.float32)
        discount = 1.0 - jnp.asarray(next_state.done, dtype=jnp.float32)
        cost = _cost_from_info(next_state.info)
        extras = self._extras(
            state_info=next_state.info,
            state_obs=obs,
            next_obs=next_obs,
            cost=cost,
        )
        return Transition(obs, action, reward, discount, extras)

    @staticmethod
    def _extras(
        *,
        state_info: Mapping[str, Any],
        state_obs: jax.Array,
        next_obs: jax.Array,
        cost: jax.Array,
    ) -> dict[str, Any]:
        zeros = jnp.zeros_like(cost, dtype=jnp.float32)
        truncation = _info_array(state_info, "truncation", zeros).astype(jnp.float32)
        seed = _info_array(state_info, "seed", _info_array(state_info, "rng", zeros))
        # GoToGoal does not emit a wall-distance diagnostic; keep a finite placeholder.
        d_wall = _info_array(state_info, "d_wall", zeros).astype(jnp.float32)
        return {
            "state": state_obs,
            "next_state": next_obs,
            "cost": cost,
            "d_wall": d_wall,
            "hard_violation": (cost > 0.0).astype(jnp.float32),
            "state_extras": {
                "seed": seed,
                "truncation": truncation,
            },
        }


def make_safe_learning_go_to_goal(
    *, num_envs: int, episode_length: int = 1000, **env_kwargs: Any
) -> SafeLearningGoToGoalAdapter:
    """Creates the vectorized safe-learning GoToGoal adapter."""

    return SafeLearningGoToGoalAdapter(
        num_envs=num_envs,
        episode_length=episode_length,
        **env_kwargs,
    )
