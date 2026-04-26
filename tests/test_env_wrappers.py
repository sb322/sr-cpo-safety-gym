from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State

from sr_cpo.env_wrappers import SafeLearningGoToGoalAdapter


class FakeGoToGoal(Env):
    documented_step_cost: float = 0.25

    @property
    def observation_size(self) -> int:
        return 5

    @property
    def action_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> State:
        obs = jnp.linspace(0.0, 1.0, self.observation_size, dtype=jnp.float32)
        info: dict[str, Any] = {
            "rng": rng,
            "cost": jnp.zeros((), dtype=jnp.float32),
            "goal_reached": jnp.zeros((), dtype=jnp.float32),
        }
        return State(
            pipeline_state=None,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics={},
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        padded_action = jnp.pad(action, (0, self.observation_size - self.action_size))
        obs = state.obs + padded_action.astype(jnp.float32)
        cost = jnp.asarray(self.documented_step_cost, dtype=jnp.float32)
        info = dict(state.info)
        info["cost"] = cost
        return state.replace(
            obs=obs,
            reward=jnp.ones((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            info=info,
        )


def test_safe_learning_go_to_goal_adapter_reset_uses_info_cost() -> None:
    num_envs = 8
    adapter = SafeLearningGoToGoalAdapter(env=FakeGoToGoal(), num_envs=num_envs)

    state, reset_transition = adapter.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (num_envs, 5)
    assert reset_transition.observation.shape == (num_envs, 5)
    assert reset_transition.action.shape == (num_envs, 2)
    assert bool(jnp.all(jnp.isfinite(reset_transition.observation)))
    assert bool(jnp.all(jnp.isfinite(reset_transition.extras["cost"])))
    assert bool(jnp.all(reset_transition.extras["cost"] >= 0.0))
    assert bool(jnp.allclose(reset_transition.extras["cost"], state.info["cost"]))
    assert bool(jnp.allclose(reset_transition.extras["cost"], 0.0))


def test_safe_learning_go_to_goal_adapter_step_uses_next_info_cost() -> None:
    num_envs = 8
    fake_env = FakeGoToGoal()
    adapter = SafeLearningGoToGoalAdapter(env=fake_env, num_envs=num_envs)

    state, _ = adapter.reset(jax.random.PRNGKey(0))
    action = jnp.zeros((num_envs, adapter.action_size), dtype=jnp.float32)
    next_state, transition = adapter.step(state, action)

    assert next_state.obs.shape == (num_envs, 5)
    assert transition.observation.shape == (num_envs, 5)
    assert transition.action.shape == (num_envs, 2)
    assert transition.reward.shape == (num_envs,)
    assert transition.discount.shape == (num_envs,)
    assert transition.extras["next_state"].shape == (num_envs, 5)
    assert transition.extras["hard_violation"].shape == (num_envs,)
    assert transition.extras["state_extras"]["truncation"].shape == (num_envs,)
    assert bool(jnp.all(jnp.isfinite(next_state.obs)))
    assert bool(jnp.all(jnp.isfinite(transition.extras["cost"])))
    assert bool(jnp.all(jnp.isfinite(transition.discount)))
    assert bool(jnp.all(transition.extras["cost"] >= 0.0))
    discount_is_binary = jnp.logical_or(
        transition.discount == 0.0, transition.discount == 1.0
    )
    assert bool(jnp.all(discount_is_binary))
    assert bool(jnp.allclose(transition.extras["cost"], next_state.info["cost"]))
    assert bool(jnp.allclose(transition.extras["cost"], fake_env.documented_step_cost))
