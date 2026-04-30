from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State

from sr_cpo.env_wrappers import (
    SafeLearningGoToGoalAdapter,
    _load_safe_learning_go_to_goal_class,
)


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
            "last_goal_dist": jnp.ones((), dtype=jnp.float32),
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
        cost = jnp.asarray(
            self.documented_step_cost + 0.1 * jnp.sum(jnp.abs(action)),
            dtype=jnp.float32,
        )
        info = dict(state.info)
        info["cost"] = cost
        info["last_goal_dist"] = jnp.asarray(0.5, dtype=jnp.float32)
        info["goal_reached"] = jnp.ones((), dtype=jnp.float32)
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
    assert bool(jnp.allclose(reset_transition.extras["goal_dist"], 1.0))
    assert bool(jnp.allclose(reset_transition.extras["goal_reached"], 0.0))


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
    assert transition.extras["goal_dist"].shape == (num_envs,)
    assert transition.extras["goal_reached"].shape == (num_envs,)
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
    assert bool(jnp.allclose(transition.extras["goal_dist"], 0.5))
    assert bool(jnp.allclose(transition.extras["goal_reached"], 1.0))


def test_safe_learning_go_to_goal_adapter_can_probe_counterfactual_costs() -> None:
    num_envs = 8
    fake_env = FakeGoToGoal()
    adapter = SafeLearningGoToGoalAdapter(
        env=fake_env,
        num_envs=num_envs,
        probe_counterfactual_costs=True,
    )

    state, _ = adapter.reset(jax.random.PRNGKey(0))
    action = jnp.ones((num_envs, adapter.action_size), dtype=jnp.float32)
    _, transition = adapter.step(state, action)

    assert transition.extras["cost_zero_action"].shape == (num_envs,)
    assert transition.extras["cost_neg_action"].shape == (num_envs,)
    assert bool(
        jnp.allclose(
            transition.extras["cost_zero_action"], fake_env.documented_step_cost
        )
    )
    assert bool(
        jnp.all(transition.extras["cost"] > transition.extras["cost_zero_action"])
    )
    assert bool(
        jnp.allclose(transition.extras["cost_neg_action"], transition.extras["cost"])
    )


def test_xy_goal_mode_uses_robot_and_target_xy_goal_space() -> None:
    adapter = object.__new__(SafeLearningGoToGoalAdapter)
    adapter.goal_mode = "xy"
    adapter.base_env = SimpleNamespace(_robot_body_id=1, _goal_mocap_id=0)
    obs = jnp.arange(2 * 55, dtype=jnp.float32).reshape(2, 55)
    xpos = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    xpos = xpos.at[:, 1, :2].set(jnp.asarray([[0.25, -0.5], [1.0, 1.5]]))
    mocap_pos = jnp.zeros((2, 1, 3), dtype=jnp.float32)
    mocap_pos = mocap_pos.at[:, 0, :2].set(jnp.asarray([[1.0, 2.0], [-2.0, 0.5]]))
    state = SimpleNamespace(
        obs=obs,
        data=SimpleNamespace(xpos=xpos, mocap_pos=mocap_pos),
    )

    state_obs = adapter._state_observation(state)

    assert state_obs.shape == (2, 57)
    assert bool(jnp.allclose(state_obs[:, :55], obs))
    assert bool(jnp.allclose(state_obs[:, 55:57], adapter.achieved_goal(state)))
    assert bool(jnp.allclose(adapter.desired_goal(state), mocap_pos[:, 0, :2]))

    adapter.goal_mode = "relative_xy"

    assert bool(
        jnp.allclose(adapter.desired_goal(state), mocap_pos[:, 0, :2] - xpos[:, 1, :2])
    )


def test_xy_goal_mode_can_mask_native_goal_lidar_from_state() -> None:
    adapter = object.__new__(SafeLearningGoToGoalAdapter)
    adapter.goal_mode = "xy"
    adapter.mask_native_goal_lidar = True
    adapter.base_env = SimpleNamespace(_robot_body_id=1, _goal_mocap_id=0)
    obs = jnp.ones((2, 55), dtype=jnp.float32)
    xpos = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    mocap_pos = jnp.zeros((2, 1, 3), dtype=jnp.float32)
    state = SimpleNamespace(
        obs=obs,
        data=SimpleNamespace(xpos=xpos, mocap_pos=mocap_pos),
    )

    state_obs = adapter._state_observation(state)

    assert state_obs.shape == (2, 57)
    assert bool(jnp.allclose(state_obs[:, :16], 1.0))
    assert bool(jnp.allclose(state_obs[:, 16:32], 0.0))
    assert bool(jnp.allclose(state_obs[:, 32:55], 1.0))
    assert bool(jnp.allclose(state_obs[:, 55:57], adapter.achieved_goal(state)))


def test_safe_learning_loader_bypasses_broad_benchmark_suite_init(
    tmp_path, monkeypatch
) -> None:
    package_root = tmp_path / "site"
    safety_root = package_root / "ss2r" / "benchmark_suites" / "safety_gym"
    safety_root.mkdir(parents=True)
    (package_root / "ss2r" / "__init__.py").write_text("")
    (package_root / "ss2r" / "benchmark_suites" / "__init__.py").write_text(
        "raise RuntimeError('broad init should not run')\n"
    )
    (safety_root / "__init__.py").write_text("")
    (safety_root / "go_to_goal.py").write_text(
        "class GoToGoal:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
    )
    monkeypatch.setattr("site.getsitepackages", lambda: [str(package_root)])
    monkeypatch.setattr("site.getusersitepackages", lambda: str(package_root))

    loaded = _load_safe_learning_go_to_goal_class()
    env = loaded(example=True)

    assert env.kwargs == {"example": True}
