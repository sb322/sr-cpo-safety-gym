from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from sr_cpo.env_wrappers import Transition
from sr_cpo.train import (
    ToyEnvState,
    TrainConfig,
    format_epoch_metrics,
    initialize_training,
    make_training_epoch,
    prefill_buffer,
    run_training,
)


def _tiny_config() -> TrainConfig:
    return TrainConfig(
        seed=0,
        epochs=1,
        steps_per_epoch=2,
        num_envs=2,
        unroll_length=3,
        prefill_steps=2,
        sgd_steps=1,
        batch_size=4,
        buffer_capacity=8,
        width=16,
        num_blocks=1,
        latent_dim=8,
    )


class FakeVectorAdapter:
    action_size = 2

    def __init__(self, *, num_envs: int, observation_dim: int = 5) -> None:
        self.num_envs = num_envs
        self.observation_dim = observation_dim

    def reset(self, key: jax.Array) -> tuple[ToyEnvState, Transition]:
        obs = jax.random.normal(
            key, (self.num_envs, self.observation_dim), dtype=jnp.float32
        )
        transition = Transition(
            observation=obs,
            action=jnp.zeros((self.num_envs, self.action_size), dtype=jnp.float32),
            reward=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            discount=jnp.ones((self.num_envs,), dtype=jnp.float32),
            extras={
                "next_state": obs,
                "cost": jnp.zeros((self.num_envs,), dtype=jnp.float32),
                "goal_dist": jnp.linalg.norm(obs[:, :3], axis=-1),
                "goal_reached": jnp.zeros((self.num_envs,), dtype=jnp.float32),
            },
        )
        return ToyEnvState(obs=obs), transition

    def step(
        self, state: ToyEnvState, action: jax.Array
    ) -> tuple[ToyEnvState, Transition]:
        action_pad = jnp.pad(
            action, ((0, 0), (0, self.observation_dim - self.action_size))
        )
        next_obs = 0.99 * state.obs + 0.05 * action_pad
        cost = jnp.maximum(0.0, 0.1 - jnp.linalg.norm(next_obs[:, :2], axis=-1))
        goal_dist = jnp.linalg.norm(next_obs[:, :3], axis=-1)
        next_state = ToyEnvState(obs=next_obs)
        transition = Transition(
            observation=state.obs,
            action=action,
            reward=-goal_dist,
            discount=jnp.ones((self.num_envs,), dtype=jnp.float32),
            extras={
                "state": state.obs,
                "next_state": next_obs,
                "cost": cost,
                "goal_dist": goal_dist,
                "goal_reached": (goal_dist <= 0.05).astype(jnp.float32),
                "d_wall": jnp.ones((self.num_envs,), dtype=jnp.float32),
                "hard_violation": (cost > 0.0).astype(jnp.float32),
            },
        )
        return next_state, transition


def test_training_epoch_runs_and_returns_finite_metrics() -> None:
    config = _tiny_config()
    state, objects = initialize_training(config)
    state = prefill_buffer(state, objects, config)
    training_epoch = make_training_epoch(objects, config)

    state, metrics = training_epoch(state)

    assert int(state.step) == config.num_envs * config.unroll_length * (
        config.prefill_steps + config.steps_per_epoch
    )
    for leaf in jax.tree_util.tree_leaves(metrics):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_training_epoch_can_collect_from_real_env_adapter_path() -> None:
    config = replace(_tiny_config(), use_real_env=True)
    adapter = FakeVectorAdapter(num_envs=config.num_envs, observation_dim=5)
    state, objects = initialize_training(config, env_adapter=adapter)

    assert state.replay.observations.shape[-1] == 5
    assert objects.env_adapter is adapter

    state = prefill_buffer(state, objects, config)
    training_epoch = make_training_epoch(objects, config)
    state, metrics = training_epoch(state)

    assert state.replay.actions.shape[-1] == adapter.action_size
    for leaf in jax.tree_util.tree_leaves(metrics):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_training_epoch_accepts_configured_goal_slice() -> None:
    config = replace(_tiny_config(), goal_start=1, goal_dim=2)
    state, objects = initialize_training(config)
    state = prefill_buffer(state, objects, config)
    training_epoch = make_training_epoch(objects, config)

    _, metrics = training_epoch(state)

    assert metrics["goal_dist"].shape == (config.steps_per_epoch,)
    for leaf in jax.tree_util.tree_leaves(metrics):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_initialize_training_rejects_invalid_goal_slice() -> None:
    config = replace(_tiny_config(), goal_start=5, goal_dim=2)

    try:
        initialize_training(config)
    except ValueError as exc:
        assert "goal_start + goal_dim" in str(exc)
    else:
        raise AssertionError("invalid goal slice should fail before training starts")


def test_initialize_training_uses_clipped_optimizers_by_default() -> None:
    config = _tiny_config()
    _, objects = initialize_training(config)

    huge_grads = {"w": jnp.asarray([1.0e20, 0.0], dtype=jnp.float32)}
    params = {"w": jnp.zeros((2,), dtype=jnp.float32)}
    updates, _ = objects.critic_optimizer.update(
        huge_grads, objects.critic_optimizer.init(params), params
    )

    update_norm = jnp.linalg.norm(updates["w"])
    assert bool(jnp.isfinite(update_norm))
    assert bool(update_norm < config.learning_rate * 2.0)


def test_default_cost_limit_matches_calibrated_dual_scale() -> None:
    assert TrainConfig().cost_limit == 0.0001


def test_run_training_prints_required_probe_sections() -> None:
    lines: list[str] = []
    result = run_training(_tiny_config(), print_fn=lines.append)
    output = "\n".join(lines)

    assert result["epochs"] == 1
    assert "[prefill probe]" in output
    assert "[epoch1 forensics]" in output
    assert "nan[obs_c=" in output
    assert "grad[c=" in output
    assert "actor[α=" in output
    assert "α_clip=" in output


def test_epoch_formatter_includes_static_diff_probe_markers() -> None:
    metrics = {
        "c_loss": jnp.asarray([1.0]),
        "c_accuracy": jnp.asarray([0.5]),
        "a_loss": jnp.asarray([-1.0]),
        "hard_viol": jnp.asarray([0.0]),
        "cost": jnp.asarray([0.0]),
        "rollout_reward": jnp.asarray([-1.2345]),
        "goal_dist": jnp.asarray([0.42]),
        "goal_reached": jnp.asarray([0.25]),
        "lambda_tilde": jnp.asarray([0.0]),
        "jc_hat": jnp.asarray([0.0]),
        "qc": jnp.asarray([0.0]),
        "cost_limit": jnp.asarray([0.0001]),
        "pid_error": jnp.asarray([-0.0001]),
        "pid_integral": jnp.asarray([-0.001]),
        "pid_raw_lambda": jnp.asarray([-0.001]),
        "nan_obs_critic": jnp.asarray([0.0]),
        "nan_sa_critic": jnp.asarray([0.0]),
        "nan_g_critic": jnp.asarray([0.0]),
        "nan_logits_critic": jnp.asarray([0.0]),
        "nan_sa_actor": jnp.asarray([0.0]),
        "nan_g_actor": jnp.asarray([0.0]),
        "nan_action_actor": jnp.asarray([0.0]),
        "nan_f_actor": jnp.asarray([0.0]),
        "sa_norm_min_critic": jnp.asarray([1.0]),
        "g_norm_min_critic": jnp.asarray([1.0]),
        "sa_norm_min_actor": jnp.asarray([1.0]),
        "g_norm_min_actor": jnp.asarray([1.0]),
        "action_abs_max": jnp.asarray([0.1]),
        "c_grad_nan": jnp.asarray([0.0]),
        "a_grad_nan": jnp.asarray([0.0]),
        "cc_grad_nan": jnp.asarray([0.0]),
        "c_grad_norm": jnp.asarray([1.0]),
        "a_grad_norm": jnp.asarray([1.0]),
        "cc_grad_norm": jnp.asarray([1.0]),
        "c_params_nan": jnp.asarray([0.0]),
        "a_params_nan": jnp.asarray([0.0]),
        "cc_params_nan": jnp.asarray([0.0]),
        "alpha_actor": jnp.asarray([1.0]),
        "log_prob_actor": jnp.asarray([-2.0]),
        "alpha_logprob_actor": jnp.asarray([-2.0]),
        "gaussian_logp_actor": jnp.asarray([-1.0]),
        "sat_correction_actor": jnp.asarray([1.0]),
        "log_std_mean_actor": jnp.asarray([0.0]),
        "f_term_actor": jnp.asarray([0.5]),
        "qc_actor": jnp.asarray([0.25]),
        "lambda_qc_actor": jnp.asarray([0.125]),
        "nu_c": jnp.asarray([0.01]),
        "alpha_clip": jnp.asarray([1.0]),
    }

    text = format_epoch_metrics(0, 1, metrics, steps=10, elapsed=0.1)

    assert "nan[obs_c=0" in text
    assert "grad[c=0/" in text
    assert "actor[α=1.0000" in text
    assert "rew=-1.2345" in text
    assert "gdist=0.4200" in text
    assert "reached=0.2500" in text
    assert "limit=1.00e-04" in text
    assert "pid_err=-1.00e-04" in text
    assert "S=-1.00e-03" in text
    assert "λraw=-1.00e-03" in text
    assert "λQc_a=1.25e-01" in text
    assert "nu_c=1.0e-02" in text
