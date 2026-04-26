from __future__ import annotations

import jax
import jax.numpy as jnp

from sr_cpo.train import (
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
        "lambda_tilde": jnp.asarray([0.0]),
        "jc_hat": jnp.asarray([0.0]),
        "qc": jnp.asarray([0.0]),
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
        "alpha_clip": jnp.asarray([1.0]),
    }

    text = format_epoch_metrics(0, 1, metrics, steps=10, elapsed=0.1)

    assert "nan[obs_c=0" in text
    assert "grad[c=0/" in text
    assert "actor[α=1.0000" in text
