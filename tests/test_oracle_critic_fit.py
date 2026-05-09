"""Tests for oracle critic-fit reporting helpers."""

from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest


def test_bootstrap_report_brackets_population_mean(tmp_path: Path) -> None:
    spearman = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    top1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    path = tmp_path / "oracle_fake.npz"
    np.savez(
        path,
        dense_H20_centered_rank_normal_spearman=spearman,
        dense_H20_centered_rank_normal_top1=top1,
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "oracle_critic_fit.py"
    result = subprocess.run(
        [sys.executable, str(script), "--report-bootstrap", str(path), "--samples", "1000"],
        check=True,
        capture_output=True,
        text=True,
    )
    out = result.stdout

    assert "dense_H20_centered_rank_normal_spearman" in out
    assert "CI=[" in out
    match = re.search(r"spearman: mean=([-0-9.eE]+) CI=\[([-0-9.eE]+), ([-0-9.eE]+)\]", out)
    assert match is not None
    mean, low, high = map(float, match.groups())
    assert mean == pytest.approx(float(np.mean(spearman)))
    assert low <= mean <= high


def _jax_helpers():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from scripts.oracle_critic_fit import _done_aware_return, _shuffle_rows

    return jax, jnp, _done_aware_return, _shuffle_rows


def test_shuffle_rows_preserves_marginal_distribution() -> None:
    jax, jnp, _, _shuffle_rows = _jax_helpers()
    goals = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    shuffled = _shuffle_rows(goals, jax.random.PRNGKey(0))
    assert shuffled.shape == goals.shape
    assert not bool(jnp.all(shuffled == goals))
    assert sorted(map(tuple, np.asarray(shuffled))) == sorted(map(tuple, np.asarray(goals)))


def test_done_aware_return_zeroes_after_done() -> None:
    _, jnp, _done_aware_return, _ = _jax_helpers()
    costs = jnp.ones((5, 1), dtype=jnp.float32)
    discounts = jnp.asarray([[1.0], [1.0], [0.0], [1.0], [1.0]], dtype=jnp.float32)
    ret = _done_aware_return(costs, discounts, gamma=0.9)
    expected = 1.0 + 0.9 + 0.9**2
    assert float(ret[0]) == pytest.approx(expected)


def test_oracle_on_rank_dump_learns_synthetic_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("jax")
    from scripts.oracle_critic_fit import OracleFitConfig, _oracle_on_rank_dump

    num_states = 12
    num_candidates = 4
    states = np.zeros((num_states, 5), dtype=np.float32)
    goals = np.zeros((num_states, 2), dtype=np.float32)
    action_values = np.linspace(-1.0, 1.0, num_candidates, dtype=np.float32)
    actions = np.zeros((num_states, num_candidates, 2), dtype=np.float32)
    actions[:, :, 0] = action_values[None, :]
    labels = np.broadcast_to(action_values[None, :], (num_states, num_candidates))
    dump_path = tmp_path / "rank_dump.npz"
    np.savez(
        dump_path,
        states=states,
        candidate_actions=actions,
        goals=goals,
        labels_dense_h50=labels,
        labels_sparse_h50=labels,
        valid_mask=np.ones((num_states,), dtype=bool),
        cost_rank_horizon=np.asarray(50, dtype=np.int32),
        cost_rank_num_candidates=np.asarray(num_candidates, dtype=np.int32),
        cost_dense_prox_tau=np.asarray(0.5, dtype=np.float32),
        epoch_dumped=np.asarray(1, dtype=np.int32),
    )

    _oracle_on_rank_dump(
        OracleFitConfig(
            load_rank_dump=str(dump_path),
            output_csv=str(tmp_path / "oracle.csv"),
            oracle_results_npz=str(tmp_path / "oracle.npz"),
            rank_dump_oracle_seeds=1,
            fit_steps=500,
            fit_batch_size=16,
            fit_learning_rate=1e-2,
            fit_val_fraction=0.5,
            fit_bootstrap_samples=50,
            width=32,
            num_blocks=1,
            use_residual=False,
        )
    )

    out = capsys.readouterr().out
    match = re.search(r"ORACLE_ON_RANK_BUFFER spearman_mean=([-0-9.eE]+)", out)
    assert match is not None
    assert float(match.group(1)) >= 0.9
