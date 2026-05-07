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
