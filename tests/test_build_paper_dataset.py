from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_sample_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "DEPTH_LABEL=dense_depth8",
                "SEED=1",
                "EPOCHS=200",
                "NUM_BLOCKS=8",
                "PID_COST_SOURCE=sparse",
                "COST_MODE=dense_proximity",
                "[200/200] hard_viol=0.0500 cost=0.0500 reached=0.0100",
                "         action_rank[ actor_qc_percentile=0.100 q_c_action_spread=2.00e+00]",
                "         counterfactual[ true_action_dense_cost_spread=4.00e-04 true_action_sparse_cost_spread=5.00e-04 corr_qc_true_cost=0.010 actor_true_cost_percentile=0.500 actor_qc_percentile_cf=0.120 best_qc_matches_best_true_cost_frac=0.250 frac_states_with_nonzero_true_cost_spread=1.000]",
                "         eval_ever_reached=0.0300 eval_cost_return=12.0000",
            ]
        )
    )


def test_build_paper_dataset_writes_canonical_csvs(tmp_path: Path) -> None:
    log_path = tmp_path / "safe_depth_dense.1_0.out"
    _write_sample_log(log_path)
    oracle_path = tmp_path / "oracle.csv"
    oracle_path.write_text(
        "kind,label,horizon,seed,fit_within_spearman,fit_top1_match,n_states\n"
        "fit,synthetic_action_norm,0,0,0.99,0.9,32\n"
    )
    variance_path = tmp_path / "variance.csv"
    variance_path.write_text(
        "kind,label,horizon,seed,within_between_ratio,n_states\n"
        "variance,dense,20,0,1e-3,64\n"
    )
    output_dir = tmp_path / "paper"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_paper_dataset.py"),
            "--output-dir",
            str(output_dir),
            "--log",
            str(log_path),
            "--csv",
            str(oracle_path),
            "--csv",
            str(variance_path),
        ],
        cwd=ROOT,
        check=True,
    )

    cells = list(csv.DictReader((output_dir / "cells.csv").open()))
    assert cells
    assert cells[0]["mode"] == "dense"
    assert cells[0]["depth"] == "8"
    assert cells[0]["true_action_dense_cost_spread"] == "4.00e-04"
    variance = list(csv.DictReader((output_dir / "variance_decomposition.csv").open()))
    assert variance and variance[0]["within_between_ratio"] == "1e-3"
    oracle = list(csv.DictReader((output_dir / "oracle_fits.csv").open()))
    assert oracle and oracle[0]["label_type"] == "synthetic_action_norm"
