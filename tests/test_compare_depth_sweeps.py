from __future__ import annotations

import csv
import io

from scripts.compare_depth_sweeps import compare_rows, main


def test_compare_rows_reports_depth_metric_deltas() -> None:
    baseline = [
        {
            "num_blocks": "8",
            "eval_ever_reached": "0.90",
            "eval_cost_return": "100",
            "hard_viol": "0.04",
        },
        {
            "num_blocks": "8",
            "eval_ever_reached": "1.00",
            "eval_cost_return": "120",
            "hard_viol": "0.06",
        },
    ]
    candidate = [
        {
            "num_blocks": "8",
            "eval_ever_reached": "0.98",
            "eval_cost_return": "90",
            "hard_viol": "0.07",
        },
        {
            "num_blocks": "8",
            "eval_ever_reached": "1.00",
            "eval_cost_return": "100",
            "hard_viol": "0.09",
        },
    ]

    rows = compare_rows(
        baseline,
        candidate,
        metrics=("eval_ever_reached", "eval_cost_return", "hard_viol"),
    )

    assert rows == [
        {
            "depth": "8",
            "metric": "eval_ever_reached",
            "baseline_n": "2",
            "candidate_n": "2",
            "baseline_mean": "0.95",
            "candidate_mean": "0.99",
            "delta": "0.04",
            "cmdp_vs_baseline": "better",
        },
        {
            "depth": "8",
            "metric": "eval_cost_return",
            "baseline_n": "2",
            "candidate_n": "2",
            "baseline_mean": "110",
            "candidate_mean": "95",
            "delta": "-15",
            "cmdp_vs_baseline": "better",
        },
        {
            "depth": "8",
            "metric": "hard_viol",
            "baseline_n": "2",
            "candidate_n": "2",
            "baseline_mean": "0.05",
            "candidate_mean": "0.08",
            "delta": "0.03",
            "cmdp_vs_baseline": "worse",
        },
    ]


def test_main_writes_csv(tmp_path, capsys) -> None:
    baseline_path = tmp_path / "pid.csv"
    candidate_path = tmp_path / "cmdp.csv"
    fieldnames = ["num_blocks", "eval_ever_reached", "eval_cost_return"]
    with baseline_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "num_blocks": "2",
                "eval_ever_reached": "0.5",
                "eval_cost_return": "100",
            }
        )
    with candidate_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "num_blocks": "2",
                "eval_ever_reached": "0.75",
                "eval_cost_return": "110",
            }
        )

    assert main(
        [
            str(baseline_path),
            str(candidate_path),
            "--metrics",
            "eval_ever_reached,eval_cost_return",
        ]
    ) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))

    assert rows[0]["delta"] == "0.25"
    assert rows[0]["cmdp_vs_baseline"] == "better"
    assert rows[1]["delta"] == "10"
    assert rows[1]["cmdp_vs_baseline"] == "worse"
