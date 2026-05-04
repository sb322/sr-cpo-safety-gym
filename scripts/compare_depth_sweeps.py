#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics as stats
import sys
from pathlib import Path

DEFAULT_METRICS = (
    "eval_ever_reached",
    "eval_first_hit_time",
    "eval_success_count",
    "eval_cost_return",
    "hard_viol",
    "hazard",
    "vase_contact",
    "cost_resid",
    "lambda_qc_a",
)
LOWER_IS_BETTER = {
    "eval_first_hit_time",
    "eval_cost_return",
    "hard_viol",
    "hazard",
    "vase_contact",
    "cost_resid",
}
HIGHER_IS_BETTER = {"eval_ever_reached", "eval_success_count"}
FIELDNAMES = (
    "depth",
    "metric",
    "baseline_n",
    "candidate_n",
    "baseline_mean",
    "candidate_mean",
    "delta",
    "cmdp_vs_baseline",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_for_metric(rows: list[dict[str, str]], metric: str) -> float | None:
    values = [_float_or_none(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return stats.mean(values)


def _group_by_depth(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        depth = row.get("num_blocks", "")
        if depth:
            grouped.setdefault(depth, []).append(row)
    return grouped


def _judgement(metric: str, delta: float | None, tolerance: float) -> str:
    if delta is None or abs(delta) <= tolerance:
        return "same"
    if metric in LOWER_IS_BETTER:
        return "better" if delta < 0 else "worse"
    if metric in HIGHER_IS_BETTER:
        return "better" if delta > 0 else "worse"
    return "n/a"


def compare_rows(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    tolerance: float = 1.0e-4,
) -> list[dict[str, str]]:
    baseline_by_depth = _group_by_depth(baseline_rows)
    candidate_by_depth = _group_by_depth(candidate_rows)
    depths = sorted(
        baseline_by_depth.keys() & candidate_by_depth.keys(),
        key=lambda value: int(value),
    )
    output: list[dict[str, str]] = []
    for depth in depths:
        baseline_depth_rows = baseline_by_depth[depth]
        candidate_depth_rows = candidate_by_depth[depth]
        for metric in metrics:
            baseline_mean = _mean_for_metric(baseline_depth_rows, metric)
            candidate_mean = _mean_for_metric(candidate_depth_rows, metric)
            delta = (
                None
                if baseline_mean is None or candidate_mean is None
                else candidate_mean - baseline_mean
            )
            output.append(
                {
                    "depth": depth,
                    "metric": metric,
                    "baseline_n": str(len(baseline_depth_rows)),
                    "candidate_n": str(len(candidate_depth_rows)),
                    "baseline_mean": ""
                    if baseline_mean is None
                    else f"{baseline_mean:.6g}",
                    "candidate_mean": ""
                    if candidate_mean is None
                    else f"{candidate_mean:.6g}",
                    "delta": "" if delta is None else f"{delta:.6g}",
                    "cmdp_vs_baseline": _judgement(metric, delta, tolerance),
                }
            )
    return output


def _write_csv(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def _write_markdown(rows: list[dict[str, str]]) -> None:
    print("| depth | metric | pid-off | CMDP | delta | read |")
    print("|---:|---|---:|---:|---:|---|")
    for row in rows:
        print(
            "| "
            f"{row['depth']} | {row['metric']} | {row['baseline_mean']} | "
            f"{row['candidate_mean']} | {row['delta']} | "
            f"{row['cmdp_vs_baseline']} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare pid-off and CMDP relative-XY depth sweep summary CSVs."
    )
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names to compare.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "markdown"),
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-4,
        help="Absolute delta tolerance treated as same.",
    )
    args = parser.parse_args(argv)

    metrics = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())
    rows = compare_rows(
        _read_rows(args.baseline_csv),
        _read_rows(args.candidate_csv),
        metrics=metrics,
        tolerance=args.tolerance,
    )
    if args.format == "markdown":
        _write_markdown(rows)
    else:
        _write_csv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
