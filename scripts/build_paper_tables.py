#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from paper_utils import (
    PAPER_DATA,
    PAPER_TABLES,
    format_float,
    latex_escape,
    mean,
    read_csv,
    se,
    to_float,
    write_csv,
)

TABLE1_COLUMNS = (
    "mode",
    "pid_source",
    "depth",
    "seed",
    "hard",
    "reach",
    "cost_ret",
    "actor_qc_percentile_cf",
    "actor_true_cost_percentile",
    "q_c_action_spread",
    "true_action_dense_cost_spread",
    "corr_qc_true_cost",
)
TABLE2_METRICS = (
    "hard",
    "reach",
    "cost_ret",
    "actor_qc_percentile_cf",
    "actor_true_cost_percentile",
    "q_c_action_spread",
    "true_action_dense_cost_spread",
    "corr_qc_true_cost",
)


def _mean_se_text(values: list[float]) -> str:
    if not values:
        return ""
    return f"{format_float(mean(values))} +/- {format_float(se(values))}"


def _write_tex(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(r"\toprule" + "\n")
        handle.write(" & ".join(latex_escape(h) for h in headers) + r" \\" + "\n")
        handle.write(r"\midrule" + "\n")
        for row in rows:
            handle.write(" & ".join(latex_escape(x) for x in row) + r" \\" + "\n")
        handle.write(r"\bottomrule" + "\n")


def build_table1(cells: list[dict[str, str]], output_dir: Path) -> None:
    rows = [{key: row.get(key, "") for key in TABLE1_COLUMNS} for row in cells]
    write_csv(output_dir / "table1_full_matrix.csv", rows, TABLE1_COLUMNS)
    tex_rows = [
        [row.get(key, "") if key in {"mode", "pid_source", "depth", "seed"} else format_float(row.get(key))]
        for row in rows
        for key in [None]
    ]
    _write_tex(output_dir / "table1_full_matrix.tex", list(TABLE1_COLUMNS), tex_rows)


def build_table2(cells: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in cells:
        key = (row.get("mode", ""), row.get("pid_source", ""), row.get("depth", ""))
        for metric in TABLE2_METRICS:
            numeric = to_float(row.get(metric))
            if numeric is not None:
                grouped[key][metric].append(numeric)
    rows = []
    for key in sorted(grouped, key=lambda x: (x[0], x[1], int(float(x[2] or 0)))):
        mode, pid_source, depth = key
        row = {"mode": mode, "pid_source": pid_source, "depth": depth}
        for metric in TABLE2_METRICS:
            row[metric] = _mean_se_text(grouped[key][metric])
        rows.append(row)
    columns = ("mode", "pid_source", "depth", *TABLE2_METRICS)
    write_csv(output_dir / "table2_aggregate.csv", rows, columns)
    _write_tex(
        output_dir / "table2_aggregate.tex",
        list(columns),
        [[row.get(key, "") for key in columns] for row in rows],
    )


def build_table3(
    variance: list[dict[str, str]], oracle: list[dict[str, str]], output_dir: Path
) -> None:
    rows: list[dict[str, str]] = []
    grouped_var: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in variance:
        ratio = to_float(row.get("within_between_ratio"))
        if ratio is not None:
            grouped_var[(row.get("mode", ""), row.get("horizon", ""))].append(ratio)
    for (mode, horizon), values in sorted(
        grouped_var.items(), key=lambda x: (x[0][0], int(float(x[0][1] or 0)))
    ):
        rows.append(
            {
                "section": "variance",
                "label": f"{mode} H={horizon}",
                "mean": format_float(mean(values)),
                "se": format_float(se(values)),
                "n": str(len(values)),
            }
        )
    grouped_oracle: dict[str, list[float]] = defaultdict(list)
    for row in oracle:
        rho = to_float(row.get("spearman"))
        if rho is not None:
            grouped_oracle[row.get("label_type", "")].append(rho)
    for label, values in sorted(grouped_oracle.items()):
        rows.append(
            {
                "section": "oracle",
                "label": label,
                "mean": format_float(mean(values)),
                "se": format_float(se(values)),
                "n": str(len(values)),
            }
        )
    columns = ("section", "label", "mean", "se", "n")
    write_csv(output_dir / "table3_variance_and_oracle.csv", rows, columns)
    _write_tex(
        output_dir / "table3_variance_and_oracle.tex",
        list(columns),
        [[row.get(key, "") for key in columns] for row in rows],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PAPER_DATA)
    parser.add_argument("--output-dir", type=Path, default=PAPER_TABLES)
    args = parser.parse_args(argv)
    cells = read_csv(args.data_dir / "cells.csv")
    variance = read_csv(args.data_dir / "variance_decomposition.csv")
    oracle = read_csv(args.data_dir / "oracle_fits.csv")
    if not cells:
        raise SystemExit(f"no cells found in {args.data_dir / 'cells.csv'}")
    build_table1(cells, args.output_dir)
    build_table2(cells, args.output_dir)
    build_table3(variance, oracle, args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
