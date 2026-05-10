#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from paper_utils import (
    CELL_COLUMNS,
    ORACLE_COLUMNS,
    PAPER_DATA,
    ROOT,
    VARIANCE_COLUMNS,
    infer_depth,
    infer_seed,
    read_csv,
    to_float,
    write_csv,
)
from summarize_relxy_logs import parse_log


DEFAULT_LOG_GLOBS = (
    "safe_depth*.out",
    "safe_relxy_depth*.out",
    "safe_dense_d8_active*.out",
    "safe_depth_dense*.out",
    "runs/**/*.out",
    "logs/**/*.out",
)
DEFAULT_CSV_GLOBS = (
    "figures/data/**/*.csv",
    "*.csv",
    "runs/**/*.csv",
    "logs/**/*.csv",
)
CORE_DEPTHS = {"4", "8", "16", "32"}
CORE_SEEDS = {"0", "1", "2"}
PREFERRED_CELL_SOURCES = (
    "dense_depth_sweep_1024412.csv",
    "safe_depth_dense.1024412",
    "dense_depth8_active_pid_sanity_1024474.csv",
    "safe_dense_d8_active.1024474",
)


def _warn(message: str) -> None:
    print(f"[build_paper_dataset] warning: {message}", file=sys.stderr)


def _glob_many(patterns: list[str] | tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(ROOT.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _mode_from_row(row: dict[str, str]) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("mode", "cost_mode", "label", "file")
    ).lower()
    if "dense" in text or "dense_proximity" in text:
        return "dense"
    if "sparse" in text or "cmdp" in text or "pid_off" in text:
        return "sparse"
    return ""


def _cell_from_summary_row(row: dict[str, str]) -> dict[str, str] | None:
    mode = _mode_from_row(row)
    depth = row.get("depth") or row.get("num_blocks") or infer_depth(
        row.get("label", ""), row.get("file", "")
    )
    seed = row.get("seed") or infer_seed(row.get("label", ""), row.get("file", ""))
    if not mode or not depth or seed == "":
        return None
    pid_source = row.get("pid_source") or row.get("pid_cost_source") or "active"
    out = {
        "mode": mode,
        "pid_source": pid_source,
        "depth": depth,
        "seed": seed,
        "hard": row.get("hard") or row.get("hard_viol", ""),
        "reach": row.get("reach") or row.get("eval_ever_reached", ""),
        "cost_ret": row.get("cost_ret") or row.get("eval_cost_return", ""),
        "actor_qc_percentile": row.get("actor_qc_percentile", ""),
        "actor_qc_percentile_cf": row.get("actor_qc_percentile_cf", ""),
        "actor_true_cost_percentile": row.get("actor_true_cost_percentile", ""),
        "q_c_action_spread": row.get("q_c_action_spread", ""),
        "true_action_dense_cost_spread": row.get(
            "true_action_dense_cost_spread", ""
        ),
        "true_action_sparse_cost_spread": row.get(
            "true_action_sparse_cost_spread", ""
        ),
        "corr_qc_true_cost": row.get("corr_qc_true_cost", ""),
        "best_qc_matches_best_true_cost_frac": row.get(
            "best_qc_matches_best_true_cost_frac", ""
        ),
        "frac_states_with_nonzero_dense_cost_spread": row.get(
            "frac_states_with_nonzero_dense_cost_spread",
            row.get("frac_states_with_nonzero_true_cost_spread", ""),
        ),
        "_source": row.get("_source", row.get("file", "")),
    }
    return out


def _cell_score(cell: dict[str, str]) -> tuple[int, int, int, int]:
    source = cell.get("_source", "")
    preferred = int(any(token in source for token in PREFERRED_CELL_SOURCES))
    has_counterfactual = int(bool(cell.get("corr_qc_true_cost")))
    has_reach = int(bool(cell.get("reach")))
    is_active_pid_sanity = int(
        cell.get("mode") == "dense"
        and cell.get("pid_source") == "active"
        and cell.get("depth") == "8"
    )
    return (preferred, has_counterfactual, has_reach, is_active_pid_sanity)


def _normalize_int_text(value: str) -> str:
    numeric = to_float(value)
    if numeric is None:
        return value
    return str(int(numeric))


def build_cells(
    logs: list[Path],
    csv_paths: list[Path],
    *,
    include_extra_cells: bool = False,
) -> list[dict[str, str]]:
    cells_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}

    def add(row: dict[str, str], *, source: str = "") -> None:
        if source and "_source" not in row:
            row = {**row, "_source": source}
        cell = _cell_from_summary_row(row)
        if cell is None:
            return
        cell["depth"] = _normalize_int_text(cell["depth"])
        cell["seed"] = _normalize_int_text(cell["seed"])
        if not include_extra_cells and (
            cell["depth"] not in CORE_DEPTHS or cell["seed"] not in CORE_SEEDS
        ):
            return
        key = (
            cell["mode"],
            cell["pid_source"],
            cell["depth"],
            cell["seed"],
        )
        existing = cells_by_key.get(key)
        if existing is None or _cell_score(cell) > _cell_score(existing):
            cells_by_key[key] = cell

    for path in logs:
        try:
            add(parse_log(path), source=path.name)
        except Exception as exc:  # pragma: no cover - defensive for user logs
            _warn(f"could not parse log {path}: {exc}")

    for path in csv_paths:
        if "figures/data/paper" in path.as_posix():
            continue
        for row in read_csv(path):
            add(row, source=f"{path.name}:{row.get('file', '')}")
    rows = list(cells_by_key.values())
    for row in rows:
        row.pop("_source", None)
    rows.sort(key=lambda r: (r["mode"], int(float(r["depth"])), int(float(r["seed"])), r["pid_source"]))
    return rows


def _first(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name, "")
        if str(value).strip():
            return str(value)
    return ""


def _label_mode(row: dict[str, str]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ("label", "label_type", "kind", "path", "file")).lower()
    if "sparse" in text:
        return "sparse"
    if "dense" in text:
        return "dense"
    return ""


def build_variance(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        if "figures/data/paper" in path.as_posix():
            continue
        for row in read_csv(path):
            if row.get("kind") and row.get("kind") != "variance":
                continue
            horizon = _first(row, ("horizon", "H", "cost_rank_horizon"))
            within = _first(
                row,
                (
                    "within_state_var",
                    "within_var",
                    "fit_within_var",
                    "label_within_var",
                    "within_state_action_var",
                ),
            )
            between = _first(
                row,
                (
                    "between_state_var",
                    "between_var",
                    "fit_between_var",
                    "label_between_var",
                    "between_state_var_mean",
                ),
            )
            ratio = _first(
                row,
                (
                    "within_between_ratio",
                    "within_between",
                    "rank_label_within_between",
                    "fit_within_between",
                ),
            )
            if not horizon or not (ratio or (within and between)):
                continue
            if not ratio and to_float(within) is not None and to_float(between):
                ratio = str(float(within) / float(between))
            mode = _label_mode(row)
            if not mode:
                mode = "dense"
            rows.append(
                {
                    "mode": mode,
                    "horizon": horizon,
                    "seed": _first(row, ("seed", "fit_seed")),
                    "within_state_var": within,
                    "between_state_var": between,
                    "within_between_ratio": ratio,
                    "n_states": _first(row, ("n_states", "num_states")),
                }
            )
    rows.sort(key=lambda r: (r["mode"], int(float(r["horizon"])), r["seed"]))
    return rows


def _oracle_label_type(row: dict[str, str]) -> str:
    label = _first(row, ("label", "label_type")).lower()
    horizon = _first(row, ("horizon", "H"))
    if label == "synthetic_action_norm":
        return "synthetic_action_norm"
    if label == "sparse":
        return f"sparse_h{horizon or '50'}"
    if label == "dense":
        return f"dense_h{horizon or '50'}"
    if "synthetic" in label:
        return "synthetic_action_norm"
    if "sparse" in label:
        return f"sparse_h{horizon or '50'}"
    if "dense" in label:
        return f"dense_h{horizon or '50'}"
    return ""


def build_oracle_fits(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        if "figures/data/paper" in path.as_posix():
            continue
        for row in read_csv(path):
            spearman = _first(row, ("fit_within_spearman", "spearman", "rho"))
            if not spearman:
                continue
            kind = row.get("kind", "")
            if kind and kind not in {"fit", "oracle_on_rank_buffer"}:
                continue
            label_type = _oracle_label_type(row)
            if not label_type:
                continue
            rows.append(
                {
                    "label_type": label_type,
                    "fit_seed": _first(row, ("fit_seed", "seed")),
                    "spearman": spearman,
                    "top1": _first(row, ("fit_top1_match", "top1")),
                    "n_states": _first(row, ("n_states", "num_states")),
                    "bootstrap_ci_low": _first(
                        row, ("fit_within_spearman_ci_low", "spearman_ci_low")
                    ),
                    "bootstrap_ci_high": _first(
                        row, ("fit_within_spearman_ci_high", "spearman_ci_high")
                    ),
                }
            )
    rows.sort(key=lambda r: (r["label_type"], r["fit_seed"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate existing SR-CPO logs and CSVs into paper datasets."
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_DATA)
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--csv", action="append", type=Path, default=[])
    parser.add_argument("--log-glob", action="append", default=[])
    parser.add_argument("--csv-glob", action="append", default=[])
    parser.add_argument(
        "--include-extra-cells",
        action="store_true",
        help="Keep depths/seeds outside the paper matrix instead of filtering them.",
    )
    args = parser.parse_args(argv)

    logs = [path for path in args.log if path.is_file()]
    logs.extend(_glob_many(args.log_glob or (() if args.log else DEFAULT_LOG_GLOBS)))
    csv_paths = [path for path in args.csv if path.is_file()]
    csv_paths.extend(_glob_many(args.csv_glob or (() if args.csv else DEFAULT_CSV_GLOBS)))
    if not logs:
        _warn("no SLURM logs found")
    if not csv_paths:
        _warn("no source CSV files found")

    cells = build_cells(
        logs, csv_paths, include_extra_cells=args.include_extra_cells
    )
    variance = build_variance(csv_paths)
    oracle = build_oracle_fits(csv_paths)

    write_csv(args.output_dir / "cells.csv", cells, CELL_COLUMNS)
    write_csv(args.output_dir / "variance_decomposition.csv", variance, VARIANCE_COLUMNS)
    write_csv(args.output_dir / "oracle_fits.csv", oracle, ORACLE_COLUMNS)

    print(f"wrote {len(cells)} rows to {args.output_dir / 'cells.csv'}")
    print(
        f"wrote {len(variance)} rows to "
        f"{args.output_dir / 'variance_decomposition.csv'}"
    )
    print(f"wrote {len(oracle)} rows to {args.output_dir / 'oracle_fits.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
