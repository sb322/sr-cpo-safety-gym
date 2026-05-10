#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAPER_DATA = ROOT / "figures" / "data" / "paper"
PAPER_TABLES = ROOT / "figures" / "tables" / "paper"

CELL_COLUMNS = (
    "mode",
    "pid_source",
    "depth",
    "seed",
    "hard",
    "reach",
    "cost_ret",
    "actor_qc_percentile",
    "actor_qc_percentile_cf",
    "actor_true_cost_percentile",
    "q_c_action_spread",
    "true_action_dense_cost_spread",
    "true_action_sparse_cost_spread",
    "corr_qc_true_cost",
    "best_qc_matches_best_true_cost_frac",
    "frac_states_with_nonzero_dense_cost_spread",
)

VARIANCE_COLUMNS = (
    "mode",
    "horizon",
    "seed",
    "within_state_var",
    "between_state_var",
    "within_between_ratio",
    "n_states",
)

ORACLE_COLUMNS = (
    "label_type",
    "fit_seed",
    "spearman",
    "top1",
    "n_states",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
)


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def se(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var) / math.sqrt(len(vals))


def group_numeric(
    rows: Iterable[Mapping[str, str]],
    *,
    keys: Sequence[str],
    value: str,
) -> dict[tuple[str, ...], tuple[float, float, int]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        numeric = to_float(row.get(value))
        if numeric is None:
            continue
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(numeric)
    return {
        key: (mean(vals), se(vals), len(vals))
        for key, vals in grouped.items()
        if vals
    }


def sorted_depths(rows: Iterable[Mapping[str, str]]) -> list[int]:
    depths = {int(float(row["depth"])) for row in rows if to_float(row.get("depth"))}
    return sorted(depths)


def infer_depth(*values: str) -> str:
    for value in values:
        match = re.search(r"(?:depth|_d)(\d+)", value or "")
        if match:
            return match.group(1)
    return ""


def infer_seed(*values: str) -> str:
    for value in values:
        match = re.search(r"seed(\d+)", value or "")
        if match:
            return match.group(1)
    return ""


def format_float(value: Any, digits: int = 3) -> str:
    numeric = to_float(value)
    if numeric is None:
        return ""
    if numeric != 0 and (abs(numeric) < 0.001 or abs(numeric) >= 1000):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
