#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_utils import PAPER_DATA, ROOT, mean, read_csv, se, set_paper_style, to_float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PAPER_DATA / "psi_safety_probe.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "fig6_psi_safety_probe.pdf",
    )
    args = parser.parse_args(argv)
    rows = read_csv(args.input)
    if not rows:
        raise SystemExit(f"optional probe data not found: {args.input}")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "")
        rho = to_float(row.get("spearman"))
        if method and rho is not None:
            grouped[method].append(rho)
    if not grouped:
        raise SystemExit(f"no method/spearman rows found in {args.input}")
    labels = sorted(grouped)
    vals = [mean(grouped[label]) for label in labels]
    errs = [se(grouped[label]) for label in labels]
    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.0, 2.2), constrained_layout=True)
    ax.bar(range(len(labels)), vals, yerr=errs, color=["#4c78a8", "#d55e00"][: len(labels)], capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Within-state Spearman")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
