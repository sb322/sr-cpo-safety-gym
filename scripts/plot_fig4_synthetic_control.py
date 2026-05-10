#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_utils import PAPER_DATA, ROOT, mean, read_csv, se, set_paper_style, to_float

LABELS = (
    ("synthetic_action_norm", "Synthetic\naction label"),
    ("dense_h20", "Dense\nH=20"),
    ("dense_h50", "Dense\nH=50"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=PAPER_DATA / "oracle_fits.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "fig4_synthetic_control.pdf",
    )
    args = parser.parse_args(argv)
    rows = read_csv(args.oracle)
    if not rows:
        raise SystemExit(f"no rows found in {args.oracle}")

    grouped: dict[str, list[float]] = defaultdict(list)
    ci_by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        rho = to_float(row.get("spearman"))
        if rho is None:
            continue
        label = row.get("label_type", "")
        grouped[label].append(rho)
        low = to_float(row.get("bootstrap_ci_low"))
        high = to_float(row.get("bootstrap_ci_high"))
        if low is not None and high is not None:
            ci_by_label[label].append((low, high))

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.0, 2.3), constrained_layout=True)
    xs = list(range(len(LABELS)))
    vals = [mean(grouped[label]) if grouped[label] else 0.0 for label, _ in LABELS]
    errs = []
    for label, _ in LABELS:
        if ci_by_label[label]:
            lows = [low for low, _ in ci_by_label[label]]
            highs = [high for _, high in ci_by_label[label]]
            errs.append([[max(vals[len(errs)] - mean(lows), 0.0)], [max(mean(highs) - vals[len(errs)], 0.0)]])
        else:
            errs.append([[se(grouped[label])], [se(grouped[label])]])
    lower = [item[0][0] for item in errs]
    upper = [item[1][0] for item in errs]
    ax.bar(xs, vals, yerr=[lower, upper], color=["#4c78a8", "#d55e00", "#d55e00"], capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([name for _, name in LABELS])
    ax.set_ylabel("Held-out Spearman")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    if vals and vals[0] > 0:
        ax.text(0, min(vals[0] + 0.08, 0.98), "same architecture", ha="center", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
