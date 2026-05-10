#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_utils import PAPER_DATA, ROOT, group_numeric, read_csv, set_paper_style

COLORS = {"sparse": "#6f6f6f", "dense": "#d55e00"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variance", type=Path, default=PAPER_DATA / "variance_decomposition.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "fig3_variance_decomposition.pdf",
    )
    args = parser.parse_args(argv)
    rows = read_csv(args.variance)
    if not rows:
        raise SystemExit(f"no rows found in {args.variance}")

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.3), constrained_layout=True)
    grouped = group_numeric(rows, keys=("mode", "horizon"), value="within_between_ratio")
    for mode in ("sparse", "dense"):
        points = []
        for (group_mode, horizon), (mu, err, _n) in grouped.items():
            if group_mode == mode:
                points.append((int(float(horizon)), mu, err))
        if not points:
            continue
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        es = [p[2] for p in points]
        ax.plot(xs, ys, marker="o", lw=1.8, label=mode.title(), color=COLORS[mode])
        ax.fill_between(
            xs,
            [max(y - e, 1e-12) for y, e in zip(ys, es)],
            [y + e for y, e in zip(ys, es)],
            color=COLORS[mode],
            alpha=0.18,
            linewidth=0,
        )
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.9)
    ax.text(1.05, 1.1, "equal within/between", fontsize=8, color="#666666")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 5, 20, 50])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Cost-return Horizon")
    ax.set_ylabel("Within/Between Variance Ratio")
    ax.grid(axis="both", color="#e8e8e8", linewidth=0.8)
    ax.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
