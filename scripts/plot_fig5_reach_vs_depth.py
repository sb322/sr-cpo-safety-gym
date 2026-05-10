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
    parser.add_argument("--cells", type=Path, default=PAPER_DATA / "cells.csv")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "figures" / "fig5_reach_vs_depth.pdf"
    )
    args = parser.parse_args(argv)
    rows = read_csv(args.cells)
    if not rows:
        raise SystemExit(f"no rows found in {args.cells}")

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
    grouped = group_numeric(rows, keys=("mode", "depth"), value="reach")
    for mode in ("sparse", "dense"):
        points = []
        for (group_mode, depth), (mu, err, _n) in grouped.items():
            if group_mode == mode:
                points.append((int(float(depth)), mu, err))
        if not points:
            continue
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        es = [p[2] for p in points]
        ax.plot(xs, ys, marker="o", lw=1.8, label=mode.title(), color=COLORS[mode])
        ax.fill_between(
            xs,
            [y - e for y, e in zip(ys, es)],
            [y + e for y, e in zip(ys, es)],
            color=COLORS[mode],
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xlabel("Network Depth (residual blocks)")
    ax.set_ylabel("Eval Ever Reached")
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
