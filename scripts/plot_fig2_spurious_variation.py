#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_utils import PAPER_DATA, ROOT, read_csv, set_paper_style, to_float

COLORS = {"sparse": "#6f6f6f", "dense": "#d55e00"}
MARKERS = {4: "o", 8: "s", 16: "^", 32: "D"}


def _scatter_by_cell(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
) -> None:
    for row in rows:
        x = to_float(row.get(x_key))
        y = to_float(row.get(y_key))
        depth = to_float(row.get("depth"))
        mode = row.get("mode", "")
        if x is None or y is None or depth is None or mode not in COLORS:
            continue
        ax.scatter(
            x,
            y,
            s=28,
            marker=MARKERS.get(int(depth), "o"),
            facecolor=COLORS[mode],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, default=PAPER_DATA / "cells.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "fig2_spurious_variation.pdf",
    )
    args = parser.parse_args(argv)
    rows = read_csv(args.cells)
    if not rows:
        raise SystemExit(f"no rows found in {args.cells}")

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
    _scatter_by_cell(
        axes[0],
        rows,
        x_key="actor_true_cost_percentile",
        y_key="actor_qc_percentile_cf",
    )
    axes[0].axhline(0.5, color="#999999", linestyle="--", linewidth=0.9)
    axes[0].axvline(0.5, color="#999999", linestyle="--", linewidth=0.9)
    axes[0].fill_between([0.0, 0.5], 0.0, 0.5, color="#2ca02c", alpha=0.08)
    axes[0].text(0.04, 0.08, "actually safer", color="#2c7a2c", fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Truth view: actor cost percentile")
    axes[0].set_ylabel(r"$Q_c$ view: actor percentile")

    derived_rows = []
    for row in rows:
        corr = to_float(row.get("corr_qc_true_cost"))
        if corr is None:
            continue
        derived = dict(row)
        derived["abs_corr_qc_true_cost"] = str(abs(corr))
        derived_rows.append(derived)
    _scatter_by_cell(
        axes[1],
        derived_rows,
        x_key="abs_corr_qc_true_cost",
        y_key="q_c_action_spread",
    )
    axes[1].set_xlabel(r"$|corr(Q_c,\ true\ cost)|$")
    axes[1].set_ylabel(r"$Q_c$ Action Spread")
    axes[1].text(
        0.02,
        0.95,
        r"$Q_c$ varies, but not with truth",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    for ax in axes:
        ax.grid(axis="both", color="#e8e8e8", linewidth=0.8)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m], label=m.title())
        for m in ("sparse", "dense")
    ]
    axes[1].legend(handles=handles, frameon=False, loc="best")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
