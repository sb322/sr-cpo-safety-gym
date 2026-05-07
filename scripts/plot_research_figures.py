#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sr-cpo-mplconfig")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEPTH_COLORS = {
    "2": "#4E79A7",
    "4": "#59A14F",
    "8": "#F28E2B",
    "16": "#E15759",
}
CONDITION_COLORS = {
    "pid_off": "#6B7280",
    "cmdp_active": "#D55E00",
    "dense_proximity": "#0072B2",
    "sparse": "#6B7280",
}
RATIO_COLORS = {
    "0": "#4E79A7",
    "0.25": "#F28E2B",
    "0.5": "#E15759",
}
METRIC_LABELS = {
    "eval_ever_reached": "Eval ever reached",
    "reached": "Train reached",
    "hard_viol": "Hard violation rate",
    "hazard": "Hazard contact rate",
    "lambda_qc_a": r"$\lambda Q_c(a_\pi)$",
    "grad_ratio_cost_reward": "Cost/reward grad. ratio",
    "grad_ratio_cost_reward_risky": "Risky cost/reward grad. ratio",
    "qc": "Mean Qc",
    "qc_actor_risky": "Qc on risky states",
    "actor_qc_percentile": "Actor Qc percentile",
    "best_qc_action_is_actor_frac": "Actor is best-Qc action",
    "cost_risky_batch_frac": "Risky batch fraction",
    "true_action_cost_spread": "Active cost spread",
    "true_action_active_cost_spread": "Active cost spread",
    "true_action_sparse_cost_spread": "Sparse cost spread",
    "true_action_dense_cost_spread": "Dense cost spread",
    "true_action_hard_viol_spread": "Hard-viol. spread",
    "true_action_hazard_spread": "Hazard spread",
    "q_c_action_spread": "Qc action spread",
    "corr_qc_true_cost": "corr(Qc, true cost)",
    "corr_qc_true_hazard": "corr(Qc, hazard)",
    "actor_true_cost_percentile": "Actor true-cost percentile",
    "actor_qc_percentile_cf": "Actor Qc percentile",
    "eval_success_count": "Eval success count",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.75,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path or not path.is_file():
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "nan", "None"):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def has_metric(rows: list[dict[str, str]], metric: str) -> bool:
    return any(numeric(row, metric) is not None for row in rows)


def row_depth(row: dict[str, str]) -> str:
    value = row.get("num_blocks", "")
    parsed = numeric(row, "num_blocks")
    if parsed is not None:
        return str(int(parsed))
    return value


def row_seed(row: dict[str, str]) -> str:
    return row.get("seed") or row.get("file", "")


def row_condition(row: dict[str, str]) -> str:
    if row.get("cost_mode") == "dense_proximity":
        return "dense_proximity"
    return row.get("condition") or "cmdp_active"


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        out.append(float(np.mean(values[start : index + 1])))
    return out


Filter = Callable[[dict[str, str]], bool]


def zip_equal(*iterables):  # type: ignore[no-untyped-def]
    return zip(*iterables)  # noqa: B905


def aggregate_curves(
    rows: list[dict[str, str]],
    *,
    group_fn: Callable[[dict[str, str]], str],
    metric: str,
    filters: Iterable[Filter] = (),
    smooth_window: int = 1,
    interval: str = "sem",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    nested: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for row in rows:
        if any(not predicate(row) for predicate in filters):
            continue
        epoch = numeric(row, "epoch")
        value = numeric(row, metric)
        if epoch is None or value is None:
            continue
        group = group_fn(row)
        seed = row_seed(row)
        nested.setdefault(group, {}).setdefault(seed, []).append((int(epoch), value))

    aggregated: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group, seed_map in nested.items():
        by_epoch: dict[int, list[float]] = {}
        for points in seed_map.values():
            points = sorted(points)
            epochs = [point[0] for point in points]
            values = rolling_mean([point[1] for point in points], smooth_window)
            for epoch, value in zip_equal(epochs, values):
                by_epoch.setdefault(epoch, []).append(value)
        xs = np.asarray(sorted(by_epoch), dtype=float)
        means: list[float] = []
        bands: list[float] = []
        for epoch in xs:
            values = np.asarray(by_epoch[int(epoch)], dtype=float)
            means.append(float(np.mean(values)))
            if len(values) < 2:
                bands.append(0.0)
            else:
                std = float(np.std(values, ddof=1))
                bands.append(std if interval == "std" else std / math.sqrt(len(values)))
        aggregated[group] = (xs, np.asarray(means), np.asarray(bands))
    return aggregated


def mean_band(values: list[float], interval: str) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    std = float(np.std(arr, ddof=1))
    band = std if interval == "std" else std / math.sqrt(arr.size)
    return float(np.mean(arr)), band


def summary_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    return [value for row in rows if (value := numeric(row, metric)) is not None]


def save_figure(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def plot_curves(
    ax: plt.Axes,
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    colors: dict[str, str],
    labels: dict[str, str] | None = None,
) -> None:
    def sort_key(item: str) -> tuple[int, float | str]:
        try:
            return (0, float(item))
        except ValueError:
            return (1, item)

    for group in sorted(curves, key=sort_key):
        xs, ys, band = curves[group]
        color = colors.get(group, "#111827")
        label = labels.get(group, group) if labels else group
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label)
        ax.fill_between(xs, ys - band, ys + band, color=color, alpha=0.18, linewidth=0)


def plot_figure1(
    rows: list[dict[str, str]], out_dir: Path, args: argparse.Namespace
) -> bool:
    reach_metric = (
        "eval_ever_reached" if has_metric(rows, "eval_ever_reached") else "reached"
    )
    if not has_metric(rows, reach_metric) or not has_metric(rows, "hard_viol"):
        return False
    depths = {depth.strip() for depth in args.depths.split(",") if depth.strip()}
    condition = args.fig1_condition
    filters: list[Filter] = [lambda row: row_depth(row) in depths]
    if condition != "all":
        filters.append(lambda row: row_condition(row) == condition)

    reach = aggregate_curves(
        rows,
        group_fn=row_depth,
        metric=reach_metric,
        filters=filters,
        smooth_window=args.smooth_window,
        interval=args.interval,
    )
    safety = aggregate_curves(
        rows,
        group_fn=row_depth,
        metric="hard_viol",
        filters=filters,
        smooth_window=args.smooth_window,
        interval=args.interval,
    )
    if not reach or not safety:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.65), sharex=True)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    labels = {depth: f"depth {depth}" for depth in depths}
    plot_curves(axes[0], reach, colors=DEPTH_COLORS, labels=labels)
    plot_curves(axes[1], safety, colors=DEPTH_COLORS, labels=labels)
    axes[0].set_title("Depth improves reaching")
    axes[1].set_title("Safety does not improve")
    axes[0].set_ylabel(METRIC_LABELS[reach_metric])
    axes[1].set_ylabel(METRIC_LABELS["hard_viol"])
    for ax in axes:
        ax.set_xlabel("Epoch")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=max(1, len(legend_labels)),
        frameon=False,
        bbox_to_anchor=(0.52, 1.08),
    )
    save_figure(fig, out_dir, "fig01_depth_reach_vs_safety", args.dpi)
    return True


def plot_figure2(
    rows: list[dict[str, str]], out_dir: Path, args: argparse.Namespace
) -> bool:
    metrics = (
        "lambda_qc_a",
        "grad_ratio_cost_reward",
        "grad_ratio_cost_reward_risky",
    )
    if not all(has_metric(rows, metric) for metric in metrics):
        return False
    target_depth = args.depth_for_mechanism
    filters: list[Filter] = [
        lambda row: row_depth(row) in {"", target_depth},
        lambda row: row_condition(row) in {"pid_off", "cmdp_active"},
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.55), sharex=True)
    for ax, metric, label in zip_equal(axes, metrics, "ABC"):
        panel_label(ax, label)
        curves = aggregate_curves(
            rows,
            group_fn=row_condition,
            metric=metric,
            filters=filters,
            smooth_window=args.smooth_window,
            interval=args.interval,
        )
        plot_curves(
            ax,
            curves,
            colors=CONDITION_COLORS,
            labels={"pid_off": "PID off", "cmdp_active": "CMDP active"},
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.08),
    )
    save_figure(fig, out_dir, "fig02_cmdp_pressure_active", args.dpi)
    return True


def plot_single_metric(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    *,
    filters: Iterable[Filter],
    color: str,
    label: str,
    args: argparse.Namespace,
) -> bool:
    curves = aggregate_curves(
        rows,
        group_fn=lambda _row: label,
        metric=metric,
        filters=filters,
        smooth_window=args.smooth_window,
        interval=args.interval,
    )
    if not curves:
        return False
    plot_curves(ax, curves, colors={label: color})
    return True


def plot_figure3(
    rows: list[dict[str, str]], out_dir: Path, args: argparse.Namespace
) -> bool:
    needed = ("qc", "qc_actor_risky", "actor_qc_percentile")
    if not all(has_metric(rows, metric) for metric in needed):
        return False
    filters: list[Filter] = [
        lambda row: row_condition(row) in {"cmdp_active", "dense_proximity"},
        lambda row: row_depth(row) in {"", args.depth_for_mechanism},
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.6), sharex=True)
    for ax, label in zip_equal(axes, "ABC"):
        panel_label(ax, label)
        ax.set_xlabel("Epoch")

    ok = plot_single_metric(
        axes[0],
        rows,
        "qc",
        filters=filters,
        color="#4E79A7",
        label="mean Qc",
        args=args,
    )
    ok |= plot_single_metric(
        axes[0],
        rows,
        "qc_actor_risky",
        filters=filters,
        color="#D55E00",
        label="risky-state Qc",
        args=args,
    )
    axes[0].set_title("State-risk awareness")
    axes[0].set_ylabel("Qc")
    axes[0].legend(frameon=False)

    ok |= plot_single_metric(
        axes[1],
        rows,
        "actor_qc_percentile",
        filters=filters,
        color="#D55E00",
        label="actor rank",
        args=args,
    )
    axes[1].axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.1)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Actor rank remains random")
    axes[1].set_ylabel("Percentile")

    if has_metric(rows, "best_qc_action_is_actor_frac"):
        ok |= plot_single_metric(
            axes[2],
            rows,
            "best_qc_action_is_actor_frac",
            filters=filters,
            color="#D55E00",
            label="actor best-Qc",
            args=args,
        )
    axes[2].axhline(1 / 35, color="#6B7280", linestyle="--", linewidth=1.1)
    axes[2].set_title("Best-action test")
    axes[2].set_ylabel("Fraction")

    if not ok:
        plt.close(fig)
        return False
    save_figure(fig, out_dir, "fig03_qc_state_risk_action_flatness", args.dpi)
    return True


def replay_group(row: dict[str, str]) -> str:
    value = numeric(row, "cost_risk_replay_ratio_actual")
    if value is None:
        value = numeric(row, "cost_risk_replay_ratio")
    if value is None:
        return ""
    return f"{value:g}"


def plot_figure4(
    rows: list[dict[str, str]], out_dir: Path, args: argparse.Namespace
) -> bool:
    metrics = (
        "cost_risky_batch_frac",
        "actor_qc_percentile",
        "hard_viol",
        "hazard",
    )
    if not all(has_metric(rows, metric) for metric in metrics):
        return False
    ratios = {replay_group(row) for row in rows if replay_group(row)}
    if len(ratios) < 2:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1), sharex=True)
    for ax, metric, label in zip_equal(axes.ravel(), metrics, "ABCD"):
        panel_label(ax, label)
        curves = aggregate_curves(
            rows,
            group_fn=replay_group,
            metric=metric,
            filters=(lambda row: replay_group(row) != "",),
            smooth_window=args.smooth_window,
            interval=args.interval,
        )
        plot_curves(ax, curves, colors=RATIO_COLORS)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.1)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        [f"replay {label}" for label in legend_labels],
        loc="upper center",
        ncol=max(1, len(legend_labels)),
        frameon=False,
        bbox_to_anchor=(0.52, 1.04),
    )
    save_figure(fig, out_dir, "fig04_risk_replay_changes_sampling_not_safety", args.dpi)
    return True


def plot_figure5(
    summary_rows: list[dict[str, str]], out_dir: Path, args: argparse.Namespace
) -> bool:
    spread_metrics = (
        "true_action_active_cost_spread",
        "true_action_sparse_cost_spread",
        "true_action_dense_cost_spread",
        "true_action_hard_viol_spread",
        "true_action_hazard_spread",
        "q_c_action_spread",
    )
    if not all(has_metric(summary_rows, metric) for metric in spread_metrics):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65))
    for ax, label in zip_equal(axes, "ABC"):
        panel_label(ax, label)

    labels = ["active cost", "sparse cost", "dense cost", "hard viol.", "hazard", "Qc"]
    colors = ["#4E79A7", "#6B7280", "#0072B2", "#9CA3AF", "#59A14F", "#D55E00"]
    means: list[float] = []
    bands: list[float] = []
    for metric in spread_metrics:
        mean, band = mean_band(summary_values(summary_rows, metric), args.interval)
        means.append(mean)
        bands.append(band)
    x = np.arange(len(labels))
    axes[0].bar(x, means, yerr=bands, color=colors, width=0.62, capsize=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_title("True labels are flat")
    axes[0].set_ylabel("Action spread")

    corr_metrics = ("corr_qc_true_cost", "corr_qc_true_hazard")
    corr_labels = ["cost", "hazard"]
    corr_means: list[float] = []
    corr_bands: list[float] = []
    for metric in corr_metrics:
        mean, band = mean_band(summary_values(summary_rows, metric), args.interval)
        corr_means.append(mean)
        corr_bands.append(band)
    axes[1].bar(corr_labels, corr_means, yerr=corr_bands, color="#D55E00", capsize=2)
    axes[1].axhline(0.0, color="#6B7280", linewidth=1.1)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_title("Qc is uncorrelated")
    axes[1].set_ylabel("Correlation")

    rank_metrics = ("actor_true_cost_percentile", "actor_qc_percentile_cf")
    rank_labels = ["true cost", "Qc"]
    rank_means: list[float] = []
    rank_bands: list[float] = []
    for metric in rank_metrics:
        mean, band = mean_band(summary_values(summary_rows, metric), args.interval)
        rank_means.append(mean)
        rank_bands.append(band)
    axes[2].bar(rank_labels, rank_means, yerr=rank_bands, color="#4E79A7", capsize=2)
    axes[2].axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.1)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Actor ranks near random")
    axes[2].set_ylabel("Percentile")

    save_figure(fig, out_dir, "fig05_one_step_sparse_action_flatness", args.dpi)
    return True


def multistep_series(
    summary_rows: list[dict[str, str]], metric_template: str, interval: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizons = []
    means = []
    bands = []
    for horizon in (5, 10, 20):
        metric = metric_template.format(horizon=horizon)
        values = summary_values(summary_rows, metric)
        if not values:
            continue
        mean, band = mean_band(values, interval)
        horizons.append(horizon)
        means.append(mean)
        bands.append(band)
    return np.asarray(horizons), np.asarray(means), np.asarray(bands)


def plot_figure6(
    summary_rows: list[dict[str, str]], out_dir: Path, args: argparse.Namespace
) -> bool:
    if not has_metric(summary_rows, "cf_5_final_hazard_dist_spread"):
        return False
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    series = {
        "sparse cost": (
            "cf_{horizon}_true_cost_spread",
            "#6B7280",
            "o",
        ),
        "hard violation": (
            "cf_{horizon}_true_hard_viol_spread",
            "#9CA3AF",
            "s",
        ),
        "hazard distance": (
            "cf_{horizon}_final_hazard_dist_spread",
            "#D55E00",
            "^",
        ),
    }
    for label, (template, color, marker) in series.items():
        xs, means, bands = multistep_series(summary_rows, template, args.interval)
        if xs.size == 0:
            continue
        ax.plot(xs, means, color=color, marker=marker, linewidth=2.0, label=label)
        ax.fill_between(xs, means - bands, means + bands, color=color, alpha=0.18)
    ax.set_xticks([5, 10, 20])
    ax.set_xlabel("Counterfactual horizon H")
    ax.set_ylabel("Action-induced spread")
    ax.set_title("Geometry separates actions; sparse costs do not")
    ax.legend(frameon=False)
    ax.annotate(
        "hazard geometry grows",
        xy=(20, ax.lines[-1].get_ydata()[-1]),
        xytext=(9, ax.get_ylim()[1] * 0.75),
        arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 1.1},
        color="#D55E00",
        fontsize=8,
    )
    save_figure(
        fig,
        out_dir,
        "fig06_multistep_sparse_flat_geometry_sensitive",
        args.dpi,
    )
    return True


def plot_figure7(
    epoch_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    out_dir: Path,
    args: argparse.Namespace,
) -> bool:
    modes = {row.get("cost_mode") for row in [*epoch_rows, *summary_rows]}
    if not {"sparse", "dense_proximity"}.issubset(modes):
        return False
    metrics = (
        "actor_qc_percentile",
        "corr_qc_true_cost",
        "hard_viol",
        "eval_ever_reached",
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1))
    if epoch_rows and all(has_metric(epoch_rows, metric) for metric in metrics):
        for ax, metric, label in zip_equal(axes.ravel(), metrics, "ABCD"):
            panel_label(ax, label)
            curves = aggregate_curves(
                epoch_rows,
                group_fn=lambda row: row.get("cost_mode", "sparse"),
                metric=metric,
                filters=(lambda row: row.get("cost_mode") in modes,),
                smooth_window=args.smooth_window,
                interval=args.interval,
            )
            plot_curves(
                ax,
                curves,
                colors=CONDITION_COLORS,
                labels={"sparse": "sparse", "dense_proximity": "dense"},
            )
            ax.set_title(METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("Epoch")
    elif summary_rows:
        for ax, metric, label in zip_equal(axes.ravel(), metrics, "ABCD"):
            panel_label(ax, label)
            values = []
            bands = []
            for mode in ("sparse", "dense_proximity"):
                rows = [row for row in summary_rows if row.get("cost_mode") == mode]
                mean, band = mean_band(summary_values(rows, metric), args.interval)
                values.append(mean)
                bands.append(band)
            ax.bar(
                ["sparse", "dense"],
                values,
                yerr=bands,
                color=[CONDITION_COLORS["sparse"], CONDITION_COLORS["dense_proximity"]],
                capsize=2,
            )
            ax.set_title(METRIC_LABELS.get(metric, metric))
    else:
        plt.close(fig)
        return False
    save_figure(fig, out_dir, "fig07_dense_cost_ablation_scaffold", args.dpi)
    return True


def write_captions(out_dir: Path) -> None:
    text = """# Figure Captions and Placement

## Main Paper Candidates

**Figure 1. Depth improves reaching but not safety.**
Mean learning curves over seeds show that larger residual towers improve
`eval_ever_reached`, while `hard_viol` does not decrease. This separates task
representation gains from safety-control gains.

**Figure 5. Sparse safety labels are locally action-flat.**
Same-state counterfactuals show near-zero sparse cost and hard-violation spread
across candidate actions, even though the learned cost critic has nonzero action
spread. Qc action variation is therefore not aligned with sparse safety labels.

**Figure 6. Hazard proximity separates actions, sparse costs do not.**
Multi-step counterfactual probes show flat sparse returns at H=5,10,20, while
final hazard-distance spread increases with horizon. The environment contains
action-sensitive geometry, but thresholded CMDP costs collapse it.

## Mechanism / Presentation

**Figure 2. CMDP pressure is active.**
The dual-weighted cost term and actor-side cost-gradient ratios are nonzero,
including on risky states, ruling out an inactive-constraint explanation.

**Figure 3. Qc learns risky states but not risky actions.**
Qc assigns higher value to risky states, yet actor action rank stays near the
random percentile and rarely selects the best-Qc action.

**Figure 4. Risk replay changes sampling but not safety.**
Risk-biased replay increases risky batch fraction as intended, but action
ranking and safety metrics remain nearly unchanged.

## Future Main Figure

**Figure 7. Dense-cost ablation.**
Sparse vs dense hazard-proximity cost under the same SR-CPO pipeline. This is
the decisive causal ablation for whether target geometry, not architecture or
PID plumbing, controls action identifiability.
"""
    (out_dir / "captions.md").write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render publication-quality SR-CPO research figures."
    )
    parser.add_argument("--epoch-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    parser.add_argument("--interval", choices=("sem", "std"), default="sem")
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--depths", default="2,4,8,16")
    parser.add_argument("--fig1-condition", default="cmdp_active")
    parser.add_argument("--depth-for-mechanism", default="8")
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args(argv)

    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows = read_rows([args.epoch_csv] if args.epoch_csv else [])
    summary_rows = read_rows(args.summary_csv)
    made = {
        "fig01": plot_figure1(epoch_rows, args.out_dir, args),
        "fig02": plot_figure2(epoch_rows, args.out_dir, args),
        "fig03": plot_figure3(epoch_rows, args.out_dir, args),
        "fig04": plot_figure4(epoch_rows, args.out_dir, args),
        "fig05": plot_figure5(summary_rows, args.out_dir, args),
        "fig06": plot_figure6(summary_rows, args.out_dir, args),
        "fig07": plot_figure7(epoch_rows, summary_rows, args.out_dir, args),
    }
    write_captions(args.out_dir)
    for name, status in made.items():
        print(f"{name}: {'rendered' if status else 'skipped'}", file=sys.stderr)
    return 0 if any(made.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
