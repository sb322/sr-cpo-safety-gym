#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HEADER_KEYS = (
    "FINAL_LABEL",
    "LONG_LABEL",
    "RUN200_LABEL",
    "RUN300_LABEL",
    "DEPTH_LABEL",
    "REL_LABEL",
    "SEED",
    "EPOCHS",
    "NUM_BLOCKS",
    "NU_C",
    "PID_KP",
    "PID_KI",
    "COST_RETURN_LOSS_WEIGHT",
)
METRIC_KEYS = (
    "hard_viol",
    "cost",
    "hazard",
    "vase_contact",
    "vase_body",
    "vase_qpos",
    "vase_disp",
    "cost_resid",
    "rew",
    "gdist",
    "g_p10",
    "g_p50",
    "g_p90",
    "g_lt0_5",
    "g_lt1",
    "g_lt2",
    "reached",
    "λ̃",
    "Qc",
    "λQc_a",
)
EVAL_METRIC_KEYS = (
    "eval_ever_reached",
    "eval_first_hit_time",
    "eval_min_goal_dist_initial_goal",
    "eval_ever_within_0.31",
    "eval_ever_within_0.5",
    "eval_ever_within_1.0",
    "eval_ever_within_2.0",
    "eval_success_count",
    "eval_cost_return",
    "eval_time_at_goal_resampled",
    "eval_final_goal_dist_resampled",
    "eval_frozen_time_within_0.31",
    "eval_frozen_time_within_0.5",
    "eval_frozen_final_dist",
    "eval_ever_reached_std0_25",
    "eval_first_hit_time_std0_25",
    "eval_min_goal_dist_initial_goal_std0_25",
    "eval_ever_within_0.31_std0_25",
    "eval_success_count_std0_25",
    "eval_cost_return_std0_25",
    "eval_ever_reached_std0_5",
    "eval_first_hit_time_std0_5",
    "eval_min_goal_dist_initial_goal_std0_5",
    "eval_ever_within_0.31_std0_5",
    "eval_success_count_std0_5",
    "eval_cost_return_std0_5",
    "eval_ever_reached_std1",
    "eval_first_hit_time_std1",
    "eval_min_goal_dist_initial_goal_std1",
    "eval_ever_within_0.31_std1",
    "eval_success_count_std1",
    "eval_cost_return_std1",
)
FIELDNAMES = (
    "file",
    "label",
    "seed",
    "epochs",
    "num_blocks",
    "nu_c",
    "pid_kp",
    "pid_ki",
    "cost_return_loss_weight",
    "hard_viol",
    "cost",
    "hazard",
    "vase_contact",
    "vase_body",
    "vase_qpos",
    "vase_disp",
    "cost_resid",
    "rew",
    "gdist",
    "g_p10",
    "g_p50",
    "g_p90",
    "g_lt0_5",
    "g_lt1",
    "g_lt2",
    "reached",
    "lambda_tilde",
    "qc",
    "lambda_qc_a",
    *EVAL_METRIC_KEYS,
)


def _parse_assignments(line: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs[key] = value
    return pairs


def parse_log(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    last_metrics: dict[str, str] = {}
    last_eval_metrics: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if any(line.startswith(f"{key}=") for key in HEADER_KEYS):
            key, value = line.split("=", 1)
            header[key] = value
        elif "hard_viol=" in line:
            parsed = _parse_assignments(line)
            if parsed:
                last_metrics = parsed
        elif "eval_ever_reached=" in line or "eval_ever_reached_std" in line:
            parsed = _parse_assignments(line)
            if parsed:
                last_eval_metrics.update(parsed)

    label = (
        header.get("DEPTH_LABEL")
        or header.get("RUN200_LABEL")
        or header.get("RUN300_LABEL")
        or header.get("LONG_LABEL")
        or header.get("FINAL_LABEL")
        or header.get("REL_LABEL")
        or path.stem
    )
    row = {
        "file": path.name,
        "label": label,
        "seed": header.get("SEED", ""),
        "epochs": header.get("EPOCHS", ""),
        "num_blocks": header.get("NUM_BLOCKS", ""),
        "nu_c": header.get("NU_C", ""),
        "pid_kp": header.get("PID_KP", ""),
        "pid_ki": header.get("PID_KI", ""),
        "cost_return_loss_weight": header.get("COST_RETURN_LOSS_WEIGHT", ""),
        "lambda_tilde": last_metrics.get("λ̃", ""),
        "qc": last_metrics.get("Qc", ""),
        "lambda_qc_a": last_metrics.get("λQc_a", ""),
    }
    for key in METRIC_KEYS:
        if key in ("λ̃", "Qc", "λQc_a"):
            continue
        row[key] = last_metrics.get(key, "")
    for key in EVAL_METRIC_KEYS:
        row[key] = last_eval_metrics.get(key, "")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize relative-XY SLURM logs into one CSV row per file."
    )
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    for path in args.logs:
        if path.is_file():
            writer.writerow(parse_log(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
