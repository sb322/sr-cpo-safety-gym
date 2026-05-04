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
    "qc_actor",
    "reward_actor_term",
    "qc_action_delta",
    "qc_action_gap",
    "qc_action_delta_frac_pos",
    "grad_norm_qr_a",
    "grad_norm_qc_a",
    "lambda_grad_norm_qc_a",
    "grad_ratio_cost_reward",
    "cosine_grad_qr_qc",
    "risk_condition_fraction",
    "qc_actor_risky",
    "qc_action_delta_risky",
    "grad_ratio_cost_reward_risky",
    "actor_qc_rank_mean",
    "actor_qc_percentile",
    "q_c_action_spread",
    "best_qc_action_is_actor_frac",
    "action_rank_hazard_available_frac",
    "actor_qc_rank_mean_risk1",
    "actor_qc_percentile_risk1",
    "q_c_action_spread_risk1",
    "actor_qc_rank_mean_risk05",
    "actor_qc_percentile_risk05",
    "q_c_action_spread_risk05",
    "actor_qc_rank_mean_risk025",
    "actor_qc_percentile_risk025",
    "q_c_action_spread_risk025",
    "cost_risk_replay_ratio_actual",
    "cost_risky_batch_frac",
    "cost_risky_available_frac",
    "cost_risky_batch_mean_cost",
    "cost_uniform_batch_mean_cost",
    *EVAL_METRIC_KEYS,
)


def _parse_assignments(line: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs[key] = value.rstrip("]")
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
        elif "action_rank[" in line:
            parsed = _parse_assignments(line)
            if parsed:
                last_metrics.update(parsed)
        elif "cost_replay[" in line:
            parsed = _parse_assignments(line)
            if parsed:
                last_metrics.update(parsed)
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
        "qc_actor": last_metrics.get("Qc_a", ""),
        "reward_actor_term": last_metrics.get("r_term", ""),
        "qc_action_delta": last_metrics.get(
            "dQc_a0", last_metrics.get("ΔQc_a0", "")
        ),
        "qc_action_gap": last_metrics.get("abs_dQc_a0", ""),
        "qc_action_delta_frac_pos": last_metrics.get("frac_dQc_pos", ""),
        "grad_norm_qr_a": last_metrics.get("dQr_da", ""),
        "grad_norm_qc_a": last_metrics.get("dQc_da", ""),
        "lambda_grad_norm_qc_a": last_metrics.get("lambda_dQc_da", ""),
        "grad_ratio_cost_reward": last_metrics.get("grad_ratio", ""),
        "cosine_grad_qr_qc": last_metrics.get("cos_qr_qc", ""),
        "risk_condition_fraction": last_metrics.get("risk_frac", ""),
        "qc_actor_risky": last_metrics.get("Qc_risk", ""),
        "qc_action_delta_risky": last_metrics.get("dQc_risk", ""),
        "grad_ratio_cost_reward_risky": last_metrics.get("grad_ratio_risk", ""),
        "actor_qc_rank_mean": last_metrics.get("actor_qc_rank_mean", ""),
        "actor_qc_percentile": last_metrics.get("actor_qc_percentile", ""),
        "q_c_action_spread": last_metrics.get("q_c_action_spread", ""),
        "best_qc_action_is_actor_frac": last_metrics.get(
            "best_qc_action_is_actor_frac", ""
        ),
        "action_rank_hazard_available_frac": last_metrics.get(
            "hazard_rank_bins_available", ""
        ),
        "actor_qc_rank_mean_risk1": last_metrics.get(
            "actor_qc_rank_mean_risk1", ""
        ),
        "actor_qc_percentile_risk1": last_metrics.get(
            "actor_qc_percentile_risk1", ""
        ),
        "q_c_action_spread_risk1": last_metrics.get("q_c_action_spread_risk1", ""),
        "actor_qc_rank_mean_risk05": last_metrics.get(
            "actor_qc_rank_mean_risk05", ""
        ),
        "actor_qc_percentile_risk05": last_metrics.get(
            "actor_qc_percentile_risk05", ""
        ),
        "q_c_action_spread_risk05": last_metrics.get(
            "q_c_action_spread_risk05", ""
        ),
        "actor_qc_rank_mean_risk025": last_metrics.get(
            "actor_qc_rank_mean_risk025", ""
        ),
        "actor_qc_percentile_risk025": last_metrics.get(
            "actor_qc_percentile_risk025", ""
        ),
        "q_c_action_spread_risk025": last_metrics.get(
            "q_c_action_spread_risk025", ""
        ),
        "cost_risk_replay_ratio_actual": last_metrics.get(
            "cost_risk_replay_ratio_actual", ""
        ),
        "cost_risky_batch_frac": last_metrics.get("cost_risky_batch_frac", ""),
        "cost_risky_available_frac": last_metrics.get(
            "cost_risky_available_frac", ""
        ),
        "cost_risky_batch_mean_cost": last_metrics.get(
            "cost_risky_batch_mean_cost", ""
        ),
        "cost_uniform_batch_mean_cost": last_metrics.get(
            "cost_uniform_batch_mean_cost", ""
        ),
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
