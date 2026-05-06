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
    "COST_MODE",
    "COST_DENSE_PROX_TAU",
)
METRIC_KEYS = (
    "hard_viol",
    "cost",
    "dense_cost_mean",
    "dense_cost_std",
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
MULTISTEP_COUNTERFACTUAL_HORIZONS = (5, 10, 20)
MULTISTEP_COUNTERFACTUAL_BASE_KEYS = (
    "true_cost_spread",
    "true_hard_viol_spread",
    "true_hazard_spread",
    "frac_nonzero_cost_spread",
    "frac_nonzero_hard_viol_spread",
    "frac_nonzero_hazard_spread",
    "corr_qc_true_cost",
    "corr_qc_true_hazard",
    "actor_true_cost_percentile",
    "actor_true_hazard_percentile",
    "actor_qc_percentile",
    "best_qc_matches_best_true_cost_frac",
    "best_qc_matches_best_true_hazard_frac",
    "qc_action_spread",
    "min_hazard_dist_over_h_spread",
    "final_hazard_dist_spread",
    "hazard_dist_available_frac",
    "costpos_frac",
    "hardpos_frac",
    "haz1_frac",
    "haz05_frac",
    "haz025_frac",
    "true_cost_spread_costpos",
    "corr_qc_true_cost_costpos",
    "true_cost_spread_hardpos",
    "corr_qc_true_cost_hardpos",
    "true_cost_spread_haz1",
    "corr_qc_true_cost_haz1",
    "true_cost_spread_haz05",
    "corr_qc_true_cost_haz05",
    "true_cost_spread_haz025",
    "corr_qc_true_cost_haz025",
)
MULTISTEP_COUNTERFACTUAL_METRIC_KEYS = tuple(
    f"cf_{horizon}_{key}"
    for horizon in MULTISTEP_COUNTERFACTUAL_HORIZONS
    for key in MULTISTEP_COUNTERFACTUAL_BASE_KEYS
)
COUNTERFACTUAL_METRIC_KEYS = (
    "true_action_cost_spread",
    "true_action_hard_viol_spread",
    "true_action_hazard_spread",
    "qc_action_spread",
    "corr_qc_true_cost",
    "corr_qc_true_hazard",
    "actor_true_cost_percentile",
    "actor_qc_percentile_cf",
    "actor_true_hazard_percentile",
    "best_qc_matches_best_true_cost_frac",
    "best_qc_matches_best_true_hazard_frac",
    "frac_states_with_nonzero_true_cost_spread",
    "frac_states_with_nonzero_hazard_spread",
    "cf_hazard_dist_available_frac",
    "cf_costpos_frac",
    "cf_hardpos_frac",
    "cf_haz1_frac",
    "cf_haz05_frac",
    "cf_haz025_frac",
    "true_action_cost_spread_costpos",
    "corr_qc_true_cost_costpos",
    "true_action_cost_spread_hardpos",
    "corr_qc_true_cost_hardpos",
    "true_action_cost_spread_haz1",
    "corr_qc_true_cost_haz1",
    "true_action_cost_spread_haz05",
    "corr_qc_true_cost_haz05",
    "true_action_cost_spread_haz025",
    "corr_qc_true_cost_haz025",
    *MULTISTEP_COUNTERFACTUAL_METRIC_KEYS,
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
    "cost_mode",
    "cost_dense_prox_tau",
    "hard_viol",
    "cost",
    "dense_cost_mean",
    "dense_cost_std",
    "cost_target",
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
    *COUNTERFACTUAL_METRIC_KEYS,
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
        elif (
            "action_rank[" in line
            or "cost_replay[" in line
            or "counterfactual[" in line
            or "counterfactual_H" in line
        ):
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
        "cost_mode": header.get("COST_MODE", ""),
        "cost_dense_prox_tau": header.get("COST_DENSE_PROX_TAU", ""),
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
        "cost_target": last_metrics.get("c_target", ""),
    }
    for key in METRIC_KEYS:
        if key in ("λ̃", "Qc", "λQc_a"):
            continue
        row[key] = last_metrics.get(key, "")
    for key in COUNTERFACTUAL_METRIC_KEYS:
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
