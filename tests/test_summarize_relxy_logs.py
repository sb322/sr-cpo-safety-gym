from __future__ import annotations

import csv
import io
from pathlib import Path

from scripts.summarize_relxy_logs import main, parse_log


def test_parse_log_uses_last_hard_violation_line(tmp_path: Path) -> None:
    log = tmp_path / "safe_relxy_200.123_0.out"
    log.write_text(
        "\n".join(
            [
                "RUN200_LABEL=pid_off_200_seed0",
                "EPOCHS=200",
                "NU_C=0.0003",
                "SEED=0",
                "PID_KP=0.0",
                "PID_KI=0.0",
                "COST_RETURN_LOSS_WEIGHT=0.0",
                "hard_viol=0.1 cost=0.2 hazard=0.3 gdist=4.0 reached=0.0",
                (
                    "hard_viol=0.03 cost=0.04 hazard=0.02 "
                    "vase_contact=0.001 vase_body=0.0000 vase_qpos=0.0000 "
                    "vase_disp=0.0000 cost_resid=0.019 rew=0.02 "
                    "gdist=1.2 g_p10=0.4 g_p50=0.9 g_p90=2.5 "
                    "g_lt0_5=0.1 g_lt1=0.6 g_lt2=0.8 reached=0.01 "
                    "λ̃=0.0 Qc=0.04 Qc_a=0.05 Qc0=0.03 Qc-=0.02 "
                    "dQc_a0=0.02 abs_dQc_a0=0.02 frac_dQc_pos=0.75 "
                    "λQc_a=0.0 r_term=-1.2 dQr_da=3.0 dQc_da=0.4 "
                    "lambda_dQc_da=0.0 grad_ratio=0.0 cos_qr_qc=-0.5 "
                    "risk_frac=0.25 Qc_risk=0.07 dQc_risk=0.04 "
                    "grad_ratio_risk=0.01"
                ),
                (
                    "action_rank[ actor_qc_rank_mean=3.00 "
                    "actor_qc_percentile=0.100 q_c_action_spread=4.00e-01 "
                    "best_qc_action_is_actor_frac=0.250 "
                    "hazard_rank_bins_available=1.000 "
                    "actor_qc_rank_mean_risk1=4.00 "
                    "actor_qc_percentile_risk1=0.200 "
                    "q_c_action_spread_risk1=5.00e-01 "
                    "actor_qc_rank_mean_risk05=5.00 "
                    "actor_qc_percentile_risk05=0.300 "
                    "q_c_action_spread_risk05=6.00e-01 "
                    "actor_qc_rank_mean_risk025=6.00 "
                    "actor_qc_percentile_risk025=0.400 "
                    "q_c_action_spread_risk025=7.00e-01]"
                ),
                (
                    "cost_replay[ cost_risk_replay_ratio_actual=0.250 "
                    "cost_risky_batch_frac=0.375 "
                    "cost_risky_available_frac=0.125 "
                    "cost_risky_batch_mean_cost=0.0800 "
                    "cost_uniform_batch_mean_cost=0.0200]"
                ),
                (
                    "eval_ever_reached=0.9961 eval_first_hit_time=110.02 "
                    "eval_min_goal_dist_initial_goal=0.1160 "
                    "eval_ever_within_0.31=0.9961 eval_ever_within_0.5=0.9961 "
                    "eval_ever_within_1.0=0.9961 eval_ever_within_2.0=0.9961 "
                    "eval_success_count=10.3008 eval_cost_return=119.7852 "
                    "eval_time_at_goal_resampled=0.0103 "
                    "eval_final_goal_dist_resampled=1.3975 "
                    "eval_frozen_time_within_0.31=0.0706 "
                    "eval_frozen_time_within_0.5=0.1341 "
                    "eval_frozen_final_dist=1.5638"
                ),
            ]
        )
    )

    row = parse_log(log)

    assert row["label"] == "pid_off_200_seed0"
    assert row["epochs"] == "200"
    assert row["seed"] == "0"
    assert row["hard_viol"] == "0.03"
    assert row["gdist"] == "1.2"
    assert row["g_p50"] == "0.9"
    assert row["g_lt1"] == "0.6"
    assert row["lambda_tilde"] == "0.0"
    assert row["qc_actor"] == "0.05"
    assert row["qc_action_delta"] == "0.02"
    assert row["qc_action_delta_frac_pos"] == "0.75"
    assert row["grad_norm_qr_a"] == "3.0"
    assert row["grad_norm_qc_a"] == "0.4"
    assert row["cosine_grad_qr_qc"] == "-0.5"
    assert row["risk_condition_fraction"] == "0.25"
    assert row["actor_qc_rank_mean"] == "3.00"
    assert row["actor_qc_percentile"] == "0.100"
    assert row["q_c_action_spread"] == "4.00e-01"
    assert row["best_qc_action_is_actor_frac"] == "0.250"
    assert row["action_rank_hazard_available_frac"] == "1.000"
    assert row["actor_qc_rank_mean_risk1"] == "4.00"
    assert row["actor_qc_percentile_risk05"] == "0.300"
    assert row["q_c_action_spread_risk025"] == "7.00e-01"
    assert row["cost_risk_replay_ratio_actual"] == "0.250"
    assert row["cost_risky_batch_frac"] == "0.375"
    assert row["cost_risky_available_frac"] == "0.125"
    assert row["cost_risky_batch_mean_cost"] == "0.0800"
    assert row["cost_uniform_batch_mean_cost"] == "0.0200"
    assert row["eval_ever_reached"] == "0.9961"
    assert row["eval_first_hit_time"] == "110.02"
    assert row["eval_success_count"] == "10.3008"
    assert row["eval_cost_return"] == "119.7852"
    assert row["eval_time_at_goal_resampled"] == "0.0103"
    assert row["eval_frozen_final_dist"] == "1.5638"


def test_parse_log_uses_300_epoch_label(tmp_path: Path) -> None:
    log = tmp_path / "safe_relxy_300.123_0.out"
    log.write_text(
        "\n".join(
            [
                "RUN300_LABEL=cmdp_nuc3e-3_300_seed0",
                "EPOCHS=300",
                "SEED=0",
                "hard_viol=0.03 cost=0.04 gdist=1.1 reached=0.01",
            ]
        )
    )

    row = parse_log(log)

    assert row["label"] == "cmdp_nuc3e-3_300_seed0"
    assert row["epochs"] == "300"


def test_parse_log_prefers_depth_label_and_records_num_blocks(tmp_path: Path) -> None:
    log = tmp_path / "safe_relxy_depth_pidoff.123_0.out"
    log.write_text(
        "\n".join(
            [
                "DEPTH_LABEL=pidoff_d8_seed0",
                "REL_LABEL=relxy_lidar_mask_l2_d8",
                "NUM_BLOCKS=8",
                "SEED=0",
                "hard_viol=0.03 cost=0.04 gdist=1.1 reached=0.01",
            ]
        )
    )

    row = parse_log(log)

    assert row["label"] == "pidoff_d8_seed0"
    assert row["num_blocks"] == "8"


def test_main_writes_csv(tmp_path: Path, capsys) -> None:
    log = tmp_path / "safe_relxy_long.123_1.out"
    log.write_text(
        "\n".join(
            [
                "LONG_LABEL=cmdp_nuc3e-3_long_seed1",
                "EPOCHS=100",
                "NU_C=0.003",
                "SEED=1",
                "PID_KP=5.0",
                "PID_KI=0.01",
                "COST_RETURN_LOSS_WEIGHT=1.0",
                "hard_viol=0.04 cost=0.05 hazard=0.03 gdist=1.5 reached=0.01",
            ]
        )
    )

    assert main([str(log)]) == 0
    output = capsys.readouterr().out
    rows = list(csv.DictReader(io.StringIO(output)))

    assert len(rows) == 1
    row = rows[0]
    assert row["file"] == "safe_relxy_long.123_1.out"
    assert row["label"] == "cmdp_nuc3e-3_long_seed1"
    assert row["seed"] == "1"
    assert row["epochs"] == "100"
    assert row["num_blocks"] == ""
    assert row["nu_c"] == "0.003"
    assert row["pid_kp"] == "5.0"
    assert row["pid_ki"] == "0.01"
    assert row["cost_return_loss_weight"] == "1.0"
    assert row["hard_viol"] == "0.04"
    assert row["cost"] == "0.05"
    assert row["hazard"] == "0.03"
    assert row["gdist"] == "1.5"
    assert row["reached"] == "0.01"
    assert row["eval_ever_reached"] == ""
