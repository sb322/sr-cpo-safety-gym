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
                    "λ̃=0.0 Qc=0.04 λQc_a=0.0"
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
