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
                    "gdist=1.2 reached=0.01 λ̃=0.0 Qc=0.04 λQc_a=0.0"
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
    assert row["lambda_tilde"] == "0.0"


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

    assert rows == [
        {
            "file": "safe_relxy_long.123_1.out",
            "label": "cmdp_nuc3e-3_long_seed1",
            "seed": "1",
            "epochs": "100",
            "nu_c": "0.003",
            "pid_kp": "5.0",
            "pid_ki": "0.01",
            "cost_return_loss_weight": "1.0",
            "hard_viol": "0.04",
            "cost": "0.05",
            "hazard": "0.03",
            "vase_contact": "",
            "vase_body": "",
            "vase_qpos": "",
            "vase_disp": "",
            "cost_resid": "",
            "rew": "",
            "gdist": "1.5",
            "reached": "0.01",
            "lambda_tilde": "",
            "qc": "",
            "lambda_qc_a": "",
        }
    ]
