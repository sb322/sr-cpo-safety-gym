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
    "SEED",
    "EPOCHS",
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
FIELDNAMES = (
    "file",
    "label",
    "seed",
    "epochs",
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
    for line in path.read_text(errors="replace").splitlines():
        if any(line.startswith(f"{key}=") for key in HEADER_KEYS):
            key, value = line.split("=", 1)
            header[key] = value
        elif "hard_viol=" in line:
            parsed = _parse_assignments(line)
            if parsed:
                last_metrics = parsed

    label = (
        header.get("RUN200_LABEL")
        or header.get("RUN300_LABEL")
        or header.get("LONG_LABEL")
        or header.get("FINAL_LABEL")
        or path.stem
    )
    row = {
        "file": path.name,
        "label": label,
        "seed": header.get("SEED", ""),
        "epochs": header.get("EPOCHS", ""),
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
