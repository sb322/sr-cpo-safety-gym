#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
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
    "COST_RANK_LOSS_WEIGHT",
    "COST_RANK_HORIZON",
    "COST_RANK_NUM_CANDIDATES",
    "COST_RANK_STATES_PER_EPOCH",
    "COST_RANK_BUFFER_CAPACITY",
    "COST_RANK_BATCH_SIZE",
    "COST_RANK_CANDIDATE_PERTURB_STD",
    "COST_RANK_UNIFORM_RANDOM_FRAC",
    "COST_RANK_LABEL_EPSILON",
    "COST_RANK_MIN_LABEL_SPREAD",
    "COST_RANK_LABEL_KIND",
    "COST_RISK_REPLAY_RATIO",
    "COST_MODE",
    "COST_DENSE_PROX_TAU",
)

HEADER_TO_FIELD = {
    "SEED": "seed",
    "EPOCHS": "epochs",
    "NUM_BLOCKS": "num_blocks",
    "NU_C": "nu_c",
    "PID_KP": "pid_kp",
    "PID_KI": "pid_ki",
    "COST_RETURN_LOSS_WEIGHT": "cost_return_loss_weight",
    "COST_RANK_LOSS_WEIGHT": "cost_rank_loss_weight",
    "COST_RANK_HORIZON": "cost_rank_horizon",
    "COST_RANK_NUM_CANDIDATES": "cost_rank_num_candidates",
    "COST_RANK_STATES_PER_EPOCH": "cost_rank_states_per_epoch",
    "COST_RANK_BUFFER_CAPACITY": "cost_rank_buffer_capacity",
    "COST_RANK_BATCH_SIZE": "cost_rank_batch_size",
    "COST_RANK_CANDIDATE_PERTURB_STD": "cost_rank_candidate_perturb_std",
    "COST_RANK_UNIFORM_RANDOM_FRAC": "cost_rank_uniform_random_frac",
    "COST_RANK_LABEL_EPSILON": "cost_rank_label_epsilon",
    "COST_RANK_MIN_LABEL_SPREAD": "cost_rank_min_label_spread",
    "COST_RANK_LABEL_KIND": "cost_rank_label_kind",
    "COST_RISK_REPLAY_RATIO": "cost_risk_replay_ratio",
    "COST_MODE": "cost_mode",
    "COST_DENSE_PROX_TAU": "cost_dense_prox_tau",
}

ALIASES = {
    "λ̃": "lambda_tilde",
    "Ĵ_c": "jc_hat",
    "Qc": "qc",
    "TD": "td_target",
    "λraw": "pid_raw_lambda",
    "Qc_a": "qc_actor",
    "Qc0": "qc_zero_action_actor",
    "Qc-": "qc_neg_action_actor",
    "Jc_mc": "cost_return",
    "Qc-Jc": "qc_return_error",
    "ΔQc_a0": "qc_action_delta_actor",
    "dQc_a0": "qc_action_delta_actor",
    "abs_dQc_a0": "qc_action_gap_actor",
    "frac_dQc_pos": "qc_action_delta_frac_pos_actor",
    "λQc_a": "lambda_qc_a",
    "r_term": "reward_actor_term",
    "dQr_da": "grad_norm_qr_a",
    "dQc_da": "grad_norm_qc_a",
    "lambda_dQc_da": "lambda_grad_norm_qc_a",
    "grad_ratio": "grad_ratio_cost_reward",
    "cos_qr_qc": "cosine_grad_qr_qc",
    "risk_frac": "risk_condition_fraction",
    "Qc_risk": "qc_actor_risky",
    "dQc_risk": "qc_action_delta_risky",
    "grad_ratio_risk": "grad_ratio_cost_reward_risky",
    "hazard_rank_bins_available": "action_rank_hazard_available_frac",
}

BASE_FIELDS = (
    "file",
    "label",
    "condition",
    "seed",
    "epoch",
    "epochs",
    "steps",
    "num_blocks",
    "nu_c",
    "pid_kp",
    "pid_ki",
    "cost_return_loss_weight",
    "cost_rank_loss_weight",
    "cost_rank_horizon",
    "cost_rank_num_candidates",
    "cost_rank_states_per_epoch",
    "cost_rank_buffer_capacity",
    "cost_rank_batch_size",
    "cost_rank_candidate_perturb_std",
    "cost_rank_uniform_random_frac",
    "cost_rank_label_epsilon",
    "cost_rank_min_label_spread",
    "cost_rank_label_kind",
    "cost_risk_replay_ratio",
    "cost_mode",
    "cost_dense_prox_tau",
)

EPOCH_RE = re.compile(r"^\[\s*(\d+)\s*/\s*(\d+)\s*\]\s+steps=([\d,]+)")


def _parse_assignments(line: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.lstrip("[")
        value = value.rstrip("],")
        pairs[ALIASES.get(key, key)] = value
    return pairs


def _label_from_header(header: dict[str, str], fallback: str) -> str:
    return (
        header.get("DEPTH_LABEL")
        or header.get("RUN200_LABEL")
        or header.get("RUN300_LABEL")
        or header.get("LONG_LABEL")
        or header.get("FINAL_LABEL")
        or header.get("REL_LABEL")
        or fallback
    )


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _condition(header: dict[str, str], label: str) -> str:
    lowered = label.lower()
    if "pid_off" in lowered or "pid-off" in lowered:
        return "pid_off"
    if "dense" in lowered:
        return "dense_proximity"
    nu_c = _float_or_none(header.get("NU_C"))
    pid_kp = _float_or_none(header.get("PID_KP"))
    pid_ki = _float_or_none(header.get("PID_KI"))
    if (nu_c is not None and nu_c <= 0.0) or (
        pid_kp is not None and pid_kp <= 0.0 and pid_ki is not None and pid_ki <= 0.0
    ):
        return "pid_off"
    cost_mode = header.get("COST_MODE", "")
    if cost_mode == "dense_proximity":
        return "dense_proximity"
    return "cmdp_active"


def parse_log(path: Path) -> list[dict[str, str]]:
    header: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        if current is not None:
            rows.append(current.copy())

    for line in path.read_text(errors="replace").splitlines():
        if any(line.startswith(f"{key}=") for key in HEADER_KEYS):
            key, value = line.split("=", 1)
            header[key] = value
            continue

        match = EPOCH_RE.match(line)
        if match:
            flush()
            label = _label_from_header(header, path.stem)
            current = {
                "file": path.name,
                "label": label,
                "condition": _condition(header, label),
                "epoch": match.group(1),
                "epochs": match.group(2),
                "steps": match.group(3).replace(",", ""),
            }
            for header_key, field in HEADER_TO_FIELD.items():
                if header_key in header:
                    current[field] = header[header_key]
            current.update(_parse_assignments(line))
            continue

        if current is None or "=" not in line:
            continue
        current.update(_parse_assignments(line))

    flush()
    return rows


def write_rows(rows: list[dict[str, str]]) -> None:
    extra_fields = sorted({key for row in rows for key in row} - set(BASE_FIELDS))
    writer = csv.DictWriter(sys.stdout, fieldnames=[*BASE_FIELDS, *extra_fields])
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one row per epoch from SR-CPO grep-friendly SLURM logs. "
            "Use this as the curve input for publication plots."
        )
    )
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    rows: list[dict[str, str]] = []
    for path in args.logs:
        if path.is_file():
            rows.extend(parse_log(path))
    if not rows:
        return 1
    write_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
