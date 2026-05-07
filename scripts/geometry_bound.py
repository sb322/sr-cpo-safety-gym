#!/usr/bin/env python3
"""Analytical action-geometry bounds for dense Safety-Gym risk labels."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@dataclass(frozen=True)
class GeometryConstants:
    max_per_step_displacement: float
    hazard_radius: float = 0.2


def _first_float(obj: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def read_safe_learning_constants() -> GeometryConstants:
    """Best-effort read of GoToGoal geometry constants without resetting the env."""

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from sr_cpo.env_wrappers import _load_safe_learning_go_to_goal_class

    cls = _load_safe_learning_go_to_goal_class()
    env = cls()
    hazard_radius = (
        _first_float(env, ("hazard_radius", "_hazard_radius", "hazards_radius"))
        or 0.2
    )
    dt = _first_float(env, ("dt", "_dt", "sys.dt"))
    if dt is None and hasattr(env, "sys") and hasattr(env.sys, "dt"):
        dt = float(env.sys.dt)
    action_scale = (
        _first_float(
            env,
            (
                "action_scale",
                "_action_scale",
                "speed",
                "_speed",
                "max_speed",
                "_max_speed",
            ),
        )
        or 1.0
    )
    if dt is None:
        raise RuntimeError("could not infer GoToGoal dt without env reset")
    return GeometryConstants(
        max_per_step_displacement=float(dt) * float(action_scale),
        hazard_radius=float(hazard_radius),
    )


def discounted_sum(gamma: float, horizon: int) -> float:
    return sum(gamma**t for t in range(horizon))


def spread_magnitude_bound(
    *, max_per_step_displacement: float, tau: float, gamma: float, horizon: int
) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return (max_per_step_displacement / tau) * discounted_sum(gamma, horizon)


def within_between_bound(
    *,
    spread_over_magnitude: float,
    label_magnitude: float,
    between_state_std: float,
) -> float:
    # Treat spread as an approximate uniform-action range.
    within_var = (spread_over_magnitude * label_magnitude) ** 2 / 12.0
    return within_var / (between_state_std**2)


def table_rows(
    *,
    constants: GeometryConstants,
    tau: float,
    gamma: float,
    horizons: tuple[int, ...],
    label_magnitude: float,
    between_state_std: float,
) -> list[dict[str, float]]:
    rows = []
    for horizon in horizons:
        spread = spread_magnitude_bound(
            max_per_step_displacement=constants.max_per_step_displacement,
            tau=tau,
            gamma=gamma,
            horizon=horizon,
        )
        rows.append(
            {
                "H": float(horizon),
                "spread_over_label_mag": spread,
                "within_between_bound": within_between_bound(
                    spread_over_magnitude=spread,
                    label_magnitude=label_magnitude,
                    between_state_std=between_state_std,
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--gamma-c", type=float, default=0.99)
    parser.add_argument("--horizons", default="1,5,20,50")
    parser.add_argument("--between-state-std", type=float, default=0.1)
    parser.add_argument("--label-magnitude", type=float, default=0.1)
    parser.add_argument("--max-per-step-displacement", type=float, default=None)
    args = parser.parse_args()

    if args.max_per_step_displacement is None:
        constants = read_safe_learning_constants()
    else:
        constants = GeometryConstants(
            max_per_step_displacement=args.max_per_step_displacement
        )
    horizons = tuple(int(part) for part in args.horizons.split(",") if part)
    rows = table_rows(
        constants=constants,
        tau=args.tau,
        gamma=args.gamma_c,
        horizons=horizons,
        label_magnitude=args.label_magnitude,
        between_state_std=args.between_state_std,
    )

    print(
        "H\tmax_step_disp\thazard_radius\tspread/label_mag\tpred_within_between"
    )
    for row in rows:
        print(
            f"{int(row['H'])}\t"
            f"{constants.max_per_step_displacement:.6g}\t"
            f"{constants.hazard_radius:.3g}\t"
            f"{row['spread_over_label_mag']:.6g}\t"
            f"{row['within_between_bound']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
