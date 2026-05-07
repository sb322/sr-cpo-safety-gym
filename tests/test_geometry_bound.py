"""Tests for analytical dense-label geometry bounds."""

import pytest

from pathlib import Path
import importlib.util
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "geometry_bound.py"
SPEC = importlib.util.spec_from_file_location("geometry_bound", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
geometry_bound = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry_bound
SPEC.loader.exec_module(geometry_bound)


def test_geometry_bound_monotone_and_h20_sane() -> None:
    constants = geometry_bound.GeometryConstants(max_per_step_displacement=0.002)
    rows = geometry_bound.table_rows(
        constants=constants,
        tau=0.5,
        gamma=0.99,
        horizons=(1, 5, 20, 50),
        label_magnitude=0.1,
        between_state_std=0.1,
    )
    within_between = [row["within_between_bound"] for row in rows]
    assert within_between == sorted(within_between)
    assert 1e-4 <= within_between[2] <= 1e-1


def test_safe_learning_constants_are_readable_when_available() -> None:
    try:
        constants = geometry_bound.read_safe_learning_constants()
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f"safe-learning constants unavailable: {exc}")
    assert constants.max_per_step_displacement > 0.0
    assert constants.hazard_radius > 0.0
