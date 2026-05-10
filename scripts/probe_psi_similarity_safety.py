#!/usr/bin/env python3
"""Placeholder entrypoint for the optional psi-similarity safety probe.

This probe needs full training-state checkpoints or replay snapshots containing
the trained encoders and hazard states. Actor-only checkpoints are insufficient.
The paper pipeline consumes figures/data/paper/psi_safety_probe.csv when that
probe has been produced separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures/data/paper/psi_safety_probe.csv"))
    parser.parse_args(argv)
    raise SystemExit(
        "psi-similarity probe requires full encoder/replay checkpoints; "
        "no compatible checkpoint format is available in this repository yet"
    )


if __name__ == "__main__":
    raise SystemExit(main())
