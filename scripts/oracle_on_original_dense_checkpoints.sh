#!/usr/bin/env bash
set -euo pipefail

# Paper-section historical comparison only; this is not a gating experiment.
CHECKPOINTS=(
    "checkpoints/dense_tau05_200e_seed0.msgpack"
    "checkpoints/dense_tau05_200e_seed1.msgpack"
    "checkpoints/dense_tau05_200e_seed2.msgpack"
)
OUTPUT="figures/data/oracle_original_dense_curated.csv"
TMP_DIR="figures/data/oracle_original_dense_curated_parts"

mkdir -p "$(dirname "$OUTPUT")" "$TMP_DIR"

for SEED in 0 1 2; do
    .venv/bin/python scripts/oracle_critic_fit.py \
        --seed "$SEED" \
        --actor-checkpoint "${CHECKPOINTS[$SEED]}" \
        --num-states 256 \
        --num-candidates 32 \
        --horizons 50 \
        --fit-labels dense \
        --fit-label-transforms centered \
        --fit-losses rank \
        --fit-goal-modes normal \
        --fit-steps 2000 \
        --fit-bootstrap-samples 1000 \
        --output-csv "$TMP_DIR/seed${SEED}.csv" \
        --no-include-synthetic-action-norm
done

.venv/bin/python - "$OUTPUT" "$TMP_DIR"/seed*.csv <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

output = Path(sys.argv[1])
parts = [Path(path) for path in sys.argv[2:]]
fieldnames: list[str] | None = None
rows: list[dict[str, str]] = []
for path in parts:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or fieldnames or [])
        rows.extend(reader)

with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {output}")
PY
