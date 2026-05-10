#!/usr/bin/env bash
# Helper for manually bisecting the historical dense reach regression.
#
# Usage sketch:
#   git bisect start BAD_COMMIT GOOD_COMMIT
#   git bisect run scripts/bisect_reach_regression.sh
#
# Each step launches a short depth=8 dense run through the existing SLURM
# wrapper and writes a log under figures/data/paper/bisect/. Treat final reach
# below REACH_THRESHOLD as bad. This script is only a helper; do not use it as
# part of the paper figure build.

set -euo pipefail

REACH_THRESHOLD="${REACH_THRESHOLD:-0.5}"
WORKDIR="${WORKDIR:-/mmfs1/home/sb3222/projects/sr-cpo-safety-gym}"
OUTDIR="${OUTDIR:-figures/data/paper/bisect}"
mkdir -p "$OUTDIR"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found; run this helper on the cluster login node." >&2
    exit 125
fi

job_id="$(
    sbatch --parsable \
        --time=01:00:00 \
        --export=ALL,SEED_OVERRIDE=0,EPOCHS_OVERRIDE=50,STEPS_PER_EPOCH_OVERRIDE=7,SGD_STEPS_OVERRIDE=4,NUM_BLOCKS_OVERRIDE=8,COST_MODE_OVERRIDE=dense_proximity,COST_DENSE_PROX_TAU_OVERRIDE=0.5,PID_COST_SOURCE_OVERRIDE=active,SLURM_ARRAY_TASK_ID=3 \
        slurm/relative_xy_sweep.sh
)"
echo "Submitted bisect job ${job_id}"

while squeue -j "$job_id" >/dev/null 2>&1 && [ -n "$(squeue -h -j "$job_id" 2>/dev/null)" ]; do
    sleep 30
done

python scripts/summarize_relxy_logs.py safe_relative_xy."${job_id}"_3.out \
    > "$OUTDIR/bisect_${job_id}.csv"

reach="$(
    python - "$OUTDIR/bisect_${job_id}.csv" <<'PY'
import csv, sys
row = next(csv.DictReader(open(sys.argv[1])))
print(row.get("eval_ever_reached") or "0")
PY
)"
echo "final reach=${reach}; threshold=${REACH_THRESHOLD}"

python - "$reach" "$REACH_THRESHOLD" <<'PY'
import sys
reach = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if reach >= threshold else 1)
PY
