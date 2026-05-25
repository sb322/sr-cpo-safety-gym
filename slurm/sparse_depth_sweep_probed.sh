#!/bin/bash -l
#SBATCH --job-name=sr_cpo_dsparse
#SBATCH --output=safe_depth_sparse.%A_%a.out
#SBATCH --error=safe_depth_sparse.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-11

set -euo pipefail

# ---------------------------------------------------------------------------
# FINAL CONFIRMATION RUN (closure only -- NO method change).
#
# Purpose: complete the SPARSE column of the depth x supervision matrix with
# counterfactual Q_c action-sensitivity diagnostics. The trusted sparse cells
# currently have corr_qc_true_cost / actor_qc_percentile_cf populated ONLY at
# depth 8; depths 4/16/32 are NaN and sparse-d32 is a single dead cell.
#
# This is the exact analogue of slurm/depth_dense_sweep.sh with two changes:
#   COST_MODE_OVERRIDE=sparse        (vs dense_proximity)
#   PID_COST_SOURCE_OVERRIDE=active  (matches the existing trusted sparse cells)
# Counterfactual probes are ON, exactly as in the dense sweep, so the rebuilt
# cells.csv will SUPERSEDE the NaN sparse cells (build_paper_dataset.py prefers
# cells that have corr_qc_true_cost populated).
#
# Output prefix `safe_depth_sparse.*.out` is matched by build_paper_dataset.py
# DEFAULT_LOG_GLOBS ("safe_depth*.out"); no parser change needed.
# ---------------------------------------------------------------------------

WORKDIR="/mmfs1/home/sb3222/projects/sr-cpo-safety-gym"
cd "$WORKDIR" || exit 1

DEPTH_LABELS=("sparse_depth4" "sparse_depth8" "sparse_depth16" "sparse_depth32")
NUM_BLOCK_VALUES=("4" "8" "16" "32")
SEED_VALUES=("0" "1" "2")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DEPTH_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))
DEPTH_LABEL="${DEPTH_LABELS[$DEPTH_INDEX]}"
NUM_BLOCK="${NUM_BLOCK_VALUES[$DEPTH_INDEX]}"
SEED="${SEED_VALUES[$SEED_INDEX]}"

echo "===== SPARSE DEPTH SWEEP (PROBED, CLOSURE) ====="
echo "DEPTH_LABEL=$DEPTH_LABEL"
echo "OUTER_ARRAY_TASK_ID=$TASK_ID"
echo "INNER_RELATIVE_XY_TASK_ID=3"
echo "SEED_OVERRIDE=$SEED"
echo "EPOCHS_OVERRIDE=200"
echo "STEPS_PER_EPOCH_OVERRIDE=7"
echo "SGD_STEPS_OVERRIDE=4"
echo "NUM_BLOCKS_OVERRIDE=$NUM_BLOCK"
echo "NU_C_OVERRIDE=0.0003"
echo "ENTROPY_PARAM_OVERRIDE=0.5"
echo "COST_MODE_OVERRIDE=sparse"
echo "PID_COST_SOURCE_OVERRIDE=active"
echo "COST_LIMIT_OVERRIDE=0.0001"
echo "PID_KP_OVERRIDE=5.0"
echo "PID_KI_OVERRIDE=0.1"
echo "PID_KD_OVERRIDE=0.0"
echo "PID_INTEGRAL_DECAY_OVERRIDE=1.0"
echo "PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true"
echo "EVAL_COUNTERFACTUAL_ACTION_PROBES_OVERRIDE=true"
echo "COUNTERFACTUAL_PROBE_RANDOM_ACTIONS_OVERRIDE=8"
echo "COUNTERFACTUAL_PROBE_PERTURB_ACTIONS_OVERRIDE=8"
echo "COUNTERFACTUAL_PROBE_INTERVAL_OVERRIDE=200"
echo ""

# Same static guard as depth_dense_sweep.sh: fail fast if the checkout is not
# the probed implementation.
"$WORKDIR/.venv/bin/python" - <<'PYCHECK'
import pathlib

src_train = pathlib.Path("src/sr_cpo/train.py").read_text()
src_relxy = pathlib.Path("slurm/relative_xy_sweep.sh").read_text()

assert "pid_cost_source: str = \"active\"" in src_train
assert "_sparse_mc_pid_cost_from_replay" in src_train
assert "true_action_sparse_cost_spread" in src_train
assert "true_action_dense_cost_spread" in src_train
assert "PID_COST_SOURCE_OVERRIDE" in src_relxy
assert "--pid-cost-source" in src_relxy
print("Sparse depth static guard PASSED.")
PYCHECK

export SEED_OVERRIDE="$SEED"
export EPOCHS_OVERRIDE=200
export STEPS_PER_EPOCH_OVERRIDE=7
export SGD_STEPS_OVERRIDE=4
export NUM_BLOCKS_OVERRIDE="$NUM_BLOCK"
export NU_C_OVERRIDE=0.0003
export ENTROPY_PARAM_OVERRIDE=0.5
export COST_MODE_OVERRIDE=sparse
export PID_COST_SOURCE_OVERRIDE=active
export COST_LIMIT_OVERRIDE=0.0001
export PID_KP_OVERRIDE=5.0
export PID_KI_OVERRIDE=0.1
export PID_KD_OVERRIDE=0.0
export PID_INTEGRAL_DECAY_OVERRIDE=1.0
export PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true
export EVAL_COUNTERFACTUAL_ACTION_PROBES_OVERRIDE=true
export COUNTERFACTUAL_PROBE_RANDOM_ACTIONS_OVERRIDE=8
export COUNTERFACTUAL_PROBE_PERTURB_ACTIONS_OVERRIDE=8
export COUNTERFACTUAL_PROBE_INTERVAL_OVERRIDE=200
export SLURM_ARRAY_TASK_ID=3

bash slurm/relative_xy_sweep.sh
