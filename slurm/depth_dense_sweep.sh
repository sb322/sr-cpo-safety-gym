#!/bin/bash -l
#SBATCH --job-name=sr_cpo_ddense
#SBATCH --output=safe_depth_dense.%A_%a.out
#SBATCH --error=safe_depth_dense.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-11

set -euo pipefail

WORKDIR="/mmfs1/home/sb3222/projects/sr-cpo-safety-gym"
cd "$WORKDIR" || exit 1

# Hand-launched dense depth sweep; this PR only lands the script.
DEPTH_LABELS=("dense_depth4" "dense_depth8" "dense_depth16" "dense_depth32")
NUM_BLOCK_VALUES=("4" "8" "16" "32")
SEED_VALUES=("0" "1" "2")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DEPTH_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))
DEPTH_LABEL="${DEPTH_LABELS[$DEPTH_INDEX]}"
NUM_BLOCK="${NUM_BLOCK_VALUES[$DEPTH_INDEX]}"
SEED="${SEED_VALUES[$SEED_INDEX]}"

echo "===== DENSE DEPTH SWEEP ====="
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
echo "COST_MODE_OVERRIDE=dense_proximity"
echo "COST_DENSE_PROX_TAU_OVERRIDE=0.5"
echo "PID_COST_SOURCE_OVERRIDE=sparse"
echo "COST_LIMIT_OVERRIDE=0.0001"
echo "PID_KP_OVERRIDE=5.0"
echo "PID_KI_OVERRIDE=0.1"
echo "PID_KD_OVERRIDE=0.0"
echo "PID_INTEGRAL_DECAY_OVERRIDE=1.0"
echo "PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true"
echo "EVAL_COUNTERFACTUAL_ACTION_PROBES_OVERRIDE=true"
echo ""

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
print("Dense depth static guard PASSED.")
PYCHECK

export SEED_OVERRIDE="$SEED"
export EPOCHS_OVERRIDE=200
export STEPS_PER_EPOCH_OVERRIDE=7
export SGD_STEPS_OVERRIDE=4
export NUM_BLOCKS_OVERRIDE="$NUM_BLOCK"
export NU_C_OVERRIDE=0.0003
export ENTROPY_PARAM_OVERRIDE=0.5
export COST_MODE_OVERRIDE=dense_proximity
export COST_DENSE_PROX_TAU_OVERRIDE=0.5
export PID_COST_SOURCE_OVERRIDE=sparse
export COST_LIMIT_OVERRIDE=0.0001
export PID_KP_OVERRIDE=5.0
export PID_KI_OVERRIDE=0.1
export PID_KD_OVERRIDE=0.0
export PID_INTEGRAL_DECAY_OVERRIDE=1.0
export PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true
export EVAL_COUNTERFACTUAL_ACTION_PROBES_OVERRIDE=true
export SLURM_ARRAY_TASK_ID=3

bash slurm/relative_xy_sweep.sh
