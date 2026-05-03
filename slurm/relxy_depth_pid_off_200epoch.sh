#!/bin/bash -l
#SBATCH --job-name=sr_cpo_dpid
#SBATCH --output=safe_relxy_depth_pidoff.%A_%a.out
#SBATCH --error=safe_relxy_depth_pidoff.%A_%a.err
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

DEPTH_LABELS=(
    "pidoff_d2_seed0"
    "pidoff_d2_seed1"
    "pidoff_d2_seed2"
    "pidoff_d4_seed0"
    "pidoff_d4_seed1"
    "pidoff_d4_seed2"
    "pidoff_d8_seed0"
    "pidoff_d8_seed1"
    "pidoff_d8_seed2"
    "pidoff_d16_seed0"
    "pidoff_d16_seed1"
    "pidoff_d16_seed2"
)
DEPTH_VALUES=("2" "2" "2" "4" "4" "4" "8" "8" "8" "16" "16" "16")
SEED_VALUES=("0" "1" "2" "0" "1" "2" "0" "1" "2" "0" "1" "2")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DEPTH_LABEL="${DEPTH_LABELS[$TASK_ID]}"

echo "===== RELATIVE-XY PID-OFF 200-EPOCH DEPTH SWEEP ====="
echo "DEPTH_LABEL=$DEPTH_LABEL"
echo "OUTER_ARRAY_TASK_ID=$TASK_ID"
echo "INNER_RELATIVE_XY_TASK_ID=3"
echo "SEED_OVERRIDE=${SEED_VALUES[$TASK_ID]}"
echo "NUM_BLOCKS_OVERRIDE=${DEPTH_VALUES[$TASK_ID]}"
echo "EPOCHS_OVERRIDE=200"
echo "STEPS_PER_EPOCH_OVERRIDE=7"
echo "SGD_STEPS_OVERRIDE=64"
echo "NU_C_OVERRIDE=0.0003"
echo "PID_KP_OVERRIDE=0.0"
echo "PID_KI_OVERRIDE=0.0"
echo "PID_KD_OVERRIDE=0.0"
echo "PID_INTEGRAL_DECAY_OVERRIDE=1.0"
echo "COST_RETURN_LOSS_WEIGHT_OVERRIDE=0.0"
echo "PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true"
echo "EVAL_FREEZE_GOAL_AFTER_SUCCESS_OVERRIDE=true"
echo "EVAL_ACTION_STD_SCALES_OVERRIDE=0.0"
echo ""

export SEED_OVERRIDE="${SEED_VALUES[$TASK_ID]}"
export NUM_BLOCKS_OVERRIDE="${DEPTH_VALUES[$TASK_ID]}"
export EPOCHS_OVERRIDE=200
export STEPS_PER_EPOCH_OVERRIDE=7
export SGD_STEPS_OVERRIDE=64
export NU_C_OVERRIDE=0.0003
export PID_KP_OVERRIDE=0.0
export PID_KI_OVERRIDE=0.0
export PID_KD_OVERRIDE=0.0
export PID_INTEGRAL_DECAY_OVERRIDE=1.0
export COST_RETURN_LOSS_WEIGHT_OVERRIDE=0.0
export PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true
export EVAL_FREEZE_GOAL_AFTER_SUCCESS_OVERRIDE=true
export EVAL_ACTION_STD_SCALES_OVERRIDE=0.0
export SLURM_ARRAY_TASK_ID=3

bash slurm/relative_xy_sweep.sh
