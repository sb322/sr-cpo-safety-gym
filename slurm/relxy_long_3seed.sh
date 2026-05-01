#!/bin/bash -l
#SBATCH --job-name=sr_cpo_rlong
#SBATCH --output=safe_relxy_long.%A_%a.out
#SBATCH --error=safe_relxy_long.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-5

set -euo pipefail

WORKDIR="/mmfs1/home/sb3222/projects/sr-cpo-safety-gym"
cd "$WORKDIR" || exit 1

LONG_LABELS=(
    "pid_off_long_seed0"
    "pid_off_long_seed1"
    "pid_off_long_seed2"
    "cmdp_nuc3e-3_long_seed0"
    "cmdp_nuc3e-3_long_seed1"
    "cmdp_nuc3e-3_long_seed2"
)
SEED_VALUES=("0" "1" "2" "0" "1" "2")
NU_C_VALUES=("0.0003" "0.0003" "0.0003" "0.003" "0.003" "0.003")
PID_KP_VALUES=("0.0" "0.0" "0.0" "5.0" "5.0" "5.0")
PID_KI_VALUES=("0.0" "0.0" "0.0" "0.01" "0.01" "0.01")
PID_KD_VALUES=("0.0" "0.0" "0.0" "0.0" "0.0" "0.0")
PID_INTEGRAL_DECAY_VALUES=("1.0" "1.0" "1.0" "0.95" "0.95" "0.95")
COST_RETURN_LOSS_WEIGHT_VALUES=("0.0" "0.0" "0.0" "1.0" "1.0" "1.0")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LONG_LABEL="${LONG_LABELS[$TASK_ID]}"

echo "===== RELATIVE-XY LONG 3-SEED COMPARISON ====="
echo "LONG_LABEL=$LONG_LABEL"
echo "OUTER_ARRAY_TASK_ID=$TASK_ID"
echo "INNER_RELATIVE_XY_TASK_ID=3"
echo "SEED_OVERRIDE=${SEED_VALUES[$TASK_ID]}"
echo "EPOCHS_OVERRIDE=100"
echo "STEPS_PER_EPOCH_OVERRIDE=7"
echo "SGD_STEPS_OVERRIDE=64"
echo "NU_C_OVERRIDE=${NU_C_VALUES[$TASK_ID]}"
echo "PID_KP_OVERRIDE=${PID_KP_VALUES[$TASK_ID]}"
echo "PID_KI_OVERRIDE=${PID_KI_VALUES[$TASK_ID]}"
echo "PID_KD_OVERRIDE=${PID_KD_VALUES[$TASK_ID]}"
echo "PID_INTEGRAL_DECAY_OVERRIDE=${PID_INTEGRAL_DECAY_VALUES[$TASK_ID]}"
echo "COST_RETURN_LOSS_WEIGHT_OVERRIDE=${COST_RETURN_LOSS_WEIGHT_VALUES[$TASK_ID]}"
echo "PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true"
echo ""

export SEED_OVERRIDE="${SEED_VALUES[$TASK_ID]}"
export EPOCHS_OVERRIDE=100
export STEPS_PER_EPOCH_OVERRIDE=7
export SGD_STEPS_OVERRIDE=64
export NU_C_OVERRIDE="${NU_C_VALUES[$TASK_ID]}"
export PID_KP_OVERRIDE="${PID_KP_VALUES[$TASK_ID]}"
export PID_KI_OVERRIDE="${PID_KI_VALUES[$TASK_ID]}"
export PID_KD_OVERRIDE="${PID_KD_VALUES[$TASK_ID]}"
export PID_INTEGRAL_DECAY_OVERRIDE="${PID_INTEGRAL_DECAY_VALUES[$TASK_ID]}"
export COST_RETURN_LOSS_WEIGHT_OVERRIDE="${COST_RETURN_LOSS_WEIGHT_VALUES[$TASK_ID]}"
export PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true
export SLURM_ARRAY_TASK_ID=3

bash slurm/relative_xy_sweep.sh
