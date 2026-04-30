#!/bin/bash -l
#SBATCH --job-name=sr_cpo_rpid
#SBATCH --output=safe_relxy_pid.%A_%a.out
#SBATCH --error=safe_relxy_pid.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-5

set -euo pipefail

WORKDIR="/mmfs1/home/sb3222/projects/sr-cpo-safety-gym"
cd "$WORKDIR" || exit 1

PRESSURE_LABELS=(
    "kp10_nuc3e-3"
    "kp25_nuc3e-3"
    "kp50_nuc3e-3"
    "kp10_nuc1e-2"
    "kp25_nuc1e-2"
    "kp50_nuc1e-2"
)
NU_C_VALUES=("0.003" "0.003" "0.003" "0.01" "0.01" "0.01")
PID_KP_VALUES=("10.0" "25.0" "50.0" "10.0" "25.0" "50.0")
PID_KI_VALUES=("0.01" "0.01" "0.01" "0.01" "0.01" "0.01")
PID_KD_VALUES=("0.0" "0.0" "0.0" "0.0" "0.0" "0.0")
PID_INTEGRAL_DECAY_VALUES=("0.95" "0.95" "0.95" "0.95" "0.95" "0.95")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PRESSURE_LABEL="${PRESSURE_LABELS[$TASK_ID]}"

echo "===== RELATIVE-XY PID PRESSURE SWEEP ====="
echo "PRESSURE_LABEL=$PRESSURE_LABEL"
echo "OUTER_ARRAY_TASK_ID=$TASK_ID"
echo "INNER_RELATIVE_XY_TASK_ID=3"
echo "SGD_STEPS_OVERRIDE=64"
echo "NU_C_OVERRIDE=${NU_C_VALUES[$TASK_ID]}"
echo "PID_KP_OVERRIDE=${PID_KP_VALUES[$TASK_ID]}"
echo "PID_KI_OVERRIDE=${PID_KI_VALUES[$TASK_ID]}"
echo "PID_KD_OVERRIDE=${PID_KD_VALUES[$TASK_ID]}"
echo "PID_INTEGRAL_DECAY_OVERRIDE=${PID_INTEGRAL_DECAY_VALUES[$TASK_ID]}"
echo "COST_RETURN_LOSS_WEIGHT_OVERRIDE=1.0"
echo "PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true"
echo ""

export SGD_STEPS_OVERRIDE=64
export NU_C_OVERRIDE="${NU_C_VALUES[$TASK_ID]}"
export PID_KP_OVERRIDE="${PID_KP_VALUES[$TASK_ID]}"
export PID_KI_OVERRIDE="${PID_KI_VALUES[$TASK_ID]}"
export PID_KD_OVERRIDE="${PID_KD_VALUES[$TASK_ID]}"
export PID_INTEGRAL_DECAY_OVERRIDE="${PID_INTEGRAL_DECAY_VALUES[$TASK_ID]}"
export COST_RETURN_LOSS_WEIGHT_OVERRIDE=1.0
export PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true
export SLURM_ARRAY_TASK_ID=3

bash slurm/relative_xy_sweep.sh
