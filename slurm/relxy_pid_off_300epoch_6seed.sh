#!/bin/bash -l
#SBATCH --job-name=sr_cpo_p300
#SBATCH --output=safe_relxy_pid300.%A_%a.out
#SBATCH --error=safe_relxy_pid300.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5

set -euo pipefail

WORKDIR="/mmfs1/home/sb3222/projects/sr-cpo-safety-gym"
cd "$WORKDIR" || exit 1

RUN300_LABELS=(
    "pid_off_300_seed0"
    "pid_off_300_seed1"
    "pid_off_300_seed2"
    "pid_off_300_seed3"
    "pid_off_300_seed4"
    "pid_off_300_seed5"
)
SEED_VALUES=("0" "1" "2" "3" "4" "5")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN300_LABEL="${RUN300_LABELS[$TASK_ID]}"

echo "===== RELATIVE-XY PID-OFF 300-EPOCH 6-SEED ====="
echo "RUN300_LABEL=$RUN300_LABEL"
echo "OUTER_ARRAY_TASK_ID=$TASK_ID"
echo "INNER_RELATIVE_XY_TASK_ID=3"
echo "SEED_OVERRIDE=${SEED_VALUES[$TASK_ID]}"
echo "EPOCHS_OVERRIDE=300"
echo "STEPS_PER_EPOCH_OVERRIDE=7"
echo "SGD_STEPS_OVERRIDE=64"
echo "NU_C_OVERRIDE=0.0003"
echo "PID_KP_OVERRIDE=0.0"
echo "PID_KI_OVERRIDE=0.0"
echo "PID_KD_OVERRIDE=0.0"
echo "PID_INTEGRAL_DECAY_OVERRIDE=1.0"
echo "COST_RETURN_LOSS_WEIGHT_OVERRIDE=0.0"
echo "PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true"
echo ""

export SEED_OVERRIDE="${SEED_VALUES[$TASK_ID]}"
export EPOCHS_OVERRIDE=300
export STEPS_PER_EPOCH_OVERRIDE=7
export SGD_STEPS_OVERRIDE=64
export NU_C_OVERRIDE=0.0003
export PID_KP_OVERRIDE=0.0
export PID_KI_OVERRIDE=0.0
export PID_KD_OVERRIDE=0.0
export PID_INTEGRAL_DECAY_OVERRIDE=1.0
export COST_RETURN_LOSS_WEIGHT_OVERRIDE=0.0
export PROBE_COUNTERFACTUAL_COSTS_OVERRIDE=true
export SLURM_ARRAY_TASK_ID=3

bash slurm/relative_xy_sweep.sh
