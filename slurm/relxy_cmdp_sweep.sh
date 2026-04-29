#!/bin/bash -l
#SBATCH --job-name=sr_cpo_rcmdp
#SBATCH --output=safe_relxy_cmdp.%A_%a.out
#SBATCH --error=safe_relxy_cmdp.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3

set -euo pipefail

WORKDIR="/mmfs1/home/sb3222/projects/sr-cpo-safety-gym"
cd "$WORKDIR" || exit 1

CMDP_LABELS=("pid_off" "cmdp_nuc1e-2" "cmdp_nuc3e-3" "cmdp_nuc1e-3")
NU_C_VALUES=("0.0003" "0.01" "0.003" "0.001")
PID_KP_VALUES=("0.0" "5.0" "5.0" "5.0")
PID_KI_VALUES=("0.0" "0.1" "0.1" "0.1")
PID_KD_VALUES=("0.0" "0.0" "0.0" "0.0")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CMDP_LABEL="${CMDP_LABELS[$TASK_ID]}"

echo "===== RELATIVE-XY CMDP SWEEP ====="
echo "CMDP_LABEL=$CMDP_LABEL"
echo "OUTER_ARRAY_TASK_ID=$TASK_ID"
echo "INNER_RELATIVE_XY_TASK_ID=3"
echo "SGD_STEPS_OVERRIDE=64"
echo "NU_C_OVERRIDE=${NU_C_VALUES[$TASK_ID]}"
echo "PID_KP_OVERRIDE=${PID_KP_VALUES[$TASK_ID]}"
echo "PID_KI_OVERRIDE=${PID_KI_VALUES[$TASK_ID]}"
echo "PID_KD_OVERRIDE=${PID_KD_VALUES[$TASK_ID]}"
echo ""

export SGD_STEPS_OVERRIDE=64
export NU_C_OVERRIDE="${NU_C_VALUES[$TASK_ID]}"
export PID_KP_OVERRIDE="${PID_KP_VALUES[$TASK_ID]}"
export PID_KI_OVERRIDE="${PID_KI_VALUES[$TASK_ID]}"
export PID_KD_OVERRIDE="${PID_KD_VALUES[$TASK_ID]}"
export SLURM_ARRAY_TASK_ID=3

bash slurm/relative_xy_sweep.sh
