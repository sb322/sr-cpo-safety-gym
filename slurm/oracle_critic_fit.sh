#!/bin/bash -l
#SBATCH --job-name=sr_cpo_oracle
#SBATCH --output=oracle_critic_fit.%j.out
#SBATCH --error=oracle_critic_fit.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

module purge
module load slurm/wulver
module load easybuild
module load CUDA/12.8.0

WORKDIR="/mmfs1/home/sb3222/projects/sr-cpo-safety-gym"
VENV="$WORKDIR/.venv"
PYTHON="$VENV/bin/python"

cd "$WORKDIR" || exit 1

export PATH="$VENV/bin:$PATH"

PY_SITE=$("$PYTHON" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

PIP_NVIDIA_LIBS=""
for subpkg in cudnn cusolver cublas cuda_runtime cuda_nvrtc cufft cuda_cupti cusparse nvjitlink nccl; do
    d="$PY_SITE/nvidia/$subpkg/lib"
    [ -d "$d" ] && PIP_NVIDIA_LIBS="${PIP_NVIDIA_LIBS:+$PIP_NVIDIA_LIBS:}$d"
done

SYS_CUDA_LIB="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0/lib64"
export LD_LIBRARY_PATH="${PIP_NVIDIA_LIBS}:${SYS_CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="/mmfs1/home/sb3222/.cache/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" figures/data

ORACLE_SEED="${ORACLE_SEED_OVERRIDE:-0}"
ORACLE_NUM_STATES="${ORACLE_NUM_STATES_OVERRIDE:-128}"
ORACLE_NUM_CANDIDATES="${ORACLE_NUM_CANDIDATES_OVERRIDE:-16}"
ORACLE_BURNIN_STEPS="${ORACLE_BURNIN_STEPS_OVERRIDE:-200}"
ORACLE_HORIZONS="${ORACLE_HORIZONS_OVERRIDE:-1,5,20}"
ORACLE_FIT_STEPS="${ORACLE_FIT_STEPS_OVERRIDE:-1000}"
ORACLE_FIT_BATCH_SIZE="${ORACLE_FIT_BATCH_SIZE_OVERRIDE:-1024}"
ORACLE_FIT_LABELS="${ORACLE_FIT_LABELS_OVERRIDE:-dense}"
ORACLE_PERTURB_STD="${ORACLE_PERTURB_STD_OVERRIDE:-0.25}"
ORACLE_TAU="${ORACLE_TAU_OVERRIDE:-0.5}"
ORACLE_ACTOR_CHECKPOINT="${ORACLE_ACTOR_CHECKPOINT_OVERRIDE:-}"
ORACLE_STATE_ACTION_SOURCE="${ORACLE_STATE_ACTION_SOURCE_OVERRIDE:-actor}"
ORACLE_OUTPUT="${ORACLE_OUTPUT_OVERRIDE:-figures/data/oracle_critic_fit.${SLURM_JOB_ID}.csv}"
ORACLE_INCLUDE_SYNTHETIC="${ORACLE_INCLUDE_SYNTHETIC_OVERRIDE:-true}"

SYNTH_ARGS=()
if [ "$ORACLE_INCLUDE_SYNTHETIC" = "false" ]; then
    SYNTH_ARGS+=(--no-include-synthetic-action-norm)
fi

echo "===== ORACLE CRITIC FIT ====="
echo "HOST=$(hostname)"
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "ORACLE_SEED=$ORACLE_SEED"
echo "ORACLE_NUM_STATES=$ORACLE_NUM_STATES"
echo "ORACLE_NUM_CANDIDATES=$ORACLE_NUM_CANDIDATES"
echo "ORACLE_BURNIN_STEPS=$ORACLE_BURNIN_STEPS"
echo "ORACLE_HORIZONS=$ORACLE_HORIZONS"
echo "ORACLE_FIT_STEPS=$ORACLE_FIT_STEPS"
echo "ORACLE_FIT_LABELS=$ORACLE_FIT_LABELS"
echo "ORACLE_TAU=$ORACLE_TAU"
echo "ORACLE_OUTPUT=$ORACLE_OUTPUT"

"$PYTHON" scripts/oracle_critic_fit.py \
    --seed "$ORACLE_SEED" \
    --num-states "$ORACLE_NUM_STATES" \
    --num-candidates "$ORACLE_NUM_CANDIDATES" \
    --burnin-steps "$ORACLE_BURNIN_STEPS" \
    --horizons "$ORACLE_HORIZONS" \
    --fit-steps "$ORACLE_FIT_STEPS" \
    --fit-batch-size "$ORACLE_FIT_BATCH_SIZE" \
    --fit-labels "$ORACLE_FIT_LABELS" \
    --perturb-std "$ORACLE_PERTURB_STD" \
    --cost-dense-prox-tau "$ORACLE_TAU" \
    --actor-checkpoint "$ORACLE_ACTOR_CHECKPOINT" \
    --state-action-source "$ORACLE_STATE_ACTION_SOURCE" \
    --output-csv "$ORACLE_OUTPUT" \
    "${SYNTH_ARGS[@]}"
