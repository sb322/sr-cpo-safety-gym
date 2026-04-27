#!/bin/bash -l
#SBATCH --job-name=sr_cpo_cl_sweep
#SBATCH --output=safe_diag_cl.%A_%a.out
#SBATCH --error=safe_diag_cl.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2

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

# LD_LIBRARY_PATH: pip nvidia packages FIRST, system CUDA as fallback.
# This sequence matters — system CUDA libs can shadow pip libs and produce
# subtle CUDA mismatch errors at JAX init time.
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
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

COST_LIMITS=("0.001" "0.0005" "0.0001")
COST_LABELS=("cl001" "cl0005" "cl0001")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
COST_LIMIT="${COST_LIMITS[$TASK_ID]}"
COST_LABEL="${COST_LABELS[$TASK_ID]}"

echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "COST_LABEL=$COST_LABEL"
echo "COST_LIMIT=$COST_LIMIT"
set +o pipefail
nvidia-smi 2>&1 | head -20
set -o pipefail

echo ""
echo "===== STATIC DIFF + PROBE VERIFICATION ====="
"$PYTHON" - <<'PYCHECK'
import pathlib
src_losses = pathlib.Path("src/sr_cpo/losses.py").read_text()
src_train  = pathlib.Path("src/sr_cpo/train.py").read_text()

# Algorithm patterns this run depends on. If you change the algorithm,
# update these asserts in the SAME commit. Stale asserts here are exactly
# the kind of bug this gate is designed to catch.

# 1. Autograd-safe row-L2 (epsilon INSIDE sqrt, in BOTH critic and actor)
assert "def row_l2_normalize" in src_losses, \
    "row-L2 helper missing from losses.py"
assert "jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)" in src_losses, \
    "autograd-safe row-L2 (sqrt-with-eps-inside) not present in losses.py"
assert src_losses.count("row_l2_normalize(") >= 5, \
    "row-L2 helper not used across critic and actor paths"
assert "jnp.linalg.norm(" not in src_losses or " + 1e-8" not in src_losses, \
    "stale unsafe row-L2 form detected in losses.py"

# 2. Probes wired through (forward NaN, grad/param, actor-component)
for marker in [
    "nan_obs_critic", "nan_sa_critic", "nan_g_critic", "nan_logits_critic",
    "alpha_logprob_actor", "sat_correction_actor", "log_std_mean_actor",
    "c_grad_nan", "a_grad_nan", "cc_grad_nan",
]:
    assert marker in src_losses or marker in src_train, f"probe missing — {marker}"

# 3. Phase-1g alpha cap
assert "alpha_max" in src_train, "alpha-cap (Args.alpha_max) missing from train.py"
assert "log_alpha_cap" in src_train, "alpha-cap clip not present in train.py sgd_step"

print("Static diff + probe verification PASSED.")
PYCHECK

if [ $? -ne 0 ]; then
    echo "Verification failed — aborting job."
    exit 1
fi

echo ""
echo "===== JAX GPU CHECK ====="
"$PYTHON" - <<'PYCHECK'
import jax
print(f"devices = {jax.devices()}")
print(f"default_backend = {jax.default_backend()}")
if jax.default_backend() != "gpu":
    print("FATAL: JAX did not detect GPU. Aborting.")
    import sys; sys.exit(1)
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== ENV PREFLIGHT ====="
"$PYTHON" - <<'PYCHECK'
# Build the actual env once, run reset, print obs shape and a sample
# of obs/cost. Catches any env-construction bugs before they show up
# 30 seconds into training.
from sr_cpo.env_wrappers import make_safe_learning_go_to_goal
import jax
adapter = make_safe_learning_go_to_goal(num_envs=4, episode_length=1000)
state, transition = adapter.reset(jax.random.PRNGKey(0))
print(f"obs.shape = {transition.observation.shape}")
print(f"cost = {transition.extras['cost']}")
print(f"obs sample = {transition.observation[0, :3]}")
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== TRAINING ====="
"$PYTHON" scripts/train_full.py \
    --seed 0 \
    --epochs 50 \
    --steps-per-epoch 7 \
    --num-envs 256 \
    --unroll-length 62 \
    --use-real-env \
    --env-episode-length 1000 \
    --prefill-steps 2 \
    --sgd-steps 1 \
    --batch-size 1024 \
    --buffer-capacity 8192 \
    --observation-dim 55 \
    --action-dim 2 \
    --goal-dim 55 \
    --width 256 \
    --num-blocks 4 \
    --latent-dim 64 \
    --learning-rate 3e-4 \
    --grad-clip-norm 10.0 \
    --tau 0.1 \
    --rho 0.1 \
    --gamma-c 0.99 \
    --target-update-rate 0.005 \
    --nu-f 1.0 \
    --nu-c 1.0 \
    --alpha-max 1.0 \
    --cost-limit "$COST_LIMIT" \
    --pid-kp 5.0 \
    --pid-ki 0.1 \
    --pid-kd 0.0 \
    --pid-integral-min -10.0 \
    --pid-integral-max 10.0
