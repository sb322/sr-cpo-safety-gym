#!/bin/bash -l
#SBATCH --job-name=sr_cpo_gmask
#SBATCH --output=safe_goal_mask.%A_%a.out
#SBATCH --error=safe_goal_mask.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1

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
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

MASK_LABELS=("lidar_d8_unmasked" "lidar_d8_masked")
MASK_GOAL_FLAGS=("false" "true")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MASK_LABEL="${MASK_LABELS[$TASK_ID]}"
MASK_GOAL_IN_STATE="${MASK_GOAL_FLAGS[$TASK_ID]}"
NUM_BLOCK="8"
GOAL_START="16"
GOAL_DIM="16"
SEED="0"
SGD_STEPS="4"
NU_C="0.0003"
COST_LIMIT="0.0001"
PID_KP="5.0"
PID_KI="0.1"
PID_KD="0.0"

MASK_ARGS=()
if [ "$MASK_GOAL_IN_STATE" = "true" ]; then
    MASK_ARGS+=(--mask-goal-in-state)
fi

echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "MASK_LABEL=$MASK_LABEL"
echo "NUM_BLOCKS=$NUM_BLOCK"
echo "USE_RESIDUAL=true"
echo "SGD_STEPS=$SGD_STEPS"
echo "GOAL_START=$GOAL_START"
echo "GOAL_DIM=$GOAL_DIM"
echo "MASK_GOAL_IN_STATE=$MASK_GOAL_IN_STATE"
echo "NU_C=$NU_C"
echo "SEED=$SEED"
echo "COST_LIMIT=$COST_LIMIT"
echo "PID_KP=$PID_KP"
echo "PID_KI=$PID_KI"
echo "PID_KD=$PID_KD"
set +o pipefail
nvidia-smi 2>&1 | head -20
set -o pipefail

echo ""
echo "===== STATIC DIFF + PROBE VERIFICATION ====="
"$PYTHON" - <<'PYCHECK'
import pathlib
src_losses = pathlib.Path("src/sr_cpo/losses.py").read_text()
src_train  = pathlib.Path("src/sr_cpo/train.py").read_text()
src_replay = pathlib.Path("src/sr_cpo/replay_buffer.py").read_text()
src_networks = pathlib.Path("src/sr_cpo/networks.py").read_text()

assert "def row_l2_normalize" in src_losses, \
    "row-L2 helper missing from losses.py"
assert "jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)" in src_losses, \
    "autograd-safe row-L2 (sqrt-with-eps-inside) not present in losses.py"
assert src_losses.count("row_l2_normalize(") >= 5, \
    "row-L2 helper not used across critic and actor paths"

for marker in [
    "goal_dist", "goal_reached", "gdist=", "reached=",
    "goal_slice_mean", "goal_slice_std", "goal_slice_min", "goal_slice_max",
    "gstart=", "gdim=", "gmean=", "gstd=", "gmin=", "gmax=",
    "mask_goal_in_state", "state_goal_masked", "gmask=",
    "lambda_qc_actor", "nu_c",
]:
    assert marker in src_losses or marker in src_train, f"probe missing - {marker}"

assert "goal_start: int = 0" in src_train, "goal-slice start missing from TrainConfig"
assert "goal_start=config.goal_start" in src_train, \
    "hindsight goal sampling does not use TrainConfig.goal_start"
assert "_goal_from_obs" in src_replay, \
    "hindsight goals do not use the shared goal-space path"
assert "_mask_goal_in_state(env_state.obs, config)" in src_train, \
    "actor rollout does not use masked state inputs"
assert "_mask_transition_state_inputs(batch, config)" in src_train, \
    "critic batches do not use masked state inputs"
assert "use_residual: bool = False" in src_train, \
    "TrainConfig residual switch missing"
assert "WangResidualBlock" in src_networks and "use_residual" in src_networks, \
    "residual tower path missing from networks.py"

print("Static diff + probe verification PASSED.")
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== JAX GPU CHECK ====="
"$PYTHON" - <<'PYCHECK'
import jax
print(f"devices = {jax.devices()}")
print(f"default_backend = {jax.default_backend()}")
assert jax.default_backend() == "gpu"
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== ENV PREFLIGHT ====="
"$PYTHON" - <<'PYCHECK'
from sr_cpo.env_wrappers import make_safe_learning_go_to_goal
import jax
adapter = make_safe_learning_go_to_goal(num_envs=4, episode_length=1000)
state, transition = adapter.reset(jax.random.PRNGKey(0))
print(f"obs.shape = {transition.observation.shape}")
print(f"cost = {transition.extras['cost']}")
print(f"goal_dist = {transition.extras['goal_dist']}")
print(f"goal_reached = {transition.extras['goal_reached']}")
print(f"goal lidar sample = {transition.observation[0, 16:32]}")
PYCHECK
[ $? -ne 0 ] && exit 1

echo ""
echo "===== TRAINING ====="
"$PYTHON" scripts/train_full.py \
    --seed "$SEED" \
    --epochs 50 \
    --steps-per-epoch 7 \
    --num-envs 256 \
    --unroll-length 62 \
    --use-real-env \
    --env-episode-length 1000 \
    --prefill-steps 2 \
    --sgd-steps "$SGD_STEPS" \
    --batch-size 1024 \
    --buffer-capacity 8192 \
    --observation-dim 55 \
    --action-dim 2 \
    --goal-start "$GOAL_START" \
    --goal-dim "$GOAL_DIM" \
    "${MASK_ARGS[@]}" \
    --width 256 \
    --num-blocks "$NUM_BLOCK" \
    --use-residual \
    --latent-dim 64 \
    --learning-rate 3e-4 \
    --grad-clip-norm 10.0 \
    --tau 0.1 \
    --rho 0.1 \
    --gamma-c 0.99 \
    --target-update-rate 0.005 \
    --nu-f 1.0 \
    --nu-c "$NU_C" \
    --alpha-max 1.0 \
    --cost-limit "$COST_LIMIT" \
    --pid-kp "$PID_KP" \
    --pid-ki "$PID_KI" \
    --pid-kd "$PID_KD" \
    --pid-integral-min -10.0 \
    --pid-integral-max 10.0
