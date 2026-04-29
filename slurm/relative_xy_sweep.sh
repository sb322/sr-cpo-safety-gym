#!/bin/bash -l
#SBATCH --job-name=sr_cpo_relxy
#SBATCH --output=safe_relative_xy.%A_%a.out
#SBATCH --error=safe_relative_xy.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

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

REL_LABELS=("xy_abs_d8" "relxy_d8" "relxy_lidar_mask_d8" "relxy_lidar_mask_l2_d8")
GOAL_MODES=("xy" "relative_xy" "relative_xy" "relative_xy")
LIDAR_MASK_FLAGS=("false" "false" "true" "true")
SCORE_MODES=("cosine" "cosine" "cosine" "l2")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
REL_LABEL="${REL_LABELS[$TASK_ID]}"
GOAL_MODE="${GOAL_MODES[$TASK_ID]}"
MASK_NATIVE_GOAL_LIDAR="${LIDAR_MASK_FLAGS[$TASK_ID]}"
SCORE_MODE="${SCORE_MODES[$TASK_ID]}"
NUM_BLOCK="8"
GOAL_START="55"
GOAL_DIM="2"
SEED="0"
SGD_STEPS="4"
NU_C="0.0003"
COST_LIMIT="0.0001"
PID_KP="5.0"
PID_KI="0.1"
PID_KD="0.0"

MASK_ARGS=()
if [ "$MASK_NATIVE_GOAL_LIDAR" = "true" ]; then
    MASK_ARGS+=(--mask-native-goal-lidar)
fi

echo "===== ENVIRONMENT ====="
echo "HOST=$(hostname)"
echo "PYTHON=$("$PYTHON" --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "REL_LABEL=$REL_LABEL"
echo "GOAL_MODE=$GOAL_MODE"
echo "MASK_NATIVE_GOAL_LIDAR=$MASK_NATIVE_GOAL_LIDAR"
echo "CRITIC_SCORE_MODE=$SCORE_MODE"
echo "NUM_BLOCKS=$NUM_BLOCK"
echo "USE_RESIDUAL=true"
echo "SGD_STEPS=$SGD_STEPS"
echo "GOAL_START=$GOAL_START"
echo "GOAL_DIM=$GOAL_DIM"
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
src_env = pathlib.Path("src/sr_cpo/env_wrappers.py").read_text()
src_losses = pathlib.Path("src/sr_cpo/losses.py").read_text()
src_train  = pathlib.Path("src/sr_cpo/train.py").read_text()
src_replay = pathlib.Path("src/sr_cpo/replay_buffer.py").read_text()
src_networks = pathlib.Path("src/sr_cpo/networks.py").read_text()

assert "def row_l2_normalize" in src_losses, \
    "row-L2 helper missing from losses.py"
assert "jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)" in src_losses, \
    "autograd-safe row-L2 missing"
assert "goal_mode: str = \"obs_slice\"" in src_train, \
    "TrainConfig goal_mode missing"
assert "goal_mode=config.goal_mode" in src_train, \
    "real env adapter does not receive TrainConfig.goal_mode"
assert "\"relative_xy\"" in src_env and "\"relative_xy\"" in src_train, \
    "relative_xy goal mode missing"
assert "target_xy - self.achieved_goal(state)" in src_env, \
    "relative_xy actor goal is not target_xy - current_robot_xy"
assert "mask_native_goal_lidar" in src_env and "mask_native_goal_lidar" in src_train, \
    "native target-lidar mask switch missing"
assert "_GOAL_LIDAR_START:_GOAL_LIDAR_END" in src_env, \
    "native target-lidar slice is not masked in env adapter"
assert "critic_score_mode: str = \"cosine\"" in src_train, \
    "critic score-mode switch missing"
assert "score_mode=config.critic_score_mode" in src_train, \
    "critic/actor losses do not receive the score-mode switch"
assert 'score_mode == "l2"' in src_losses and "-jnp.sqrt(sq_dist" in src_losses, \
    "reference-style negative-L2 score mode missing"
assert "relative_goal=config.goal_mode == \"relative_xy\"" in src_train, \
    "hindsight critic goals are not switched to relative mode"
assert "goal = goal - _goal_from_obs(obs" in src_replay, \
    "relative hindsight goals do not subtract current achieved XY"
assert "goal_mode_relative" in src_train and "grel=" in src_train and "glmask=" in src_train \
    and "score_l2=" in src_train, "relative/native-mask/score logging missing"
assert "def desired_goal" in src_env and "def achieved_goal" in src_env, \
    "xy desired/achieved goal accessors missing"
assert "_state_observation" in src_env and "self.achieved_goal(state)" in src_env, \
    "state observation path does not append achieved robot XY"
assert "desired_goal" in src_train and "transitions.extras[\"desired_goal\"]" in src_train, \
    "actor rollout metrics do not use adapter desired goals"
assert "goal_start=config.goal_start" in src_train and "_goal_from_obs" in src_replay, \
    "hindsight critic goals do not use configured future achieved-goal slice"
assert "goal_mode_xy" in src_train and "gxy=" in src_train, \
    "xy goal-mode logging missing"
assert "WangResidualBlock" in src_networks and "use_residual" in src_networks, \
    "residual tower path missing"

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
"$PYTHON" - "$GOAL_MODE" "$MASK_NATIVE_GOAL_LIDAR" <<'PYCHECK'
import sys
import jax
from sr_cpo.env_wrappers import make_safe_learning_go_to_goal

goal_mode = sys.argv[1]
mask_native_goal_lidar = sys.argv[2] == "true"
adapter = make_safe_learning_go_to_goal(
    num_envs=4,
    episode_length=1000,
    goal_mode=goal_mode,
    mask_native_goal_lidar=mask_native_goal_lidar,
)
state, transition = adapter.reset(jax.random.PRNGKey(0))
print(f"obs.shape = {transition.observation.shape}")
print(f"cost = {transition.extras['cost']}")
print(f"goal_dist = {transition.extras['goal_dist']}")
print(f"goal_reached = {transition.extras['goal_reached']}")
print(f"desired_goal = {transition.extras['desired_goal']}")
print(f"achieved_goal = {transition.extras['achieved_goal']}")
print(f"goal_lidar = {transition.observation[:, 16:32]}")
print(f"robot_xy_from_obs = {transition.observation[:, -2:]}")
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
    --goal-mode "$GOAL_MODE" \
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
    --critic-score-mode "$SCORE_MODE" \
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
