from __future__ import annotations

import re
from pathlib import Path


def _static_check_block(script: str) -> str:
    source = Path(script).read_text()
    blocks = re.findall(r"<<'PYCHECK'\n(.*?)\nPYCHECK\n", source, flags=re.DOTALL)
    assert blocks
    return blocks[0]


def test_slurm_scripts_have_wulver_header_and_static_gate() -> None:
    for script in (
        "slurm/smoke.sh",
        "slurm/full.sh",
        "slurm/diagnostic_cost_limit.sh",
        "slurm/baseline_cl0001_3seed.sh",
        "slurm/pid_gain_sweep.sh",
        "slurm/depth_sweep_residual.sh",
        "slurm/depth8_sgd_sweep.sh",
        "slurm/depth_sgd4_residual.sh",
        "slurm/constraint_activation_sweep.sh",
    ):
        source = Path(script).read_text()

        assert "#SBATCH --account=ag2682" in source
        assert "#SBATCH --partition=gpu" in source
        assert "#SBATCH --qos=standard" in source
        assert "#SBATCH --gres=gpu:a100_40g:1" in source
        assert "STATIC DIFF + PROBE VERIFICATION" in source
        assert "JAX GPU CHECK" in source
        assert "ENV PREFLIGHT" in source
        assert "--use-real-env" in source
        assert "--observation-dim 55" in source
        assert "--goal-dim 55" in source
        assert "--grad-clip-norm 10.0" in source
        assert "set +o pipefail\nnvidia-smi 2>&1 | head -20\nset -o pipefail" in source


def test_cost_limit_diagnostic_script_is_array_sweep() -> None:
    source = Path("slurm/diagnostic_cost_limit.sh").read_text()

    assert "#SBATCH --array=0-2" in source
    assert "safe_diag_cl.%A_%a.out" in source
    assert 'COST_LIMITS=("0.001" "0.0005" "0.0001")' in source
    assert 'COST_LABELS=("cl001" "cl0005" "cl0001")' in source
    assert '--cost-limit "$COST_LIMIT"' in source


def test_cl0001_baseline_script_is_three_seed_array() -> None:
    source = Path("slurm/baseline_cl0001_3seed.sh").read_text()

    assert "#SBATCH --array=0-2" in source
    assert "safe_base_cl0001.%A_%a.out" in source
    assert 'SEEDS=("0" "1" "2")' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert '--seed "$SEED"' in source
    assert '--cost-limit "$COST_LIMIT"' in source


def test_pid_gain_sweep_script_is_three_gain_array() -> None:
    source = Path("slurm/pid_gain_sweep.sh").read_text()

    assert "#SBATCH --array=0-2" in source
    assert "safe_pid.%A_%a.out" in source
    assert 'PID_LABELS=("pid_soft" "pid_base" "pid_strong")' in source
    assert 'PID_KPS=("2.5" "5.0" "10.0")' in source
    assert 'PID_KIS=("0.05" "0.1" "0.2")' in source
    assert 'PID_KD="0.0"' in source
    assert 'SEED="0"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert '--pid-kp "$PID_KP"' in source
    assert '--pid-ki "$PID_KI"' in source
    assert '--pid-kd "$PID_KD"' in source


def test_depth_sweep_script_is_residual_depth_array() -> None:
    source = Path("slurm/depth_sweep_residual.sh").read_text()

    assert "#SBATCH --array=0-2" in source
    assert "safe_depth.%A_%a.out" in source
    assert 'DEPTH_LABELS=("depth4" "depth8" "depth16")' in source
    assert 'NUM_BLOCKS=("4" "8" "16")' in source
    assert 'SEED="0"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert "--use-residual" in source
    assert '--num-blocks "$NUM_BLOCK"' in source
    assert 'PID_KP="5.0"' in source
    assert 'PID_KI="0.1"' in source
    assert 'PID_KD="0.0"' in source


def test_depth8_sgd_sweep_script_is_residual_update_array() -> None:
    source = Path("slurm/depth8_sgd_sweep.sh").read_text()

    assert "#SBATCH --array=0-2" in source
    assert "safe_depth8_sgd.%A_%a.out" in source
    assert 'SGD_LABELS=("sgd1" "sgd2" "sgd4")' in source
    assert 'SGD_STEPS=("1" "2" "4")' in source
    assert 'NUM_BLOCKS="8"' in source
    assert "--use-residual" in source
    assert '--sgd-steps "$SGD_STEP"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert 'PID_KP="5.0"' in source
    assert 'PID_KI="0.1"' in source
    assert 'PID_KD="0.0"' in source


def test_depth_sgd4_script_is_residual_depth_array() -> None:
    source = Path("slurm/depth_sgd4_residual.sh").read_text()

    assert "#SBATCH --array=0-3" in source
    assert "safe_depth_sgd4.%A_%a.out" in source
    assert (
        'DEPTH_LABELS=("depth4_sgd4" "depth8_sgd4" '
        '"depth16_sgd4" "depth32_sgd4")'
    ) in source
    assert 'NUM_BLOCKS=("4" "8" "16" "32")' in source
    assert 'SGD_STEPS="4"' in source
    assert 'SEED="0"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert "--use-residual" in source
    assert '--num-blocks "$NUM_BLOCK"' in source
    assert '--sgd-steps "$SGD_STEPS"' in source
    assert 'PID_KP="5.0"' in source
    assert 'PID_KI="0.1"' in source
    assert 'PID_KD="0.0"' in source


def test_constraint_activation_script_is_dual_activation_array() -> None:
    source = Path("slurm/constraint_activation_sweep.sh").read_text()

    assert "#SBATCH --array=0-4" in source
    assert "safe_constraint.%A_%a.out" in source
    assert (
        'CONSTRAINT_LABELS=("unconstrained" "cl1e-4" '
        '"cl5e-5" "cl1e-5" "cl0")'
    ) in source
    assert (
        'COST_LIMITS=("0.0001" "0.0001" '
        '"0.00005" "0.00001" "0.0")'
    ) in source
    assert 'PID_KPS=("0.0" "5.0" "5.0" "5.0" "5.0")' in source
    assert 'PID_KIS=("0.0" "0.1" "0.1" "0.1" "0.1")' in source
    assert 'PID_KD="0.0"' in source
    assert 'USE_RESIDUAL=false' in source
    assert "--use-residual" not in source
    assert "--num-blocks 4" in source
    assert "--sgd-steps 1" in source
    assert '--cost-limit "$COST_LIMIT"' in source
    assert '--pid-kp "$PID_KP"' in source
    assert '--pid-ki "$PID_KI"' in source
    assert 'pid_error", "pid_integral", "pid_raw_lambda"' in source


def test_production_launchers_use_calibrated_cost_limit() -> None:
    for script in ("slurm/smoke.sh", "slurm/full.sh"):
        source = Path(script).read_text()

        assert "--cost-limit 0.0001" in source
        assert "--cost-limit 0.05" not in source


def test_slurm_static_diff_heredocs_pass_locally() -> None:
    for script in (
        "slurm/smoke.sh",
        "slurm/full.sh",
        "slurm/diagnostic_cost_limit.sh",
        "slurm/baseline_cl0001_3seed.sh",
        "slurm/pid_gain_sweep.sh",
        "slurm/depth_sweep_residual.sh",
        "slurm/depth8_sgd_sweep.sh",
        "slurm/depth_sgd4_residual.sh",
        "slurm/constraint_activation_sweep.sh",
    ):
        block = _static_check_block(script)

        exec(block, {})
