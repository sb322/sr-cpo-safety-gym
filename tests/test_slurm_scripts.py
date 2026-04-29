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
        "slurm/constraint_pressure_sweep.sh",
        "slurm/constraint_effect_3seed.sh",
        "slurm/depth_sgd4_cmdp.sh",
        "slurm/goal_conditioning_sweep.sh",
        "slurm/goal_mask_sweep.sh",
        "slurm/xy_goal_sweep.sh",
        "slurm/relative_xy_sweep.sh",
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
        if script in (
            "slurm/goal_conditioning_sweep.sh",
            "slurm/goal_mask_sweep.sh",
            "slurm/xy_goal_sweep.sh",
            "slurm/relative_xy_sweep.sh",
        ):
            assert '--goal-start "$GOAL_START"' in source
            assert '--goal-dim "$GOAL_DIM"' in source
        else:
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


def test_constraint_pressure_script_is_nu_c_array() -> None:
    source = Path("slurm/constraint_pressure_sweep.sh").read_text()

    assert "#SBATCH --array=0-4" in source
    assert "safe_pressure.%A_%a.out" in source
    assert (
        'PRESSURE_LABELS=("nuc1" "nuc1e-2" '
        '"nuc1e-3" "nuc3e-4" "nuc1e-4")'
    ) in source
    assert 'NU_C_VALUES=("1.0" "0.01" "0.001" "0.0003" "0.0001")' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert 'PID_KP="5.0"' in source
    assert 'PID_KI="0.1"' in source
    assert 'PID_KD="0.0"' in source
    assert 'USE_RESIDUAL=false' in source
    assert "--use-residual" not in source
    assert "--num-blocks 4" in source
    assert "--sgd-steps 1" in source
    assert '--nu-c "$NU_C"' in source
    assert "lambda_qc_actor" in source


def test_constraint_effect_script_pairs_unconstrained_and_constrained_seeds() -> None:
    source = Path("slurm/constraint_effect_3seed.sh").read_text()

    assert "#SBATCH --array=0-5" in source
    assert "safe_effect.%A_%a.out" in source
    assert (
        '"unconstrained_seed0" "unconstrained_seed1" "unconstrained_seed2"'
    ) in source
    assert '"constrained_seed0" "constrained_seed1" "constrained_seed2"' in source
    assert 'SEEDS=("0" "1" "2" "0" "1" "2")' in source
    assert 'PID_KPS=("0.0" "0.0" "0.0" "5.0" "5.0" "5.0")' in source
    assert 'PID_KIS=("0.0" "0.0" "0.0" "0.1" "0.1" "0.1")' in source
    assert 'NU_C="0.0003"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert 'USE_RESIDUAL=false' in source
    assert "--use-residual" not in source
    assert "--num-blocks 4" in source
    assert "--sgd-steps 1" in source
    assert '--seed "$SEED"' in source
    assert '--nu-c "$NU_C"' in source
    assert '--pid-kp "$PID_KP"' in source
    assert '--pid-ki "$PID_KI"' in source


def test_depth_sgd4_cmdp_script_uses_calibrated_constraint_pressure() -> None:
    source = Path("slurm/depth_sgd4_cmdp.sh").read_text()

    assert "#SBATCH --array=0-3" in source
    assert "safe_depth_cmdp.%A_%a.out" in source
    assert (
        'DEPTH_LABELS=("depth4_cmdp" "depth8_cmdp" '
        '"depth16_cmdp" "depth32_cmdp")'
    ) in source
    assert 'NUM_BLOCKS=("4" "8" "16" "32")' in source
    assert 'SGD_STEPS="4"' in source
    assert 'NU_C="0.0003"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert 'PID_KP="5.0"' in source
    assert 'PID_KI="0.1"' in source
    assert 'PID_KD="0.0"' in source
    assert "--use-residual" in source
    assert '--num-blocks "$NUM_BLOCK"' in source
    assert '--sgd-steps "$SGD_STEPS"' in source
    assert '--nu-c "$NU_C"' in source
    assert "lambda_qc_actor" in source


def test_goal_conditioning_sweep_compares_full_obs_and_goal_lidar() -> None:
    source = Path("slurm/goal_conditioning_sweep.sh").read_text()

    assert "#SBATCH --array=0-3" in source
    assert "safe_goal.%A_%a.out" in source
    assert 'GOAL_LABELS=("full_d8" "lidar_d8" "full_d16" "lidar_d16")' in source
    assert 'NUM_BLOCKS=("8" "8" "16" "16")' in source
    assert 'GOAL_STARTS=("0" "16" "0" "16")' in source
    assert 'GOAL_DIMS=("55" "16" "55" "16")' in source
    assert 'NU_C="0.0003"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert 'PID_KP="5.0"' in source
    assert 'PID_KI="0.1"' in source
    assert 'PID_KD="0.0"' in source
    assert "--use-residual" in source
    assert '--goal-start "$GOAL_START"' in source
    assert '--goal-dim "$GOAL_DIM"' in source
    assert "goal_start=config.goal_start" in source
    assert "_assert_goal_shape" in source
    assert "src_replay" in source
    assert "goal_dist" in source
    assert "goal_reached" in source
    assert "goal_slice_mean" in source
    assert "goal_slice_std" in source
    assert "goal_slice_min" in source
    assert "goal_slice_max" in source
    assert "gstart=" in source
    assert "gdim=" in source
    assert "gmin=" in source
    assert "gmax=" in source


def test_goal_mask_sweep_compares_lidar_state_masking() -> None:
    source = Path("slurm/goal_mask_sweep.sh").read_text()

    assert "#SBATCH --array=0-1" in source
    assert "safe_goal_mask.%A_%a.out" in source
    assert 'MASK_LABELS=("lidar_d8_unmasked" "lidar_d8_masked")' in source
    assert 'MASK_GOAL_FLAGS=("false" "true")' in source
    assert 'GOAL_START="16"' in source
    assert 'GOAL_DIM="16"' in source
    assert 'NU_C="0.0003"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert "--mask-goal-in-state" in source
    assert "mask_goal_in_state" in source
    assert "state_goal_masked" in source
    assert "gmask=" in source
    assert "_mask_goal_in_state(env_state.obs, config)" in source
    assert "_mask_transition_state_inputs(batch, config)" in source


def test_xy_goal_sweep_compares_obs_slice_and_reference_xy_goals() -> None:
    source = Path("slurm/xy_goal_sweep.sh").read_text()

    assert "#SBATCH --array=0-1" in source
    assert "safe_xy_goal.%A_%a.out" in source
    assert 'XY_LABELS=("obs_full_d8" "xy_d8")' in source
    assert 'GOAL_MODES=("obs_slice" "xy")' in source
    assert 'GOAL_STARTS=("0" "55")' in source
    assert 'GOAL_DIMS=("55" "2")' in source
    assert 'NU_C="0.0003"' in source
    assert 'COST_LIMIT="0.0001"' in source
    assert "--use-residual" in source
    assert '--goal-mode "$GOAL_MODE"' in source
    assert '--goal-start "$GOAL_START"' in source
    assert '--goal-dim "$GOAL_DIM"' in source
    assert "desired_goal" in source
    assert "achieved_goal" in source
    assert "goal_mode_xy" in source
    assert "gxy=" in source


def test_relative_xy_sweep_compares_absolute_and_relative_xy_goals() -> None:
    source = Path("slurm/relative_xy_sweep.sh").read_text()

    assert "#SBATCH --array=0-3" in source
    assert "safe_relative_xy.%A_%a.out" in source
    assert (
        'REL_LABELS=("xy_abs_d8" "relxy_d8" "relxy_lidar_mask_d8" '
        '"relxy_lidar_mask_l2_d8")'
    ) in source
    assert 'GOAL_MODES=("xy" "relative_xy" "relative_xy" "relative_xy")' in source
    assert 'LIDAR_MASK_FLAGS=("false" "false" "true" "true")' in source
    assert 'SCORE_MODES=("cosine" "cosine" "cosine" "l2")' in source
    assert 'GOAL_START="55"' in source
    assert 'GOAL_DIM="2"' in source
    assert 'SGD_STEPS="${SGD_STEPS_OVERRIDE:-4}"' in source
    assert 'NU_C="${NU_C_OVERRIDE:-0.0003}"' in source
    assert 'ENTROPY_PARAM="${ENTROPY_PARAM_OVERRIDE:-0.5}"' in source
    assert 'COST_LIMIT="${COST_LIMIT_OVERRIDE:-0.0001}"' in source
    assert 'PID_KP="${PID_KP_OVERRIDE:-5.0}"' in source
    assert 'PID_KI="${PID_KI_OVERRIDE:-0.1}"' in source
    assert 'PID_KD="${PID_KD_OVERRIDE:-0.0}"' in source
    assert "--use-residual" in source
    assert '--goal-mode "$GOAL_MODE"' in source
    assert '--goal-start "$GOAL_START"' in source
    assert '--goal-dim "$GOAL_DIM"' in source
    assert "relative_goal=config.goal_mode" in source
    assert "goal_mode_relative" in source
    assert "grel=" in source
    assert "--mask-native-goal-lidar" in source
    assert "mask_native_goal_lidar" in source
    assert "glmask=" in source
    assert "--critic-score-mode" in source
    assert "score_l2=" in source
    assert '--entropy-param "$ENTROPY_PARAM"' in source


def test_relxy_cmdp_sweep_reuses_reference_relative_xy_arm() -> None:
    source = Path("slurm/relxy_cmdp_sweep.sh").read_text()

    assert "#SBATCH --array=0-3" in source
    assert "safe_relxy_cmdp.%A_%a.out" in source
    assert (
        'CMDP_LABELS=("pid_off" "cmdp_nuc1e-2" '
        '"cmdp_nuc3e-3" "cmdp_nuc1e-3")'
    ) in source
    assert 'NU_C_VALUES=("0.0003" "0.01" "0.003" "0.001")' in source
    assert 'PID_KP_VALUES=("0.0" "5.0" "5.0" "5.0")' in source
    assert 'PID_KI_VALUES=("0.0" "0.1" "0.1" "0.1")' in source
    assert "export SGD_STEPS_OVERRIDE=64" in source
    assert "export SLURM_ARRAY_TASK_ID=3" in source
    assert "bash slurm/relative_xy_sweep.sh" in source


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
        "slurm/constraint_pressure_sweep.sh",
        "slurm/constraint_effect_3seed.sh",
        "slurm/depth_sgd4_cmdp.sh",
        "slurm/goal_conditioning_sweep.sh",
        "slurm/goal_mask_sweep.sh",
        "slurm/xy_goal_sweep.sh",
        "slurm/relative_xy_sweep.sh",
    ):
        block = _static_check_block(script)

        exec(block, {})
