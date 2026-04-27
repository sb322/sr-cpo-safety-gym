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


def test_slurm_static_diff_heredocs_pass_locally() -> None:
    for script in (
        "slurm/smoke.sh",
        "slurm/full.sh",
        "slurm/diagnostic_cost_limit.sh",
    ):
        block = _static_check_block(script)

        exec(block, {})
