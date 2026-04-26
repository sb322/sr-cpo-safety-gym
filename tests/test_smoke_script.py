from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_module() -> object:
    path = Path("scripts/smoke_1epoch.py")
    spec = importlib.util.spec_from_file_location("smoke_1epoch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_uses_one_env_hundred_step_toy_run() -> None:
    module = _load_smoke_module()
    config = module.build_config()

    assert config.epochs == 1
    assert config.num_envs == 1
    total_steps = (
        (config.prefill_steps + config.steps_per_epoch)
        * config.unroll_length
        * config.num_envs
    )
    assert total_steps == 100
    assert config.alpha_max == 1.0
