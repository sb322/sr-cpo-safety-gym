from __future__ import annotations

from pathlib import Path


def test_train_full_uses_tyro_and_train_config() -> None:
    source = Path("scripts/train_full.py").read_text()

    assert "tyro.cli(TrainConfig)" in source
    assert "run_training(config)" in source
