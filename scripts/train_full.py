"""Production SR-CPO training entrypoint."""

from __future__ import annotations

import tyro

from sr_cpo.train import TrainConfig, run_training


def main() -> None:
    config = tyro.cli(TrainConfig)
    run_training(config)


if __name__ == "__main__":
    main()
