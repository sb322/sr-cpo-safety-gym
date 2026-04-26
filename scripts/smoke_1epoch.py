"""Runs a one-epoch local SR-CPO smoke test."""

from __future__ import annotations

from sr_cpo.train import TrainConfig, run_training


def build_config() -> TrainConfig:
    """Returns the CPU/Mac smoke configuration: 1 env, 100 train steps."""

    return TrainConfig(
        seed=0,
        epochs=1,
        steps_per_epoch=8,
        num_envs=1,
        unroll_length=10,
        prefill_steps=2,
        sgd_steps=1,
        batch_size=16,
        buffer_capacity=32,
        width=32,
        num_blocks=1,
        latent_dim=16,
        alpha_max=1.0,
    )


def main() -> None:
    run_training(build_config())


if __name__ == "__main__":
    main()
