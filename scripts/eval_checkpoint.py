"""Evaluate a saved actor checkpoint under deterministic/stochastic policies."""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import tyro

from sr_cpo.train import (
    TrainConfig,
    _parse_eval_std_scales,
    _suffix_eval_metrics,
    initialize_training,
    load_actor_checkpoint,
    make_policy_evaluator,
)


@dataclass(frozen=True)
class EvalCheckpointConfig(TrainConfig):
    """Configures full-episode checkpoint evaluation without training."""

    checkpoint_path: str = ""
    eval_seed: int = 12345
    eval_action_std_scales: str = "0.0,0.25,0.5,1.0"


def _train_config(config: EvalCheckpointConfig) -> TrainConfig:
    values = {field.name: getattr(config, field.name) for field in fields(TrainConfig)}
    return TrainConfig(**values)


def _format_value(value: jax.Array) -> str:
    scalar = float(jnp.mean(value))
    if abs(scalar) >= 100.0 or abs(scalar) < 0.001:
        return f"{scalar:.4e}"
    return f"{scalar:.4f}"


def main() -> None:
    args = tyro.cli(EvalCheckpointConfig)
    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required")
    config = _train_config(args)
    state, objects = initialize_training(config)
    actor_params = load_actor_checkpoint(args.checkpoint_path, state.actor_params)
    eval_key = jax.random.PRNGKey(args.eval_seed)
    merged_metrics: dict[str, jax.Array] = {}
    for std_scale in _parse_eval_std_scales(args.eval_action_std_scales):
        eval_key, scale_key = jax.random.split(eval_key)
        evaluator = make_policy_evaluator(objects, config, std_scale=std_scale)
        metrics = evaluator(actor_params, scale_key)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        merged_metrics.update(_suffix_eval_metrics(metrics, std_scale=std_scale))

    for key in sorted(merged_metrics):
        print(f"{key}={_format_value(merged_metrics[key])}")


if __name__ == "__main__":
    main()
