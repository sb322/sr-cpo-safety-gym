"""Trace deterministic evaluation goal resampling around success events."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from sr_cpo.train import (
    TrainConfig,
    initialize_training,
    make_training_epoch,
    prefill_buffer,
    trace_deterministic_eval_goal_resampling,
)


@dataclass(frozen=True)
class TraceConfig(TrainConfig):
    """Train briefly, then trace deterministic eval goal/robot positions."""

    trace_episodes: int = 2
    trace_window: int = 10
    trace_seed: int = 12345
    trace_output: str = "eval_goal_resampling_trace.csv"


def _train_config(config: TraceConfig) -> TrainConfig:
    values = {field.name: getattr(config, field.name) for field in fields(TrainConfig)}
    return TrainConfig(**values)


def _selected_episode_indices(trace: dict[str, np.ndarray], count: int) -> list[int]:
    reached = trace["goal_reached"] > 0.5
    success_indices = np.flatnonzero(np.any(reached, axis=0))
    selected = list(success_indices[:count])
    for index in range(reached.shape[1]):
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    return selected


def _first_success_t(trace: dict[str, np.ndarray], episode: int) -> int | None:
    hits = np.flatnonzero(trace["goal_reached"][:, episode] > 0.5)
    if hits.size == 0:
        return None
    return int(hits[0])


def _success_run_length(
    trace: dict[str, np.ndarray], episode: int, first_t: int
) -> int:
    reached = trace["goal_reached"][:, episode] > 0.5
    run_length = 0
    for flag in reached[first_t:]:
        if not flag:
            break
        run_length += 1
    return run_length


def _write_trace_csv(
    path: Path, trace: dict[str, np.ndarray], selected_episodes: list[int]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "t",
                "robot_x",
                "robot_y",
                "goal_x",
                "goal_y",
                "goal_dist",
                "goal_reached",
                "goal_x_before",
                "goal_y_before",
                "goal_shift",
                "old_goal_dist_after_step",
            ]
        )
        for episode in selected_episodes:
            for t in range(trace["goal_reached"].shape[0]):
                robot_xy = trace["robot_xy"][t, episode]
                goal_xy = trace["goal_xy"][t, episode]
                goal_xy_before = trace["goal_xy_before"][t, episode]
                writer.writerow(
                    [
                        episode,
                        t,
                        f"{robot_xy[0]:.6f}",
                        f"{robot_xy[1]:.6f}",
                        f"{goal_xy[0]:.6f}",
                        f"{goal_xy[1]:.6f}",
                        f"{trace['goal_dist'][t, episode]:.6f}",
                        int(trace["goal_reached"][t, episode] > 0.5),
                        f"{goal_xy_before[0]:.6f}",
                        f"{goal_xy_before[1]:.6f}",
                        f"{trace['goal_shift'][t, episode]:.6f}",
                        f"{trace['old_goal_dist_after_step'][t, episode]:.6f}",
                    ]
                )


def _print_success_window(
    trace: dict[str, np.ndarray], episode: int, first_t: int, window: int
) -> None:
    start = max(0, first_t - window)
    end = min(trace["goal_reached"].shape[0], first_t + window + 1)
    print(f"===== TRACE WINDOW episode={episode} first_success_t={first_t} =====")
    print(
        "t robot_x robot_y goal_x goal_y goal_dist reached "
        "goal_shift old_goal_dist_after_step"
    )
    for t in range(start, end):
        robot_xy = trace["robot_xy"][t, episode]
        goal_xy = trace["goal_xy"][t, episode]
        print(
            f"{t} "
            f"{robot_xy[0]:.4f} {robot_xy[1]:.4f} "
            f"{goal_xy[0]:.4f} {goal_xy[1]:.4f} "
            f"{trace['goal_dist'][t, episode]:.4f} "
            f"{int(trace['goal_reached'][t, episode] > 0.5)} "
            f"{trace['goal_shift'][t, episode]:.4f} "
            f"{trace['old_goal_dist_after_step'][t, episode]:.4f}"
        )


def main() -> None:
    args = tyro.cli(TraceConfig)
    config = _train_config(args)
    if not config.use_real_env:
        raise ValueError("trace_eval_goal_resampling requires --use-real-env")

    state, objects = initialize_training(config)
    state = prefill_buffer(state, objects, config)
    training_epoch = make_training_epoch(objects, config)
    for epoch in range(config.epochs):
        start = time.perf_counter()
        state, metrics = training_epoch(state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        print(
            f"[trace-train {epoch + 1}/{config.epochs}] "
            f"steps={int(state.step):,} t={time.perf_counter() - start:.1f}s"
        )

    trace_jax = trace_deterministic_eval_goal_resampling(
        objects,
        config,
        state.actor_params,
        jax.random.PRNGKey(args.trace_seed),
    )
    trace = {
        key: np.asarray(value)
        for key, value in jax.tree_util.tree_map(jnp.asarray, trace_jax).items()
    }
    selected_episodes = _selected_episode_indices(trace, args.trace_episodes)
    output_path = Path(args.trace_output)
    _write_trace_csv(output_path, trace, selected_episodes)
    print(f"TRACE_CSV={output_path}")
    print(f"SELECTED_EPISODES={selected_episodes}")

    for episode in selected_episodes:
        first_t = _first_success_t(trace, episode)
        if first_t is None:
            print(f"episode={episode} first_success_t=None")
            continue
        goal_changed = trace["goal_shift"][first_t, episode] > 1.0e-6
        run_length = _success_run_length(trace, episode, first_t)
        print(
            f"episode={episode} first_success_t={first_t} "
            f"goal_changed_same_step={int(goal_changed)} "
            f"goal_shift={trace['goal_shift'][first_t, episode]:.6f} "
            f"success_run_length={run_length}"
        )
        _print_success_window(trace, episode, first_t, args.trace_window)


if __name__ == "__main__":
    main()
