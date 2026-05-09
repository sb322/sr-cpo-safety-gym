#!/usr/bin/env python3
"""Oracle dense-cost action-geometry diagnostics for Safety-Gym GoToGoal.

This script intentionally stays outside the SR-CPO training loop.  It samples
frozen environment states, evaluates candidate actions with oracle env rollouts,
and optionally fits the production CostCritic directly on those labels.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def bootstrap_mean_ci_np(
    values: np.ndarray, *, num_samples: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("bootstrap values must be one-dimensional")
    rng = np.random.default_rng(seed)
    idx = rng.choice(values.shape[0], size=(num_samples, values.shape[0]), replace=True)
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def report_bootstrap(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_npz")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args(argv)

    data = np.load(args.results_npz)
    spearman_keys = sorted(k for k in data.files if k.endswith("_spearman"))
    if args.prefix:
        spearman_keys = [k for k in spearman_keys if k.startswith(args.prefix)]
    if not spearman_keys:
        raise ValueError("no per-state spearman arrays found")
    for spearman_key in spearman_keys:
        top1_key = spearman_key[: -len("_spearman")] + "_top1"
        spear_mean, spear_low, spear_high = bootstrap_mean_ci_np(
            data[spearman_key], num_samples=args.samples
        )
        print(
            f"{spearman_key}: mean={spear_mean:.6g} "
            f"CI=[{spear_low:.6g}, {spear_high:.6g}]"
        )
        if top1_key in data:
            top1_mean, top1_low, top1_high = bootstrap_mean_ci_np(
                data[top1_key], num_samples=args.samples
            )
            print(
                f"{top1_key}: mean={top1_mean:.6g} "
                f"CI=[{top1_low:.6g}, {top1_high:.6g}]"
            )
    return 0


if len(sys.argv) > 1 and sys.argv[1] == "--report-bootstrap":
    raise SystemExit(report_bootstrap(sys.argv[2:]))

import jax
import jax.numpy as jnp
import optax
import tyro

from sr_cpo.networks import CostCritic
from sr_cpo.train import (
    TrainConfig,
    _deterministic_action,
    initialize_training,
    load_actor_checkpoint,
)


Array = jax.Array


@dataclass
class OracleFitConfig:
    seed: int = 0
    num_states: int = 128
    num_candidates: int = 16
    burnin_steps: int = 200
    horizons: str = "1,5,20"
    gamma_c: float = 0.99
    cost_dense_prox_tau: float = 0.5
    actor_checkpoint: str = ""
    state_action_source: str = "actor"
    state_sample_mode: str = "final"
    state_sample_min_step: int = 0
    state_sample_max_step: int = 0
    perturb_std: float = 0.25
    fit_steps: int = 1000
    fit_batch_size: int = 1024
    fit_learning_rate: float = 3e-4
    fit_val_fraction: float = 0.1
    fit_labels: str = "dense"
    fit_label_transforms: str = "absolute"
    fit_losses: str = "mse"
    fit_goal_modes: str = "normal"
    shuffle_goals_across_states: bool = False
    fit_rank_eps: float = 1e-6
    fit_bootstrap_samples: int = 200
    action_effect_eps: float = 1e-6
    done_mode: str = "extend"
    load_rank_dump: str = ""
    rank_dump_oracle_seeds: int = 5
    oracle_results_npz: str = ""
    include_synthetic_action_norm: bool = True
    output_csv: str = "figures/data/oracle_critic_fit.csv"
    width: int = 256
    num_blocks: int = 8
    latent_dim: int = 64
    use_residual: bool = True
    goal_mode: str = "relative_xy"
    mask_native_goal_lidar: bool = True
    critic_score_mode: str = "l2"
    env_episode_length: int = 1000


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one horizon is required")
    if any(value <= 0 for value in values):
        raise ValueError("horizons must be positive")
    return tuple(sorted(set(values)))


def _parse_csv_values(
    raw: str, *, allowed: set[str], name: str
) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"at least one {name} is required")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {name}(s): {sorted(unknown)}")
    return values


def _repeat_env_state(env_state: Any, repeats: int, batch_size: int) -> Any:
    def repeat_leaf(x: Any) -> Any:
        if isinstance(x, jax.Array) and x.ndim > 0 and x.shape[0] == batch_size:
            return jnp.repeat(x, repeats, axis=0)
        return x

    return jax.tree_util.tree_map(repeat_leaf, env_state)


def _select_env_state(
    old_state: Any, new_state: Any, mask: Array, batch_size: int
) -> Any:
    def select_leaf(old: Any, new: Any) -> Any:
        if isinstance(new, jax.Array) and new.ndim > 0 and new.shape[0] == batch_size:
            shaped_mask = jnp.reshape(mask, (batch_size,) + (1,) * (new.ndim - 1))
            return jnp.where(shaped_mask, new, old)
        return old

    return jax.tree_util.tree_map(select_leaf, old_state, new_state)


def _reshape_candidate(values: Array, num_states: int, num_candidates: int) -> Array:
    return jnp.reshape(jnp.asarray(values, dtype=jnp.float32), (num_states, num_candidates))


def _variance_decomposition(labels: Array) -> dict[str, Array]:
    labels = jnp.asarray(labels, dtype=jnp.float32)
    state_mean = jnp.mean(labels, axis=1)
    within_var = jnp.mean(jnp.var(labels, axis=1))
    between_var = jnp.var(state_mean)
    spread = jnp.max(labels, axis=1) - jnp.min(labels, axis=1)
    magnitude = jnp.mean(jnp.abs(labels))
    return {
        "label_mean": jnp.mean(labels),
        "label_std": jnp.std(labels),
        "within_var": within_var,
        "between_var": between_var,
        "within_between_ratio": within_var / (between_var + 1e-12),
        "mean_action_spread": jnp.mean(spread),
        "median_action_spread": jnp.median(spread),
        "spread_magnitude_ratio": jnp.mean(spread) / (magnitude + 1e-12),
        "nonzero_spread_frac": jnp.mean((spread > 1e-6).astype(jnp.float32)),
    }


def _transform_labels(labels: Array, transform: str) -> Array:
    labels = jnp.asarray(labels, dtype=jnp.float32)
    if transform == "absolute":
        return labels
    centered = labels - jnp.mean(labels, axis=1, keepdims=True)
    if transform == "centered":
        return centered
    if transform == "zscore":
        scale = jnp.std(labels, axis=1, keepdims=True)
        return centered / (scale + 1e-6)
    raise ValueError("label transform must be absolute, centered, or zscore")


def _rank_rows(x: Array) -> Array:
    order = jnp.argsort(x, axis=1)
    ranks = jnp.argsort(order, axis=1)
    return ranks.astype(jnp.float32)


def _spearman_by_state(pred: Array, label: Array) -> Array:
    pred_rank = _rank_rows(pred)
    label_rank = _rank_rows(label)
    pred_centered = pred_rank - jnp.mean(pred_rank, axis=1, keepdims=True)
    label_centered = label_rank - jnp.mean(label_rank, axis=1, keepdims=True)
    numer = jnp.sum(pred_centered * label_centered, axis=1)
    denom = jnp.sqrt(
        jnp.sum(pred_centered * pred_centered, axis=1)
        * jnp.sum(label_centered * label_centered, axis=1)
        + 1e-12
    )
    return numer / denom


def _bootstrap_mean_interval(
    values: Array, key: Array, num_samples: int
) -> tuple[Array, Array]:
    values = jnp.asarray(values, dtype=jnp.float32)
    if num_samples <= 0:
        mean = jnp.mean(values)
        return mean, mean
    idx = jax.random.randint(key, (num_samples, values.shape[0]), 0, values.shape[0])
    boot_means = jnp.mean(values[idx], axis=1)
    return jnp.quantile(boot_means, 0.025), jnp.quantile(boot_means, 0.975)


def _candidate_actions(
    *,
    actor_action: Array,
    key: Array,
    num_candidates: int,
    perturb_std: float,
) -> Array:
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2")
    num_states, action_dim = actor_action.shape
    key_perturb, key_uniform = jax.random.split(key)
    num_perturb = max((num_candidates - 1) // 2, 0)
    num_uniform = num_candidates - 1 - num_perturb
    pieces = [actor_action[:, None, :]]
    if num_perturb:
        noise = perturb_std * jax.random.normal(
            key_perturb, (num_states, num_perturb, action_dim), dtype=jnp.float32
        )
        pieces.append(jnp.clip(actor_action[:, None, :] + noise, -1.0, 1.0))
    if num_uniform:
        pieces.append(
            jax.random.uniform(
                key_uniform,
                (num_states, num_uniform, action_dim),
                minval=-1.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
        )
    return jnp.concatenate(pieces, axis=1)


def _make_train_config(config: OracleFitConfig) -> TrainConfig:
    return TrainConfig(
        seed=config.seed,
        epochs=1,
        steps_per_epoch=1,
        num_envs=config.num_states,
        unroll_length=8,
        env_episode_length=config.env_episode_length,
        prefill_steps=0,
        sgd_steps=1,
        batch_size=min(config.fit_batch_size, max(config.num_states, 1)),
        buffer_capacity=8,
        width=config.width,
        num_blocks=config.num_blocks,
        latent_dim=config.latent_dim,
        use_residual=config.use_residual,
        use_real_env=True,
        goal_mode=config.goal_mode,
        goal_start=55,
        goal_dim=2,
        mask_native_goal_lidar=config.mask_native_goal_lidar,
        critic_score_mode=config.critic_score_mode,
        gamma_c=config.gamma_c,
        cost_mode="dense_proximity",
        cost_dense_prox_tau=config.cost_dense_prox_tau,
    )


def _collect_states(
    *,
    config: OracleFitConfig,
    train_config: TrainConfig,
    objects: Any,
    actor_params: Any,
    env_state: Any,
    key: Array,
) -> Any:
    env_adapter = objects.env_adapter
    if env_adapter is None:
        raise ValueError("real env adapter is required")

    if config.state_sample_mode not in {"final", "uniform"}:
        raise ValueError("state_sample_mode must be 'final' or 'uniform'")

    def collect_step(carry: tuple[Any, Array], _: Array) -> tuple[tuple[Any, Array], None]:
        state, step_key = carry
        step_key, action_key = jax.random.split(step_key)
        obs = env_adapter._state_observation(state)
        goal = env_adapter.desired_goal(state)
        if config.state_action_source == "random":
            action = jax.random.uniform(
                action_key,
                (config.num_states, env_adapter.action_size),
                minval=-1.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
        elif config.state_action_source == "actor":
            action = _deterministic_action(
                objects.actor, actor_params, obs, goal, train_config
            )
        else:
            raise ValueError("state_action_source must be 'actor' or 'random'")
        next_state, _ = env_adapter.step(state, action)
        return (next_state, step_key), None

    if config.state_sample_mode == "final":
        (env_state, _), _ = jax.lax.scan(
            collect_step, (env_state, key), jnp.arange(config.burnin_steps)
        )
        return env_state

    max_step = config.state_sample_max_step or config.burnin_steps
    min_step = config.state_sample_min_step
    if min_step < 0 or max_step < min_step:
        raise ValueError("uniform state sampling requires 0 <= min_step <= max_step")
    sample_key, rollout_key = jax.random.split(key)
    target_steps = jax.random.randint(
        sample_key,
        (config.num_states,),
        minval=min_step,
        maxval=max_step + 1,
    )

    def uniform_step(
        carry: tuple[Any, Any, Array], t: Array
    ) -> tuple[tuple[Any, Any, Array], None]:
        state, selected_state, step_key = carry
        (next_state, step_key), _ = collect_step((state, step_key), t)
        mask = target_steps == t
        selected_state = _select_env_state(
            selected_state, next_state, mask, config.num_states
        )
        return (next_state, selected_state, step_key), None

    (_, selected_state, _), _ = jax.lax.scan(
        uniform_step,
        (env_state, env_state, rollout_key),
        jnp.arange(1, max_step + 1),
    )
    return selected_state


def _oracle_rollout_labels(
    *,
    config: OracleFitConfig,
    train_config: TrainConfig,
    objects: Any,
    actor_params: Any,
    env_state: Any,
    candidate_actions: Array,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    env_adapter = objects.env_adapter
    if env_adapter is None:
        raise ValueError("real env adapter is required")
    num_states, num_candidates, action_dim = candidate_actions.shape
    flat_state = _repeat_env_state(env_state, num_candidates, num_states)
    flat_actions = jnp.reshape(candidate_actions, (num_states * num_candidates, action_dim))
    next_state, transition = env_adapter.step(flat_state, flat_actions)
    done_step = _reshape_candidate(
        1.0 - transition.discount, num_states, num_candidates
    )
    next_obs_by_action = jnp.reshape(
        env_adapter._state_observation(next_state),
        (num_states, num_candidates, -1),
    )
    next_obs_delta = next_obs_by_action - next_obs_by_action[:, :1, :]
    next_obs_spread = jnp.max(jnp.linalg.norm(next_obs_delta, axis=-1), axis=1)
    dense_step = _reshape_candidate(
        transition.extras["dense_cost"], num_states, num_candidates
    )
    sparse_step = _reshape_candidate(
        transition.extras["sparse_cost"], num_states, num_candidates
    )
    min_hazard_dist = _reshape_candidate(
        transition.extras["min_hazard_dist"], num_states, num_candidates
    )

    dense_return = dense_step
    sparse_return = sparse_step
    alive = 1.0 - done_step
    any_done = done_step > 0.5
    dense_returns: dict[int, Array] = {}
    sparse_returns: dict[int, Array] = {}
    valid_by_horizon: dict[int, Array] = {}
    if 1 in horizons:
        dense_returns[1] = dense_return
        sparse_returns[1] = sparse_return
        valid_by_horizon[1] = ~jnp.any(any_done, axis=1)

    current_state = next_state
    discount = config.gamma_c
    for step in range(2, max(horizons) + 1):
        obs = env_adapter._state_observation(current_state)
        goal = env_adapter.desired_goal(current_state)
        action = _deterministic_action(objects.actor, actor_params, obs, goal, train_config)
        current_state, transition = env_adapter.step(current_state, action)
        dense = _reshape_candidate(
            transition.extras["dense_cost"], num_states, num_candidates
        )
        sparse = _reshape_candidate(
            transition.extras["sparse_cost"], num_states, num_candidates
        )
        step_done = _reshape_candidate(
            1.0 - transition.discount, num_states, num_candidates
        )
        # Done-aware oracle labels: after the first terminal transition, extend
        # the trajectory with zero future safety cost. This avoids treating early
        # success as more dangerous just because the env wrapper reset.
        dense_return = dense_return + alive * discount * dense
        sparse_return = sparse_return + alive * discount * sparse
        any_done = any_done | ((step_done > 0.5) & (alive > 0.5))
        alive = alive * (1.0 - step_done)
        if step in horizons:
            dense_returns[step] = dense_return
            sparse_returns[step] = sparse_return
            valid_by_horizon[step] = ~jnp.any(any_done, axis=1)
        discount *= config.gamma_c

    obs = env_adapter._state_observation(env_state)
    goal = env_adapter.desired_goal(env_state)
    actor_action = _deterministic_action(objects.actor, actor_params, obs, goal, train_config)
    synthetic_action_norm = jnp.sum(jnp.square(candidate_actions), axis=-1)
    return {
        "obs": obs,
        "goal": goal,
        "actor_action": actor_action,
        "actions": candidate_actions,
        "dense_returns": dense_returns,
        "sparse_returns": sparse_returns,
        "valid_by_horizon": valid_by_horizon,
        "done_mode": config.done_mode,
        "min_hazard_dist_h1": min_hazard_dist,
        "next_obs_action_spread": next_obs_spread,
        "synthetic_action_norm": synthetic_action_norm,
    }


def _action_effect_sanity(spread: Array, eps: float) -> dict[str, Array]:
    spread = jnp.asarray(spread, dtype=jnp.float32)
    return {
        "next_obs_action_spread_mean": jnp.mean(spread),
        "next_obs_action_spread_min": jnp.min(spread),
        "next_obs_action_spread_p10": jnp.quantile(spread, 0.10),
        "next_obs_action_spread_p50": jnp.quantile(spread, 0.50),
        "next_obs_action_spread_frac_gt_eps": jnp.mean((spread > eps).astype(jnp.float32)),
    }


def _shuffle_rows(values: Array, key: Array) -> Array:
    return values[jax.random.permutation(key, values.shape[0])]


def _done_aware_return(costs: Array, discounts: Array, gamma: float) -> Array:
    """Returns discounted costs, zeroing future terms after first done."""

    costs = jnp.asarray(costs, dtype=jnp.float32)
    discounts = jnp.asarray(discounts, dtype=jnp.float32)
    alive = jnp.ones(costs.shape[1:], dtype=jnp.float32)
    total = jnp.zeros(costs.shape[1:], dtype=jnp.float32)
    discount = jnp.asarray(1.0, dtype=jnp.float32)
    for step in range(costs.shape[0]):
        total = total + alive * discount * costs[step]
        alive = alive * discounts[step]
        discount = discount * gamma
    return total


def _flatten_by_state(values: Array, state_indices: Array) -> Array:
    selected = values[state_indices]
    return jnp.reshape(selected, (-1, *values.shape[2:]))


def _repeat_by_candidate(values: Array, state_indices: Array, num_candidates: int) -> Array:
    selected = values[state_indices]
    repeated = jnp.repeat(selected[:, None, :], num_candidates, axis=1)
    return jnp.reshape(repeated, (-1, selected.shape[-1]))


def _predict_by_state(
    critic: CostCritic,
    params: Any,
    states: Array,
    goals: Array,
    actions: Array,
) -> Array:
    num_states, num_candidates, action_dim = actions.shape
    flat_states = jnp.repeat(states[:, None, :], num_candidates, axis=1).reshape(
        num_states * num_candidates, states.shape[-1]
    )
    flat_goals = jnp.repeat(goals[:, None, :], num_candidates, axis=1).reshape(
        num_states * num_candidates, goals.shape[-1]
    )
    flat_actions = actions.reshape(num_states * num_candidates, action_dim)
    preds = critic.apply(params, flat_states, flat_actions, flat_goals)
    return preds.reshape(num_states, num_candidates)


def _fit_oracle_critic(
    *,
    config: OracleFitConfig,
    states: Array,
    goals: Array,
    actions: Array,
    labels: Array,
    actor_actions: Array,
    fit_loss: str,
    fit_goal_mode: str,
    key: Array,
) -> tuple[dict[str, Array], dict[str, Array]]:
    num_states, num_candidates, action_dim = actions.shape
    perm_key, goal_key, init_key, train_key, boot_key = jax.random.split(key, 5)
    perm = jax.random.permutation(perm_key, num_states)
    if fit_goal_mode not in {"normal", "shuffled"}:
        raise ValueError("fit_goal_mode must be normal or shuffled")
    val_count = max(1, int(num_states * config.fit_val_fraction))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    if train_idx.size == 0:
        raise ValueError("fit_val_fraction leaves no training states")

    train_fit_goals = goals[train_idx]
    if fit_goal_mode == "shuffled":
        train_fit_goals = _shuffle_rows(train_fit_goals, goal_key)

    train_states = _repeat_by_candidate(states, train_idx, num_candidates)
    train_goals = jnp.repeat(train_fit_goals[:, None, :], num_candidates, axis=1)
    train_goals = jnp.reshape(train_goals, (-1, train_fit_goals.shape[-1]))
    train_actions = _flatten_by_state(actions, train_idx)
    train_labels = jnp.ravel(labels[train_idx])
    train_state_states = states[train_idx]
    train_state_goals = train_fit_goals
    train_state_actions = actions[train_idx]
    train_state_labels = labels[train_idx]

    val_states = states[val_idx]
    val_goals = goals[val_idx]
    val_actions = actions[val_idx]
    val_labels = labels[val_idx]
    val_actor_actions = actor_actions[val_idx]

    critic = CostCritic(
        width=config.width,
        num_blocks=config.num_blocks,
        use_residual=config.use_residual,
    )
    params = critic.init(
        init_key,
        train_states[:1],
        train_actions[:1],
        train_goals[:1],
    )
    optimizer = optax.adam(config.fit_learning_rate)
    opt_state = optimizer.init(params)
    batch_size = min(config.fit_batch_size, train_labels.shape[0])
    state_batch_size = min(
        max(1, config.fit_batch_size // num_candidates), train_state_labels.shape[0]
    )

    @jax.jit
    def mse_train_step(
        params: Any, opt_state: optax.OptState, step_key: Array
    ) -> tuple[Any, optax.OptState, Array]:
        idx = jax.random.randint(step_key, (batch_size,), 0, train_labels.shape[0])

        def loss_fn(p: Any) -> Array:
            pred = critic.apply(
                p, train_states[idx], train_actions[idx], train_goals[idx]
            )
            return jnp.mean(jnp.square(pred - train_labels[idx]))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def rank_train_step(
        params: Any, opt_state: optax.OptState, step_key: Array
    ) -> tuple[Any, optax.OptState, Array]:
        idx = jax.random.randint(
            step_key, (state_batch_size,), 0, train_state_labels.shape[0]
        )
        batch_states = train_state_states[idx]
        batch_goals = train_state_goals[idx]
        batch_actions = train_state_actions[idx]
        batch_labels = train_state_labels[idx]

        def loss_fn(p: Any) -> Array:
            pred = _predict_by_state(
                critic, p, batch_states, batch_goals, batch_actions
            )
            pred_pair = pred[:, None, :] - pred[:, :, None]
            label_pair = batch_labels[:, None, :] - batch_labels[:, :, None]
            target = jnp.sign(label_pair)
            mask = jnp.abs(label_pair) > config.fit_rank_eps
            loss = jax.nn.softplus(-target * pred_pair)
            return jnp.sum(jnp.where(mask, loss, 0.0)) / (jnp.sum(mask) + 1e-6)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    if fit_loss not in {"mse", "rank"}:
        raise ValueError("fit_loss must be mse or rank")

    losses = []
    for _ in range(config.fit_steps):
        train_key, step_key = jax.random.split(train_key)
        if fit_loss == "mse":
            params, opt_state, loss = mse_train_step(params, opt_state, step_key)
        else:
            params, opt_state, loss = rank_train_step(params, opt_state, step_key)
        losses.append(loss)

    pred = _predict_by_state(critic, params, val_states, val_goals, val_actions)
    mse = jnp.mean(jnp.square(pred - val_labels))
    state_mean = jnp.mean(val_labels, axis=1, keepdims=True)
    state_mean_mse = jnp.mean(jnp.square(state_mean - val_labels))
    global_mean_mse = jnp.mean(jnp.square(jnp.mean(train_labels) - val_labels))
    spearman = _spearman_by_state(pred, val_labels)
    top1_by_state = (
        jnp.argmin(pred, axis=1) == jnp.argmin(val_labels, axis=1)
    ).astype(jnp.float32)
    top1 = jnp.mean(top1_by_state)
    spear_key, top1_key = jax.random.split(boot_key)
    spearman_ci_low, spearman_ci_high = _bootstrap_mean_interval(
        spearman, spear_key, config.fit_bootstrap_samples
    )
    top1_ci_low, top1_ci_high = _bootstrap_mean_interval(
        top1_by_state, top1_key, config.fit_bootstrap_samples
    )

    def q_sum(action: Array) -> Array:
        return jnp.sum(critic.apply(params, val_states, action, val_goals))

    action_grad = jax.grad(q_sum)(val_actor_actions)
    grad_norm = jnp.mean(jnp.linalg.norm(action_grad, axis=-1))
    metrics = {
        "fit_train_loss_final": losses[-1] if losses else jnp.asarray(float("nan")),
        "fit_val_mse": mse,
        "fit_state_mean_mse": state_mean_mse,
        "fit_global_mean_mse": global_mean_mse,
        "fit_mse_over_state_mean": mse / (state_mean_mse + 1e-12),
        "fit_within_spearman": jnp.mean(spearman),
        "fit_within_spearman_ci_low": spearman_ci_low,
        "fit_within_spearman_ci_high": spearman_ci_high,
        "fit_top1_match": top1,
        "fit_top1_match_ci_low": top1_ci_low,
        "fit_top1_match_ci_high": top1_ci_high,
        "fit_random_top1": jnp.asarray(1.0 / num_candidates, dtype=jnp.float32),
        "fit_actor_action_grad_norm": grad_norm,
        "fit_val_states": jnp.asarray(val_idx.size, dtype=jnp.float32),
    }
    per_state = {
        "spearman": spearman,
        "top1": top1_by_state,
        "val_idx": val_idx.astype(jnp.int32),
    }
    return metrics, per_state


def _float_dict(values: dict[str, Any]) -> dict[str, float | str | int]:
    out: dict[str, float | str | int] = {}
    for key, value in values.items():
        if isinstance(value, (str, int)):
            out[key] = value
        else:
            out[key] = float(jnp.asarray(value))
    return out


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _results_npz_path(config: OracleFitConfig) -> Path:
    if config.oracle_results_npz:
        return Path(config.oracle_results_npz)
    stem = Path(config.output_csv).stem
    return Path("results") / f"oracle_{stem}.npz"


def _save_per_state_results(path: Path, records: list[tuple[str, dict[str, Array]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, arrays in records:
        for key, value in arrays.items():
            payload[f"{name}_{key}"] = np.asarray(value)
    if payload:
        np.savez(path, **payload)


def _load_rank_dump_arrays(path: Path) -> dict[str, Array]:
    data = np.load(path)
    required = {"states", "candidate_actions", "goals", "labels_dense_h50", "valid_mask"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"rank dump missing required arrays: {sorted(missing)}")
    valid = np.asarray(data["valid_mask"], dtype=bool)
    if not np.any(valid):
        raise ValueError("rank dump has no valid rows")
    states = np.asarray(data["states"], dtype=np.float32)[valid]
    actions = np.asarray(data["candidate_actions"], dtype=np.float32)[valid]
    goals = np.asarray(data["goals"], dtype=np.float32)[valid]
    labels = np.asarray(data["labels_dense_h50"], dtype=np.float32)[valid]
    if states.shape[0] < 2:
        raise ValueError("rank dump needs at least two valid rows")
    return {
        "states": jnp.asarray(states),
        "actions": jnp.asarray(actions),
        "goals": jnp.asarray(goals),
        "labels": jnp.asarray(labels),
        "horizon": jnp.asarray(
            int(np.asarray(data["cost_rank_horizon"])) if "cost_rank_horizon" in data else 50,
            dtype=jnp.int32,
        ),
    }


def _oracle_on_rank_dump(config: OracleFitConfig) -> int:
    dump_path = Path(config.load_rank_dump)
    dump = _load_rank_dump_arrays(dump_path)
    states = dump["states"]
    actions = dump["actions"]
    goals = dump["goals"]
    labels = _transform_labels(dump["labels"], "centered")
    actor_actions = actions[:, 0, :]
    horizon = int(jnp.asarray(dump["horizon"]))
    per_state_records: list[tuple[str, dict[str, Array]]] = []
    rows: list[dict[str, Any]] = []
    spearman_arrays: list[np.ndarray] = []
    top1_arrays: list[np.ndarray] = []
    seeds = max(1, config.rank_dump_oracle_seeds)
    for seed_offset in range(seeds):
        seed = config.seed + seed_offset
        fit_metrics, per_state = _fit_oracle_critic(
            config=config,
            states=states,
            goals=goals,
            actions=actions,
            labels=labels,
            actor_actions=actor_actions,
            fit_loss="rank",
            fit_goal_mode="normal",
            key=jax.random.PRNGKey(seed),
        )
        rows.append(
            _float_dict(
                {
                    "seed": seed,
                    "kind": "oracle_on_rank_buffer",
                    "label": "dense",
                    "label_transform": "centered",
                    "fit_loss": "rank",
                    "fit_goal_mode": "normal",
                    "horizon": horizon,
                    "n_states": int(states.shape[0]),
                    "fit_steps": config.fit_steps,
                    "fit_batch_size": config.fit_batch_size,
                    "fit_learning_rate": config.fit_learning_rate,
                    **fit_metrics,
                }
            )
        )
        record_name = f"rank_buffer_seed{seed}"
        per_state_records.append((record_name, per_state))
        spearman_arrays.append(np.asarray(per_state["spearman"]))
        top1_arrays.append(np.asarray(per_state["top1"]))
    all_spearman = np.concatenate(spearman_arrays)
    all_top1 = np.concatenate(top1_arrays)
    spear_mean, spear_low, spear_high = bootstrap_mean_ci_np(
        all_spearman, num_samples=max(1, config.fit_bootstrap_samples), seed=config.seed
    )
    top1_mean, _, _ = bootstrap_mean_ci_np(
        all_top1, num_samples=max(1, config.fit_bootstrap_samples), seed=config.seed + 1
    )
    output = Path(config.output_csv)
    _write_rows(output, rows)
    if per_state_records:
        results_path = _results_npz_path(config)
        _save_per_state_results(results_path, per_state_records)
        print(f"wrote {results_path}")
    print(f"wrote {output}")
    print(
        "ORACLE_ON_RANK_BUFFER "
        f"spearman_mean={spear_mean:.6g} "
        f"spearman_ci=[{spear_low:.6g},{spear_high:.6g}] "
        f"top1_mean={top1_mean:.6g} "
        f"n_states={states.shape[0]}"
    )
    return 0


def main() -> int:
    config = tyro.cli(OracleFitConfig)
    if config.load_rank_dump:
        return _oracle_on_rank_dump(config)
    if config.done_mode not in {"extend", "filter"}:
        raise ValueError("done_mode must be 'extend' or 'filter'")
    horizons = _parse_csv_ints(config.horizons)
    fit_labels = _parse_csv_values(
        config.fit_labels, allowed={"dense", "sparse"}, name="fit label"
    )
    fit_label_transforms = _parse_csv_values(
        config.fit_label_transforms,
        allowed={"absolute", "centered", "zscore"},
        name="fit label transform",
    )
    fit_losses = _parse_csv_values(
        config.fit_losses, allowed={"mse", "rank"}, name="fit loss"
    )
    fit_goal_modes = _parse_csv_values(
        config.fit_goal_modes, allowed={"normal", "shuffled"}, name="fit goal mode"
    )
    if config.shuffle_goals_across_states and "shuffled" not in fit_goal_modes:
        fit_goal_modes = (*fit_goal_modes, "shuffled")
    train_config = _make_train_config(config)
    key = jax.random.PRNGKey(config.seed)
    state, objects = initialize_training(train_config)
    actor_params = state.actor_params
    if config.actor_checkpoint:
        actor_params = load_actor_checkpoint(config.actor_checkpoint, actor_params)
    key, collect_key, candidate_key, fit_key = jax.random.split(key, 4)
    env_state = _collect_states(
        config=config,
        train_config=train_config,
        objects=objects,
        actor_params=actor_params,
        env_state=state.env_state,
        key=collect_key,
    )
    env_adapter = objects.env_adapter
    if env_adapter is None:
        raise ValueError("real env adapter is required")
    obs = env_adapter._state_observation(env_state)
    goal = env_adapter.desired_goal(env_state)
    actor_action = _deterministic_action(objects.actor, actor_params, obs, goal, train_config)
    actions = _candidate_actions(
        actor_action=actor_action,
        key=candidate_key,
        num_candidates=config.num_candidates,
        perturb_std=config.perturb_std,
    )
    dataset = _oracle_rollout_labels(
        config=config,
        train_config=train_config,
        objects=objects,
        actor_params=actor_params,
        env_state=env_state,
        candidate_actions=actions,
        horizons=horizons,
    )

    rows: list[dict[str, Any]] = []
    base = {
        "seed": config.seed,
        "num_states": config.num_states,
        "num_candidates": config.num_candidates,
        "burnin_steps": config.burnin_steps,
        "gamma_c": config.gamma_c,
        "tau": config.cost_dense_prox_tau,
        "state_action_source": config.state_action_source,
        "state_sample_mode": config.state_sample_mode,
        "state_sample_min_step": config.state_sample_min_step,
        "state_sample_max_step": config.state_sample_max_step,
        "actor_checkpoint": config.actor_checkpoint,
    }

    label_mats: dict[tuple[str, int], Array] = {}
    valid_masks: dict[int, Array] = {}
    for horizon in horizons:
        label_mats[("dense", horizon)] = dataset["dense_returns"][horizon]
        label_mats[("sparse", horizon)] = dataset["sparse_returns"][horizon]
        valid_masks[horizon] = dataset["valid_by_horizon"][horizon]
    if config.include_synthetic_action_norm:
        label_mats[("synthetic_action_norm", 0)] = dataset["synthetic_action_norm"]

    sanity_row = {
        **base,
        "kind": "sanity",
        "label": "next_obs_action_effect",
        "horizon": 1,
    }
    sanity_row.update(
        _float_dict(
            _action_effect_sanity(
                dataset["next_obs_action_spread"], config.action_effect_eps
            )
        )
    )
    rows.append(sanity_row)

    for (label_name, horizon), labels in label_mats.items():
        variance_labels = labels
        states_after_filter = labels.shape[0]
        if config.done_mode == "filter" and horizon in valid_masks:
            variance_labels = labels[valid_masks[horizon]]
            states_after_filter = variance_labels.shape[0]
        if states_after_filter == 0:
            continue
        row = {
            **base,
            "kind": "variance",
            "label": label_name,
            "done_mode": config.done_mode,
            "states_after_done_filter": int(states_after_filter),
            "horizon": horizon,
        }
        row.update(_float_dict(_variance_decomposition(variance_labels)))
        rows.append(row)

    fit_key_base = fit_key
    per_state_records: list[tuple[str, dict[str, Array]]] = []
    if config.fit_steps > 0:
        for horizon in horizons:
            for label_name in fit_labels:
                raw_labels = label_mats[(label_name, horizon)]
                valid_mask = valid_masks[horizon]
                if config.done_mode == "filter":
                    raw_labels = raw_labels[valid_mask]
                    fit_obs = dataset["obs"][valid_mask]
                    fit_goal = dataset["goal"][valid_mask]
                    fit_actions = dataset["actions"][valid_mask]
                    fit_actor_action = dataset["actor_action"][valid_mask]
                else:
                    fit_obs = dataset["obs"]
                    fit_goal = dataset["goal"]
                    fit_actions = dataset["actions"]
                    fit_actor_action = dataset["actor_action"]
                if raw_labels.shape[0] < 2:
                    continue
                for transform in fit_label_transforms:
                    labels = _transform_labels(raw_labels, transform)
                    for fit_loss in fit_losses:
                        for fit_goal_mode in fit_goal_modes:
                            fit_key_base, label_key = jax.random.split(fit_key_base)
                            row = {
                                **base,
                                "kind": "fit",
                                "label": label_name,
                                "label_transform": transform,
                                "fit_loss": fit_loss,
                                "fit_goal_mode": fit_goal_mode,
                                "done_mode": config.done_mode,
                                "fit_states_after_done_filter": int(raw_labels.shape[0]),
                                "horizon": horizon,
                                "fit_steps": config.fit_steps,
                                "fit_batch_size": config.fit_batch_size,
                                "fit_learning_rate": config.fit_learning_rate,
                            }
                            fit_metrics, per_state = _fit_oracle_critic(
                                config=config,
                                states=fit_obs,
                                goals=fit_goal,
                                actions=fit_actions,
                                labels=labels,
                                actor_actions=fit_actor_action,
                                fit_loss=fit_loss,
                                fit_goal_mode=fit_goal_mode,
                                key=label_key,
                            )
                            row.update(_float_dict(fit_metrics))
                            record_name = (
                                f"{label_name}_H{horizon}_{transform}_{fit_loss}_"
                                f"{fit_goal_mode}"
                            )
                            per_state_records.append((record_name, per_state))
                            rows.append(row)
        if config.include_synthetic_action_norm:
            raw_labels = label_mats[("synthetic_action_norm", 0)]
            for transform in fit_label_transforms:
                labels = _transform_labels(raw_labels, transform)
                for fit_loss in fit_losses:
                    for fit_goal_mode in fit_goal_modes:
                        fit_key_base, synth_key = jax.random.split(fit_key_base)
                        row = {
                            **base,
                            "kind": "fit",
                            "label": "synthetic_action_norm",
                            "label_transform": transform,
                            "fit_loss": fit_loss,
                            "fit_goal_mode": fit_goal_mode,
                            "done_mode": config.done_mode,
                            "horizon": 0,
                            "fit_steps": config.fit_steps,
                            "fit_batch_size": config.fit_batch_size,
                            "fit_learning_rate": config.fit_learning_rate,
                        }
                        fit_metrics, per_state = _fit_oracle_critic(
                            config=config,
                            states=dataset["obs"],
                            goals=dataset["goal"],
                            actions=dataset["actions"],
                            labels=labels,
                            actor_actions=dataset["actor_action"],
                            fit_loss=fit_loss,
                            fit_goal_mode=fit_goal_mode,
                            key=synth_key,
                        )
                        row.update(_float_dict(fit_metrics))
                        record_name = (
                            f"synthetic_action_norm_H0_{transform}_{fit_loss}_"
                            f"{fit_goal_mode}"
                        )
                        per_state_records.append((record_name, per_state))
                        rows.append(row)

    rows = [_float_dict(row) for row in rows]
    output = Path(config.output_csv)
    _write_rows(output, rows)
    if per_state_records:
        results_path = _results_npz_path(config)
        _save_per_state_results(results_path, per_state_records)
        print(f"wrote {results_path}")
    print(f"wrote {output}")
    print("config:", asdict(config))
    for row in rows:
        if row["kind"] == "variance":
            print(
                "variance",
                f"label={row['label']}",
                f"H={row['horizon']}",
                f"within/between={float(row['within_between_ratio']):.3e}",
                f"spread/mag={float(row['spread_magnitude_ratio']):.3e}",
                f"mean_spread={float(row['mean_action_spread']):.3e}",
            )
        elif row["kind"] == "sanity":
            print(
                "sanity",
                f"label={row['label']}",
                f"mean_next_obs_spread="
                f"{float(row['next_obs_action_spread_mean']):.3e}",
                f"frac_gt_eps="
                f"{float(row['next_obs_action_spread_frac_gt_eps']):.3f}",
            )
        else:
            print(
                "fit",
                f"label={row['label']}",
                f"transform={row.get('label_transform', 'absolute')}",
                f"loss={row.get('fit_loss', 'mse')}",
                f"goal_mode={row.get('fit_goal_mode', 'normal')}",
                f"H={row['horizon']}",
                f"spearman={float(row['fit_within_spearman']):.3f}",
                f"spear_ci=[{float(row['fit_within_spearman_ci_low']):.3f},"
                f"{float(row['fit_within_spearman_ci_high']):.3f}]",
                f"top1={float(row['fit_top1_match']):.3f}",
                f"top1_ci=[{float(row['fit_top1_match_ci_low']):.3f},"
                f"{float(row['fit_top1_match_ci_high']):.3f}]",
                f"mse/state={float(row['fit_mse_over_state_mean']):.3f}",
                f"grad={float(row['fit_actor_action_grad_norm']):.3e}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
