"""Loss functions and per-step probes for SR-CPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jax.Array
Params = Mapping[str, Any]
LOG_2 = jnp.log(2.0)
_L2_EPS = 1e-12


def row_l2_normalize(x: Array, eps: float = 1e-12) -> tuple[Array, Array]:
    """Normalizes rows with epsilon inside sqrt for finite backward passes."""

    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm, norm


def contrastive_logits(sa_repr: Array, g_repr: Array, tau: float) -> Array:
    """Computes cosine InfoNCE logits from unnormalized representations."""

    sa_hat, _ = row_l2_normalize(sa_repr)
    g_hat, _ = row_l2_normalize(g_repr)
    return jnp.einsum("ik,jk->ij", sa_hat, g_hat) / tau


def _pairwise_scores(
    sa_repr: Array,
    g_repr: Array,
    *,
    tau: float,
    score_mode: str,
) -> Array:
    if score_mode == "cosine":
        return contrastive_logits(sa_repr, g_repr, tau)
    if score_mode == "l2":
        sq_dist = jnp.sum(
            jnp.square(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1
        )
        return -jnp.sqrt(sq_dist + _L2_EPS)
    raise ValueError(f"unknown critic score mode: {score_mode!r}")


def _paired_scores(
    sa_repr: Array,
    g_repr: Array,
    *,
    tau: float,
    score_mode: str,
) -> Array:
    if score_mode == "cosine":
        sa_hat, _ = row_l2_normalize(sa_repr)
        g_hat, _ = row_l2_normalize(g_repr)
        return jnp.sum(sa_hat * g_hat, axis=-1) / tau
    if score_mode == "l2":
        sq_dist = jnp.sum(jnp.square(sa_repr - g_repr), axis=-1)
        return -jnp.sqrt(sq_dist + _L2_EPS)
    raise ValueError(f"unknown critic score mode: {score_mode!r}")


def _has_nonfinite(x: Array) -> Array:
    return jnp.any(~jnp.isfinite(x)).astype(jnp.float32)


@dataclass(frozen=True)
class TanhGaussianSample:
    """Sample and diagnostics from a tanh-Gaussian policy."""

    action: Array
    log_prob: Array
    gaussian_logp: Array
    sat_correction: Array
    log_std: Array


def sample_tanh_gaussian(
    actor: Any,
    actor_params: Params,
    state: Array,
    goal: Array,
    key: Array,
) -> TanhGaussianSample:
    """Samples a reparameterized tanh-Gaussian action and its log density."""

    mean, log_std = actor.apply(actor_params, state, goal)
    std = jnp.exp(log_std)
    pre_tanh = mean + std * jax.random.normal(key, shape=mean.shape)
    action = nn.tanh(pre_tanh)
    gaussian_logp = (
        -0.5 * jnp.square((pre_tanh - mean) / std)
        - log_std
        - 0.5 * jnp.log(2.0 * jnp.pi)
    ).sum(axis=-1)
    sat_correction = (2.0 * (LOG_2 - pre_tanh - nn.softplus(-2.0 * pre_tanh))).sum(
        axis=-1
    )
    log_prob = gaussian_logp - sat_correction
    return TanhGaussianSample(
        action=action,
        log_prob=log_prob,
        gaussian_logp=gaussian_logp,
        sat_correction=sat_correction,
        log_std=log_std,
    )


def _extras(transitions: Any) -> Mapping[str, Any]:
    extras = getattr(transitions, "extras", None)
    if extras is None:
        raise AttributeError("transitions must expose an extras mapping")
    return extras


def _goal_from_transitions(transitions: Any, goal: Array | None = None) -> Array:
    if goal is not None:
        return jnp.asarray(goal, dtype=jnp.float32)
    extras = _extras(transitions)
    if "goal" not in extras:
        raise KeyError('critic_loss_fn requires goal or transitions.extras["goal"]')
    return jnp.asarray(extras["goal"], dtype=jnp.float32)


def critic_loss_fn(
    critic_params: Params,
    transitions: Any,
    *,
    sa_encoder: Any,
    g_encoder: Any,
    tau: float = 0.1,
    rho: float = 0.1,
    goal: Array | None = None,
    score_mode: str = "cosine",
) -> tuple[Array, dict[str, Array]]:
    """Symmetric InfoNCE reward-critic loss with autograd-safe row-L2."""

    observation = jnp.asarray(transitions.observation, dtype=jnp.float32)
    action = jnp.asarray(transitions.action, dtype=jnp.float32)
    goal_arr = _goal_from_transitions(transitions, goal)

    sa_repr = sa_encoder.apply(critic_params["sa_encoder"], observation, action)
    g_repr = g_encoder.apply(critic_params["g_encoder"], goal_arr)
    sa_repr_norm = jnp.mean(jnp.linalg.norm(sa_repr, axis=-1))
    g_repr_norm = jnp.mean(jnp.linalg.norm(g_repr, axis=-1))
    sa_norm_min = jnp.min(jnp.linalg.norm(sa_repr, axis=-1))
    g_norm_min = jnp.min(jnp.linalg.norm(g_repr, axis=-1))

    logits = _pairwise_scores(
        sa_repr,
        g_repr,
        tau=tau,
        score_mode=score_mode,
    )
    logsumexp = jax.nn.logsumexp(logits, axis=1)
    diag = jnp.diag(logits)
    loss = -jnp.mean(diag - logsumexp) + rho * jnp.mean(logsumexp**2)

    eye = jnp.eye(logits.shape[0], dtype=logits.dtype)
    logits_pos = jnp.sum(logits * eye) / jnp.sum(eye)
    logits_neg = jnp.sum(logits * (1.0 - eye)) / jnp.sum(1.0 - eye)
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0]))

    probes = {
        "nan_obs": _has_nonfinite(observation),
        "nan_sa": _has_nonfinite(sa_repr),
        "nan_g": _has_nonfinite(g_repr),
        "nan_logits": _has_nonfinite(logits),
        "sa_norm_min": sa_norm_min,
        "g_norm_min": g_norm_min,
        "sa_repr_norm": sa_repr_norm,
        "g_repr_norm": g_repr_norm,
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "accuracy": accuracy.astype(jnp.float32),
    }
    return loss, probes


def actor_loss_fn(
    actor_params: Params,
    critic_params: Params,
    cost_critic_params: Params,
    transitions: Any,
    key: Array,
    *,
    actor: Any,
    sa_encoder: Any,
    g_encoder: Any,
    cost_critic: Any,
    log_alpha: Array,
    lambda_tilde: Array,
    tau: float = 0.1,
    nu_f: float = 1.0,
    nu_c: float = 1.0,
    goal: Array | None = None,
    score_mode: str = "cosine",
) -> tuple[Array, dict[str, Array]]:
    """Preconditioned SR-CPO actor loss with component probes."""

    state = jnp.asarray(transitions.observation, dtype=jnp.float32)
    goal_arr = _goal_from_transitions(transitions, goal)
    sample = sample_tanh_gaussian(actor, actor_params, state, goal_arr, key)

    sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, sample.action)
    g_repr = g_encoder.apply(critic_params["g_encoder"], goal_arr)
    sa_norm_min = jnp.min(jnp.linalg.norm(sa_repr, axis=-1))
    g_norm_min = jnp.min(jnp.linalg.norm(g_repr, axis=-1))
    f_value = _paired_scores(
        sa_repr,
        g_repr,
        tau=tau,
        score_mode=score_mode,
    )

    alpha = jnp.exp(log_alpha)
    qc_value = cost_critic.apply(
        cost_critic_params, state, sample.action, goal_arr
    )
    constraint_term = lambda_tilde * qc_value / nu_c
    loss = jnp.mean(
        alpha * sample.log_prob
        - f_value / nu_f
        + constraint_term
    )

    probes = {
        "alpha": alpha,
        "log_prob_mean": jnp.mean(sample.log_prob),
        "alpha_logprob_mean": alpha * jnp.mean(sample.log_prob),
        "gaussian_logp_mean": jnp.mean(sample.gaussian_logp),
        "sat_correction_mean": jnp.mean(sample.sat_correction),
        "log_std_mean": jnp.mean(sample.log_std),
        "f_term_mean": jnp.mean(f_value) / nu_f,
        "qc_actor_mean": jnp.mean(qc_value),
        "constraint_term_mean": jnp.mean(constraint_term),
        "nan_sa": _has_nonfinite(sa_repr),
        "nan_g": _has_nonfinite(g_repr),
        "nan_action": _has_nonfinite(sample.action),
        "nan_f": _has_nonfinite(f_value),
        "sa_norm_min": sa_norm_min,
        "g_norm_min": g_norm_min,
        "action_abs_max": jnp.max(jnp.abs(sample.action)),
    }
    return loss, probes


def cost_critic_loss_fn(
    cost_critic_params: Params,
    cost_critic_target_params: Params,
    actor_params: Params,
    transitions: Any,
    key: Array,
    *,
    actor: Any,
    cost_critic: Any,
    gamma_c: float = 0.99,
    goal: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """TD(0) cost-critic loss with Bellman-B c(s_{t+1}) targets."""

    state = jnp.asarray(transitions.observation, dtype=jnp.float32)
    action = jnp.asarray(transitions.action, dtype=jnp.float32)
    extras = _extras(transitions)
    next_state = jnp.asarray(extras["next_state"], dtype=jnp.float32)
    cost = jnp.asarray(extras["cost"], dtype=jnp.float32)
    discount = jnp.asarray(transitions.discount, dtype=jnp.float32)
    goal_arr = _goal_from_transitions(transitions, goal)

    next_sample = sample_tanh_gaussian(actor, actor_params, next_state, goal_arr, key)
    qc_target = cost_critic.apply(
        cost_critic_target_params, next_state, next_sample.action, goal_arr
    )
    target = jax.lax.stop_gradient(cost + gamma_c * discount * qc_target)
    qc_online = cost_critic.apply(cost_critic_params, state, action, goal_arr)
    loss = 0.5 * jnp.mean(jnp.square(qc_online - target))
    probes = {
        "cost_critic_loss": loss,
        "mean_cost": jnp.mean(cost),
        "mean_qc": jnp.mean(qc_online),
        "mean_target": jnp.mean(target),
    }
    return loss, probes


def alpha_loss_fn(log_alpha: Array, log_prob: Array, action_size: int) -> Array:
    """SAC entropy-temperature loss."""

    return -jnp.mean(log_alpha * (log_prob + action_size))
