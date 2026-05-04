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
    qc_zero_action = cost_critic.apply(
        cost_critic_params, state, jnp.zeros_like(sample.action), goal_arr
    )
    qc_neg_action = cost_critic.apply(
        cost_critic_params, state, -sample.action, goal_arr
    )
    reward_actor_term = -f_value / nu_f
    qc_action_delta = qc_value - qc_zero_action
    constraint_term = lambda_tilde * qc_value / nu_c
    loss = jnp.mean(
        alpha * sample.log_prob
        + reward_actor_term
        + constraint_term
    )

    def reward_score_for_action(action: Array) -> Array:
        sa_repr_for_action = sa_encoder.apply(
            critic_params["sa_encoder"], state, action
        )
        g_repr_for_action = g_encoder.apply(critic_params["g_encoder"], goal_arr)
        return _paired_scores(
            sa_repr_for_action,
            g_repr_for_action,
            tau=tau,
            score_mode=score_mode,
        ) / nu_f

    def cost_score_for_action(action: Array) -> Array:
        return cost_critic.apply(cost_critic_params, state, action, goal_arr)

    grad_qr = jax.grad(lambda action: jnp.sum(reward_score_for_action(action)))(
        sample.action
    )
    grad_qc = jax.grad(lambda action: jnp.sum(cost_score_for_action(action)))(
        sample.action
    )
    grad_norm_qr_rows = jnp.linalg.norm(grad_qr, axis=-1)
    grad_norm_qc_rows = jnp.linalg.norm(grad_qc, axis=-1)
    grad_norm_qr = jnp.mean(grad_norm_qr_rows)
    grad_norm_qc = jnp.mean(grad_norm_qc_rows)
    lambda_scale = jnp.abs(lambda_tilde / nu_c)
    lambda_grad_norm_qc_rows = lambda_scale * grad_norm_qc_rows
    lambda_grad_norm_qc = jnp.mean(lambda_grad_norm_qc_rows)
    grad_ratio_rows = lambda_grad_norm_qc_rows / (grad_norm_qr_rows + 1e-8)
    grad_ratio = jnp.mean(grad_ratio_rows)
    cosine_grad_qr_qc_rows = jnp.sum(grad_qr * grad_qc, axis=-1) / (
        grad_norm_qr_rows * grad_norm_qc_rows + 1e-8
    )
    cosine_grad_qr_qc = jnp.mean(cosine_grad_qr_qc_rows)

    extras = _extras(transitions)
    reference = jnp.zeros_like(qc_value)
    hard_violation = jnp.asarray(
        extras.get("hard_violation", reference), dtype=jnp.float32
    )
    cost_sample = jnp.asarray(extras.get("cost", reference), dtype=jnp.float32)
    risk_mask = jnp.logical_or(hard_violation > 0.5, cost_sample > 0.0)
    risk_weight = risk_mask.astype(jnp.float32)
    risk_denom = jnp.maximum(jnp.sum(risk_weight), 1.0)

    def risky_mean(x: Array) -> Array:
        return jnp.sum(jnp.asarray(x, dtype=jnp.float32) * risk_weight) / risk_denom

    probes = {
        "alpha": alpha,
        "log_prob_mean": jnp.mean(sample.log_prob),
        "alpha_logprob_mean": alpha * jnp.mean(sample.log_prob),
        "gaussian_logp_mean": jnp.mean(sample.gaussian_logp),
        "sat_correction_mean": jnp.mean(sample.sat_correction),
        "log_std_mean": jnp.mean(sample.log_std),
        "f_term_mean": jnp.mean(f_value) / nu_f,
        "reward_actor_term_mean": jnp.mean(reward_actor_term),
        "qc_actor_mean": jnp.mean(qc_value),
        "qc_zero_action_mean": jnp.mean(qc_zero_action),
        "qc_neg_action_mean": jnp.mean(qc_neg_action),
        "qc_action_delta_mean": jnp.mean(qc_action_delta),
        "qc_action_gap_mean": jnp.mean(jnp.abs(qc_value - qc_zero_action)),
        "qc_action_delta_frac_pos": jnp.mean(
            (qc_action_delta > 0.0).astype(jnp.float32)
        ),
        "qc_actor_std": jnp.std(qc_value),
        "constraint_term_mean": jnp.mean(constraint_term),
        "grad_norm_qr_a": grad_norm_qr,
        "grad_norm_qc_a": grad_norm_qc,
        "lambda_grad_norm_qc_a": lambda_grad_norm_qc,
        "grad_ratio_cost_reward": grad_ratio,
        "cosine_grad_qr_qc": cosine_grad_qr_qc,
        "risk_condition_fraction": jnp.mean(risk_weight),
        "qc_actor_risky_mean": risky_mean(qc_value),
        "qc_action_delta_risky_mean": risky_mean(qc_action_delta),
        "grad_ratio_cost_reward_risky": risky_mean(grad_ratio_rows),
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
    cost_return_loss_weight: float = 0.0,
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
    td_loss = 0.5 * jnp.mean(jnp.square(qc_online - target))
    if "cost_return" in extras:
        cost_return = jnp.asarray(extras["cost_return"], dtype=jnp.float32)
    else:
        cost_return = target
    cost_return = jax.lax.stop_gradient(cost_return)
    return_loss = 0.5 * jnp.mean(jnp.square(qc_online - cost_return))
    loss = td_loss + cost_return_loss_weight * return_loss
    probes = {
        "cost_critic_loss": loss,
        "cost_critic_td_loss": td_loss,
        "cost_return_loss": return_loss,
        "mean_cost": jnp.mean(cost),
        "mean_qc": jnp.mean(qc_online),
        "mean_target": jnp.mean(target),
        "mean_cost_return": jnp.mean(cost_return),
        "qc_return_error": jnp.mean(qc_online - cost_return),
    }
    return loss, probes


def alpha_loss_fn(
    log_alpha: Array,
    log_prob: Array,
    action_size: int,
    entropy_param: float = 0.5,
) -> Array:
    """SAC entropy-temperature loss with a scaled target entropy."""

    alpha = jnp.exp(log_alpha)
    target_entropy = -jnp.asarray(
        entropy_param * action_size, dtype=jnp.asarray(log_prob).dtype
    )
    entropy_error = jax.lax.stop_gradient(-log_prob - target_entropy)
    return alpha * jnp.mean(entropy_error)
