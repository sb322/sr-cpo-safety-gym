"""Loss functions and per-step probes for SR-CPO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Any]


def row_l2_normalize(x: Array, eps: float = 1e-12) -> tuple[Array, Array]:
    """Normalizes rows with epsilon inside sqrt for finite backward passes."""

    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm, norm


def contrastive_logits(sa_repr: Array, g_repr: Array, tau: float) -> Array:
    """Computes cosine InfoNCE logits from unnormalized representations."""

    sa_hat, _ = row_l2_normalize(sa_repr)
    g_hat, _ = row_l2_normalize(g_repr)
    return jnp.einsum("ik,jk->ij", sa_hat, g_hat) / tau


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

    sa_hat, _ = row_l2_normalize(sa_repr)
    g_hat, _ = row_l2_normalize(g_repr)
    logits = jnp.einsum("ik,jk->ij", sa_hat, g_hat) / tau
    logsumexp = jax.nn.logsumexp(logits, axis=1)
    diag = jnp.diag(logits)
    loss = -jnp.mean(diag - logsumexp) + rho * jnp.mean(logsumexp**2)

    eye = jnp.eye(logits.shape[0], dtype=logits.dtype)
    logits_pos = jnp.sum(logits * eye) / jnp.sum(eye)
    logits_neg = jnp.sum(logits * (1.0 - eye)) / jnp.sum(1.0 - eye)
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0]))

    probes = {
        "nan_obs": jnp.any(jnp.isnan(observation)).astype(jnp.float32),
        "nan_sa": jnp.any(jnp.isnan(sa_repr)).astype(jnp.float32),
        "nan_g": jnp.any(jnp.isnan(g_repr)).astype(jnp.float32),
        "nan_logits": jnp.any(jnp.isnan(logits)).astype(jnp.float32),
        "sa_norm_min": sa_norm_min,
        "g_norm_min": g_norm_min,
        "sa_repr_norm": sa_repr_norm,
        "g_repr_norm": g_repr_norm,
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "accuracy": accuracy.astype(jnp.float32),
    }
    return loss, probes
