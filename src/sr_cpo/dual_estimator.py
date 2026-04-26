"""Critic-based cost estimator for the dual update."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


def estimate_discounted_cost(
    *,
    cost_critic: Any,
    cost_critic_params: Any,
    actor: Any,
    actor_params: Any,
    initial_states: jax.Array,
    goals: jax.Array,
    key: jax.Array,
    gamma_c: float,
    num_action_samples: int = 4,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Estimates J_c = (1 - gamma) E[Q_c(s0, a, g)] by action Monte Carlo."""

    mean, log_std = actor.apply(actor_params, initial_states, goals)
    keys = jax.random.split(key, num_action_samples)

    def sample_q(sample_key: jax.Array) -> jax.Array:
        noise = jax.random.normal(sample_key, shape=mean.shape)
        action = nn.tanh(mean + jnp.exp(log_std) * noise)
        return cost_critic.apply(cost_critic_params, initial_states, action, goals)

    q_values = jax.vmap(sample_q)(keys)
    v_values = jnp.mean(q_values, axis=0)
    estimate = (1.0 - gamma_c) * jnp.mean(v_values)
    probes = {
        "dual_jc": estimate,
        "dual_qc_mean": jnp.mean(v_values),
        "dual_qc_min": jnp.min(v_values),
        "dual_qc_max": jnp.max(v_values),
    }
    return estimate.astype(jnp.float32), probes
