"""PID-Lagrangian dual update with anti-windup."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class PIDState:
    """State carried by the PID-Lagrangian dual controller."""

    integral: jax.Array
    previous_error: jax.Array
    lambda_tilde: jax.Array


def make_pid_state() -> PIDState:
    """Creates a zeroed PID state."""

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return PIDState(integral=zero, previous_error=zero, lambda_tilde=zero)


def update_pid_lagrangian(
    state: PIDState,
    *,
    estimated_cost: jax.Array,
    budget: float,
    kp: float = 0.1,
    ki: float = 0.003,
    kd: float = 0.001,
    lambda_max: float = 100.0,
) -> PIDState:
    """Updates lambda_tilde and clamps the integral state for anti-windup."""

    error = jnp.asarray(estimated_cost, dtype=jnp.float32) - jnp.asarray(
        budget, dtype=jnp.float32
    )
    integral_limit = jnp.asarray(lambda_max / max(ki, 1e-8), dtype=jnp.float32)
    integral = jnp.clip(state.integral + error, -integral_limit, integral_limit)
    derivative = error - state.previous_error
    raw_lambda = kp * error + ki * integral + kd * derivative
    lambda_tilde = jnp.clip(raw_lambda, 0.0, lambda_max).astype(jnp.float32)
    return PIDState(
        integral=integral.astype(jnp.float32),
        previous_error=error.astype(jnp.float32),
        lambda_tilde=lambda_tilde,
    )
