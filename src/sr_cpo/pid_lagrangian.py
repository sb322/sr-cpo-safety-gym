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
    integral_min: float = -100.0,
    integral_max: float = 100.0,
    integral_decay: float = 1.0,
) -> PIDState:
    """Updates lambda_tilde with anti-windup clamp on the integral state."""

    error = jnp.asarray(estimated_cost, dtype=jnp.float32) - jnp.asarray(
        budget, dtype=jnp.float32
    )
    decay = jnp.asarray(integral_decay, dtype=jnp.float32)
    carried_integral = jnp.where(error < 0.0, state.integral * decay, state.integral)
    integral = jnp.clip(
        carried_integral + error,
        jnp.asarray(integral_min, dtype=jnp.float32),
        jnp.asarray(integral_max, dtype=jnp.float32),
    )
    derivative = error - state.previous_error
    raw_lambda = kp * error + ki * integral + kd * derivative
    lambda_tilde = jnp.maximum(raw_lambda, 0.0).astype(jnp.float32)
    return PIDState(
        integral=integral.astype(jnp.float32),
        previous_error=error.astype(jnp.float32),
        lambda_tilde=lambda_tilde,
    )
