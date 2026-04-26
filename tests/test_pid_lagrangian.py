import jax.numpy as jnp

from sr_cpo.pid_lagrangian import make_pid_state, update_pid_lagrangian


def test_pid_lagrangian_clamps_integral_state_for_anti_windup() -> None:
    state = make_pid_state()

    state = update_pid_lagrangian(
        state,
        estimated_cost=jnp.asarray(10_000.0, dtype=jnp.float32),
        budget=0.0,
        kp=0.0,
        ki=1.0,
        kd=0.0,
        lambda_max=5.0,
    )

    assert float(state.integral) == 5.0
    assert float(state.lambda_tilde) == 5.0


def test_pid_lagrangian_clamps_lambda_without_rewriting_integral_lower() -> None:
    state = make_pid_state()

    state = update_pid_lagrangian(
        state,
        estimated_cost=jnp.asarray(-10_000.0, dtype=jnp.float32),
        budget=0.0,
        kp=0.0,
        ki=1.0,
        kd=0.0,
        lambda_max=5.0,
    )

    assert float(state.integral) == -5.0
    assert float(state.lambda_tilde) == 0.0


def test_pid_lagrangian_derivative_uses_previous_error() -> None:
    state = make_pid_state()
    state = update_pid_lagrangian(
        state, estimated_cost=2.0, budget=1.0, kp=0.0, ki=0.0, kd=1.0
    )
    state = update_pid_lagrangian(
        state, estimated_cost=1.5, budget=1.0, kp=0.0, ki=0.0, kd=1.0
    )

    assert float(state.previous_error) == 0.5
    assert float(state.lambda_tilde) == 0.0
