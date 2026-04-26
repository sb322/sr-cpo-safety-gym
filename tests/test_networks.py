import jax
import jax.numpy as jnp

from sr_cpo.networks import Actor, CostCritic, GEncoder, SAEncoder


def _network_outputs(
    state: jnp.ndarray, action: jnp.ndarray, goal: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(0)
    key_sa, key_g, key_actor, key_qc = jax.random.split(key, 4)
    obs = jnp.concatenate([state, goal], axis=-1)

    sa_encoder = SAEncoder()
    sa_params = sa_encoder.init(key_sa, state, action)
    sa_repr = sa_encoder.apply(sa_params, state, action)

    g_encoder = GEncoder()
    g_params = g_encoder.init(key_g, goal)
    g_repr = g_encoder.apply(g_params, goal)

    actor = Actor(action_size=action.shape[-1])
    actor_params = actor.init(key_actor, obs)
    mean, log_std = actor.apply(actor_params, obs)

    cost_critic = CostCritic()
    qc_params = cost_critic.init(key_qc, state, action, goal)
    qc = cost_critic.apply(qc_params, state, action, goal)
    return sa_repr, g_repr, mean, log_std, qc


def _assert_finite_float32(value: jnp.ndarray, shape: tuple[int, ...]) -> None:
    assert value.shape == shape
    assert value.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(value)))


def test_network_forward_shapes_dtype_and_no_nan() -> None:
    batch_size = 4
    state_dim = 8
    action_dim = 2
    goal_dim = 2
    key = jax.random.PRNGKey(0)

    state = jax.random.normal(key, (batch_size, state_dim), dtype=jnp.float32)
    action = jax.random.normal(key, (batch_size, action_dim), dtype=jnp.float32)
    goal = jax.random.normal(key, (batch_size, goal_dim), dtype=jnp.float32)

    sa_repr, g_repr, mean, log_std, qc = _network_outputs(state, action, goal)
    _assert_finite_float32(sa_repr, (batch_size, 64))
    _assert_finite_float32(g_repr, (batch_size, 64))
    _assert_finite_float32(mean, (batch_size, action_dim))
    _assert_finite_float32(log_std, (batch_size, action_dim))
    _assert_finite_float32(qc, (batch_size,))


def test_actor_log_std_is_clipped() -> None:
    batch_size = 4
    action_dim = 2
    key = jax.random.PRNGKey(1)
    obs = jax.random.normal(key, (batch_size, 10), dtype=jnp.float32)
    actor = Actor(action_size=action_dim)
    actor_params = actor.init(key, obs)
    mean, log_std = actor.apply(actor_params, obs)
    _assert_finite_float32(mean, (batch_size, action_dim))
    _assert_finite_float32(log_std, (batch_size, action_dim))
    assert bool(jnp.all(log_std >= -5.0))
    assert bool(jnp.all(log_std <= 2.0))


def test_network_forward_is_finite_with_one_exact_zero_input_row() -> None:
    batch_size = 4
    state_dim = 8
    action_dim = 2
    goal_dim = 2
    key = jax.random.PRNGKey(2)

    state = jax.random.normal(key, (batch_size, state_dim), dtype=jnp.float32)
    action = jax.random.normal(key, (batch_size, action_dim), dtype=jnp.float32)
    goal = jax.random.normal(key, (batch_size, goal_dim), dtype=jnp.float32)
    state = state.at[0].set(jnp.zeros((state_dim,), dtype=jnp.float32))
    action = action.at[0].set(jnp.zeros((action_dim,), dtype=jnp.float32))
    goal = goal.at[0].set(jnp.zeros((goal_dim,), dtype=jnp.float32))

    sa_repr, g_repr, mean, log_std, qc = _network_outputs(state, action, goal)
    assert bool(jnp.all(jnp.isfinite(sa_repr[0])))
    assert bool(jnp.all(jnp.isfinite(g_repr[0])))
    assert bool(jnp.all(jnp.isfinite(mean[0])))
    assert bool(jnp.all(jnp.isfinite(log_std[0])))
    assert bool(jnp.all(jnp.isfinite(qc[0])))
    _assert_finite_float32(sa_repr, (batch_size, 64))
    _assert_finite_float32(g_repr, (batch_size, 64))
    _assert_finite_float32(mean, (batch_size, action_dim))
    _assert_finite_float32(log_std, (batch_size, action_dim))
    _assert_finite_float32(qc, (batch_size,))
