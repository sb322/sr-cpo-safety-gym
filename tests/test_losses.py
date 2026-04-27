from functools import partial

import jax
import jax.numpy as jnp

from sr_cpo.env_wrappers import Transition
from sr_cpo.losses import (
    actor_loss_fn,
    alpha_loss_fn,
    contrastive_logits,
    cost_critic_loss_fn,
    critic_loss_fn,
    sample_tanh_gaussian,
)
from sr_cpo.networks import Actor, CostCritic, GEncoder, SAEncoder


def _assert_finite_tree(tree: object) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    assert leaves
    for leaf in leaves:
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _critic_setup(
    *,
    zero_sa_row: bool = False,
    zero_g_row: bool = False,
) -> tuple[dict[str, object], Transition, SAEncoder, GEncoder, jax.Array]:
    batch_size = 4
    state_dim = 6
    action_dim = 2
    goal_dim = 3
    key = jax.random.PRNGKey(0)
    state_key, action_key, goal_key, sa_key, g_key = jax.random.split(key, 5)

    state = jax.random.normal(state_key, (batch_size, state_dim), dtype=jnp.float32)
    action = jax.random.normal(
        action_key, (batch_size, action_dim), dtype=jnp.float32
    )
    goal = jax.random.normal(goal_key, (batch_size, goal_dim), dtype=jnp.float32)
    if zero_sa_row:
        state = state.at[0].set(jnp.zeros((state_dim,), dtype=jnp.float32))
        action = action.at[0].set(jnp.zeros((action_dim,), dtype=jnp.float32))
    if zero_g_row:
        goal = goal.at[0].set(jnp.zeros((goal_dim,), dtype=jnp.float32))

    transition = Transition(
        observation=state,
        action=action,
        reward=jnp.zeros((batch_size,), dtype=jnp.float32),
        discount=jnp.ones((batch_size,), dtype=jnp.float32),
        extras={"goal": goal},
    )
    sa_encoder = SAEncoder()
    g_encoder = GEncoder()
    params = {
        "sa_encoder": sa_encoder.init(sa_key, state, action),
        "g_encoder": g_encoder.init(g_key, goal),
    }
    return params, transition, sa_encoder, g_encoder, goal


def _actor_setup(
    *,
    zero_row: bool = False,
) -> tuple[
    dict[str, object],
    dict[str, object],
    object,
    Transition,
    Actor,
    SAEncoder,
    GEncoder,
    CostCritic,
]:
    batch_size = 4
    state_dim = 6
    action_dim = 2
    goal_dim = 3
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    state = jax.random.normal(keys[0], (batch_size, state_dim), dtype=jnp.float32)
    action = jax.random.normal(keys[1], (batch_size, action_dim), dtype=jnp.float32)
    goal = jax.random.normal(keys[2], (batch_size, goal_dim), dtype=jnp.float32)
    if zero_row:
        state = state.at[0].set(jnp.zeros((state_dim,), dtype=jnp.float32))
        goal = goal.at[0].set(jnp.zeros((goal_dim,), dtype=jnp.float32))
    next_state = state + jnp.asarray(0.05, dtype=jnp.float32)
    cost = jnp.asarray([0.0, 1.0, 0.0, 0.25], dtype=jnp.float32)

    transition = Transition(
        observation=state,
        action=action,
        reward=jnp.zeros((batch_size,), dtype=jnp.float32),
        discount=jnp.ones((batch_size,), dtype=jnp.float32),
        extras={
            "goal": goal,
            "next_state": next_state,
            "cost": cost,
        },
    )
    actor = Actor(action_size=action_dim)
    sa_encoder = SAEncoder()
    g_encoder = GEncoder()
    cost_critic = CostCritic()
    actor_params = actor.init(keys[3], state, goal)
    critic_params = {
        "sa_encoder": sa_encoder.init(keys[4], state, action),
        "g_encoder": g_encoder.init(keys[5], goal),
    }
    cost_critic_params = cost_critic.init(keys[6], state, action, goal)
    return (
        actor_params,
        critic_params,
        cost_critic_params,
        transition,
        actor,
        sa_encoder,
        g_encoder,
        cost_critic,
    )


def test_critic_loss_forward_finite_on_random_batch() -> None:
    params, transition, sa_encoder, g_encoder, _ = _critic_setup()

    loss, probes = critic_loss_fn(
        params,
        transition,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        tau=0.1,
        rho=0.1,
    )

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    _assert_finite_tree(probes)


def test_critic_loss_backward_finite_with_exact_zero_prenorm_rows() -> None:
    params, transition, sa_encoder, g_encoder, _ = _critic_setup(
        zero_sa_row=True, zero_g_row=True
    )
    loss_fn = partial(
        critic_loss_fn,
        transitions=transition,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        tau=0.1,
        rho=0.1,
    )

    (loss, probes), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    assert bool(jnp.isfinite(loss))
    _assert_finite_tree(probes)
    _assert_finite_tree(grads)
    assert probes["sa_norm_min"] == 0.0
    assert probes["g_norm_min"] == 0.0


def test_contrastive_logits_are_temperature_bounded_after_row_l2() -> None:
    tau = 0.2
    sa_repr = jnp.array(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 2.0]], dtype=jnp.float32
    )
    g_repr = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, -2.0, 1.0]], dtype=jnp.float32
    )

    logits = contrastive_logits(sa_repr, g_repr, tau=tau)

    assert bool(jnp.all(jnp.isfinite(logits)))
    assert bool(jnp.all(jnp.abs(logits) <= (1.0 / tau) + 1e-6))


def test_actor_loss_forward_and_grad_finite_on_random_batch() -> None:
    (
        actor_params,
        critic_params,
        cost_critic_params,
        transition,
        actor,
        sa_encoder,
        g_encoder,
        cost_critic,
    ) = _actor_setup()
    tau = 0.1
    loss_fn = partial(
        actor_loss_fn,
        critic_params=critic_params,
        cost_critic_params=cost_critic_params,
        transitions=transition,
        key=jax.random.PRNGKey(7),
        actor=actor,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        cost_critic=cost_critic,
        log_alpha=jnp.array(0.0, dtype=jnp.float32),
        lambda_tilde=jnp.array(0.5, dtype=jnp.float32),
        tau=tau,
        nu_f=1.0,
        nu_c=1.0,
    )

    (loss, probes), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)

    assert bool(jnp.isfinite(loss))
    _assert_finite_tree(probes)
    _assert_finite_tree(grads)
    assert bool(jnp.abs(probes["f_term_mean"]) <= (1.0 / tau) + 1e-6)
    assert "constraint_term_mean" in probes
    assert bool(jnp.isfinite(probes["constraint_term_mean"]))
    assert "qc_actor_mean" in probes
    assert bool(jnp.isfinite(probes["qc_actor_mean"]))


def test_actor_loss_forward_and_grad_finite_with_zero_row_inputs() -> None:
    (
        actor_params,
        critic_params,
        cost_critic_params,
        transition,
        actor,
        sa_encoder,
        g_encoder,
        cost_critic,
    ) = _actor_setup(zero_row=True)
    tau = 0.1
    loss_fn = partial(
        actor_loss_fn,
        critic_params=critic_params,
        cost_critic_params=cost_critic_params,
        transitions=transition,
        key=jax.random.PRNGKey(8),
        actor=actor,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        cost_critic=cost_critic,
        log_alpha=jnp.array(0.0, dtype=jnp.float32),
        lambda_tilde=jnp.array(0.5, dtype=jnp.float32),
        tau=tau,
        nu_f=1.0,
        nu_c=1.0,
    )

    (loss, probes), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)

    assert bool(jnp.isfinite(loss))
    _assert_finite_tree(probes)
    _assert_finite_tree(grads)
    assert bool(jnp.abs(probes["f_term_mean"]) <= (1.0 / tau) + 1e-6)
    assert bool(jnp.isfinite(probes["constraint_term_mean"]))


def test_actor_sampling_respects_log_std_clipping() -> None:
    actor_params, _, _, transition, actor, *_ = _actor_setup()
    goal = transition.extras["goal"]

    sample = sample_tanh_gaussian(
        actor,
        actor_params,
        transition.observation,
        goal,
        jax.random.PRNGKey(9),
    )

    assert bool(jnp.all(sample.log_std >= -5.0))
    assert bool(jnp.all(sample.log_std <= 2.0))


def test_cost_critic_loss_forward_and_grad_finite() -> None:
    (
        actor_params,
        _,
        cost_critic_params,
        transition,
        actor,
        _,
        _,
        cost_critic,
    ) = _actor_setup()
    loss_fn = partial(
        cost_critic_loss_fn,
        cost_critic_target_params=cost_critic_params,
        actor_params=actor_params,
        transitions=transition,
        key=jax.random.PRNGKey(10),
        actor=actor,
        cost_critic=cost_critic,
        gamma_c=0.99,
    )

    (loss, probes), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        cost_critic_params
    )

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    _assert_finite_tree(probes)
    _assert_finite_tree(grads)


def test_alpha_loss_forward_and_grad_finite() -> None:
    log_prob = jnp.asarray([-1.0, -0.5, -2.0, -1.5], dtype=jnp.float32)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)

    loss, grad = jax.value_and_grad(alpha_loss_fn)(log_alpha, log_prob, 2)

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(grad))
