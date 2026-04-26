from functools import partial

import jax
import jax.numpy as jnp

from sr_cpo.env_wrappers import Transition
from sr_cpo.losses import contrastive_logits, critic_loss_fn
from sr_cpo.networks import GEncoder, SAEncoder


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
