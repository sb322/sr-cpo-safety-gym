import jax
import jax.numpy as jnp

from sr_cpo.dual_estimator import estimate_discounted_cost
from sr_cpo.networks import Actor, CostCritic


def test_critic_based_dual_estimator_shape_and_finiteness() -> None:
    batch_size = 5
    state_dim = 6
    goal_dim = 3
    action_dim = 2
    key = jax.random.PRNGKey(0)
    state_key, goal_key, actor_key, critic_key, estimate_key = jax.random.split(key, 5)
    states = jax.random.normal(state_key, (batch_size, state_dim), dtype=jnp.float32)
    goals = jax.random.normal(goal_key, (batch_size, goal_dim), dtype=jnp.float32)
    actor = Actor(action_size=action_dim)
    cost_critic = CostCritic()
    actor_params = actor.init(actor_key, states, goals)
    action = jnp.zeros((batch_size, action_dim), dtype=jnp.float32)
    cost_critic_params = cost_critic.init(critic_key, states, action, goals)

    estimate, probes = estimate_discounted_cost(
        cost_critic=cost_critic,
        cost_critic_params=cost_critic_params,
        actor=actor,
        actor_params=actor_params,
        initial_states=states,
        goals=goals,
        key=estimate_key,
        gamma_c=0.99,
        num_action_samples=3,
    )

    assert estimate.shape == ()
    assert estimate.dtype == jnp.float32
    assert bool(jnp.isfinite(estimate))
    for leaf in jax.tree_util.tree_leaves(probes):
        assert bool(jnp.all(jnp.isfinite(leaf)))
