"""Diagnostic probe helpers for SR-CPO."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _numeric_leaves(tree: Any) -> list[jax.Array]:
    """Returns numeric JAX leaves, skipping ``None`` and metadata leaves."""

    leaves: list[jax.Array] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        if leaf is None:
            continue
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.number):
            leaves.append(arr)
    return leaves


def _tree_has_nonfinite(tree: Any) -> jax.Array:
    leaves = _numeric_leaves(tree)
    if not leaves:
        return jnp.asarray(False)
    flags = [jnp.any(~jnp.isfinite(leaf)) for leaf in leaves]
    return jnp.any(jnp.stack(flags))


def _grads_have_nan(grads: Any) -> jax.Array:
    """Returns a scalar flag for NaN/Inf in a gradient pytree."""

    return _tree_has_nonfinite(grads).astype(jnp.float32)


def _params_have_nan(params: Any) -> jax.Array:
    """Returns a scalar flag for NaN/Inf in a parameter pytree."""

    return _tree_has_nonfinite(params).astype(jnp.float32)


def _grads_global_norm(grads: Any) -> jax.Array:
    """Computes the global L2 norm of a gradient pytree."""

    leaves = _numeric_leaves(grads)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    sq_sum = sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves)
    return jnp.sqrt(sq_sum).astype(jnp.float32)


def _first_one_idx(flags: Any) -> jax.Array:
    """Returns the first flattened index with a nonzero flag, or -1."""

    arr = jnp.ravel(jnp.asarray(flags))
    hit = arr != 0
    return jnp.where(jnp.any(hit), jnp.argmax(hit), -1).astype(jnp.int32)
