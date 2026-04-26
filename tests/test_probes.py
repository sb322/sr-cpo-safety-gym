from __future__ import annotations

import jax
import jax.numpy as jnp

from sr_cpo.probes import (
    _first_one_idx,
    _grads_global_norm,
    _grads_have_nan,
    _params_have_nan,
)


def test_gradient_probe_helpers_detect_nonfinite_and_norm() -> None:
    grads = {
        "a": jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        "b": {"c": jnp.asarray([12.0], dtype=jnp.float32)},
    }
    assert float(_grads_have_nan(grads)) == 0.0
    assert float(_grads_global_norm(grads)) == 13.0

    bad_grads = {"a": jnp.asarray([1.0, jnp.inf], dtype=jnp.float32)}
    assert float(_grads_have_nan(bad_grads)) == 1.0


def test_parameter_probe_detects_nan() -> None:
    params = {"w": jnp.asarray([[1.0, jnp.nan]], dtype=jnp.float32)}
    assert float(_params_have_nan(params)) == 1.0


def test_grads_global_norm_stays_finite_for_large_finite_values() -> None:
    grads = {
        "huge": jnp.asarray([1.0e20, -1.0e20], dtype=jnp.float32),
        "small": jnp.asarray([3.0, 4.0], dtype=jnp.float32),
    }

    norm = _grads_global_norm(grads)

    assert bool(jnp.isfinite(norm))
    assert bool(jnp.allclose(norm / 1.0e20, jnp.sqrt(2.0), rtol=1e-6))


def test_first_one_idx_returns_first_flat_hit_or_negative_one() -> None:
    flags = jnp.asarray([[0, 0, 0], [0, 1, 1]], dtype=jnp.int32)
    assert int(_first_one_idx(flags)) == 4
    assert int(_first_one_idx(jnp.zeros((2, 3), dtype=jnp.int32))) == -1


def test_probe_helpers_are_jittable() -> None:
    @jax.jit
    def probe(flags: jax.Array) -> tuple[jax.Array, jax.Array]:
        grads = {"g": flags.astype(jnp.float32)}
        return _first_one_idx(flags), _grads_have_nan(grads)

    idx, has_nan = probe(jnp.asarray([0, 1, 0], dtype=jnp.int32))
    assert int(idx) == 1
    assert float(has_nan) == 0.0
