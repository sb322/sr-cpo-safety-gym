"""Flax network modules for SR-CPO.

The encoders are pure feature extractors. Row-L2 normalization for contrastive
energies belongs in :mod:`sr_cpo.losses`, where the autograd-safe normalization
contract can be tested directly.
"""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

KERNEL_INIT = nn.initializers.lecun_normal()
BIAS_INIT = nn.initializers.zeros


def _concat(xs: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Concatenates float32 inputs along the feature axis."""
    return jnp.concatenate([jnp.asarray(x, dtype=jnp.float32) for x in xs], axis=-1)


class WangResidualBlock(nn.Module):
    """Pre-norm Wang-style residual MLP block."""

    width: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.width, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT)(x)
        x = nn.swish(x)
        x = nn.Dense(self.width, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT)(x)
        return residual + x


class ResidualTower(nn.Module):
    """Input projection followed by Wang residual blocks."""

    width: int = 256
    num_blocks: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float32)
        x = nn.Dense(self.width, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT)(x)
        x = nn.swish(nn.LayerNorm()(x))
        for _ in range(self.num_blocks):
            x = WangResidualBlock(width=self.width)(x)
        return x


class SAEncoder(nn.Module):
    """State-action encoder phi(s, a) -> R^K."""

    width: int = 256
    num_blocks: int = 4
    latent_dim: int = 64

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = _concat((state, action))
        x = ResidualTower(width=self.width, num_blocks=self.num_blocks)(x)
        return nn.Dense(
            self.latent_dim, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT
        )(x)


class GEncoder(nn.Module):
    """Goal encoder psi(g) -> R^K."""

    width: int = 256
    num_blocks: int = 4
    latent_dim: int = 64

    @nn.compact
    def __call__(self, goal: jnp.ndarray) -> jnp.ndarray:
        x = ResidualTower(width=self.width, num_blocks=self.num_blocks)(goal)
        return nn.Dense(
            self.latent_dim, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT
        )(x)


class Actor(nn.Module):
    """Tanh-Gaussian actor parameter network."""

    action_size: int
    width: int = 256
    num_blocks: int = 4
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self, state_or_obs: jnp.ndarray, goal: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = _concat((state_or_obs, goal)) if goal is not None else state_or_obs
        x = ResidualTower(width=self.width, num_blocks=self.num_blocks)(x)
        x = nn.Dense(
            2 * self.action_size, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT
        )(x)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class CostCritic(nn.Module):
    """Cost critic Q_c(s, a, g) -> scalar."""

    width: int = 256
    num_blocks: int = 4

    @nn.compact
    def __call__(
        self, state: jnp.ndarray, action: jnp.ndarray, goal: jnp.ndarray
    ) -> jnp.ndarray:
        x = _concat((state, action, goal))
        x = ResidualTower(width=self.width, num_blocks=self.num_blocks)(x)
        q_value = nn.Dense(1, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT)(x)
        return jnp.squeeze(q_value, axis=-1)


SA_encoder = SAEncoder
G_encoder = GEncoder
