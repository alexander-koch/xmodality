import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random
from typing import Union, Any
from einops import rearrange
from utils import triple


class MLP(nn.Module):
    dim: int
    mlp_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(features=self.mlp_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            #bias_init=nn.initializers.zeros,
            use_bias=False,
            dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.dim,
            kernel_init=nn.initializers.glorot_uniform(),
            #bias_init=nn.initializers.zeros,
            use_bias=False,
            dtype=self.dtype)(x)
        return x

class LayerScale(nn.Module):
    dim: int
    init_value: float = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma_init = nn.initializers.ones
        gamma = self.param("gamma", gamma_init, (self.dim,), self.dtype)
        return x * gamma

class Transformer(nn.Module):
    dim: int
    mlp_dim: int
    num_heads: int
    depth: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.depth):
            x_hat = nn.RMSNorm()(x)
            x_hat = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                    kernel_init=nn.initializers.glorot_uniform(),
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype)(x_hat)
            #x = x + LayerScale(self.dim, dtype=self.dtype)(x_hat)
            x = x + x_hat

            x_hat = nn.RMSNorm()(x)
            x_hat = MLP(dim=self.dim, mlp_dim=self.mlp_dim, dtype=self.dtype)(x_hat)
            #x = x + LayerScale(self.dim, dtype=self.dtype)(x_hat)
            x = x + x_hat
        return nn.RMSNorm()(x)

class VisionTransformer(nn.Module):
    image_size: Union[int, tuple[int, int, int]]
    patch_size: Union[int, tuple[int, int, int]]
    dim: int
    depth: int
    num_heads: int
    num_classes: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        patch_size = triple(self.patch_size)

        x = rearrange(x, "b (d p1) (h p2) (w p3) c -> b (d h w) (p1 p2 p3 c)", p1=patch_size[0], p2=patch_size[1], p3=patch_size[2])
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype)(x)
        x = nn.LayerNorm()(x)

        pos_emb_init = nn.initializers.truncated_normal(stddev=0.02)
        pos_emb_shape = (1, x.shape[1], x.shape[2])
        pos_emb = self.param("pos_emb", pos_emb_init, pos_emb_shape, self.dtype)
        x = x + pos_emb

        x = Transformer(
            dim=self.dim,
            mlp_dim=self.dim * 4,
            num_heads=self.num_heads,
            depth=self.depth,
            dtype=self.dtype
        )(x)
        x = jnp.mean(x, axis=1)
        x = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype
        )(x)
        return x
