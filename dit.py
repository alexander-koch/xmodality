import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random
from typing import Union, Any, Optional
from einops import rearrange
import math
#from talking_heads import talking_heads_dot_product_attention
from functools import partial
import scipy
import pickle
#from fourier import FourierTransform
from uvit import Wavelet

# previous model:
# no bias in swiglu
# no scale in rmsnorm

def pair(x: Union[int, tuple[int, int]]) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)

def sinusoidal_embedding_2d(h: int, w: int, dim: int, temperature: int = 10000) -> jax.Array:
    y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    omega = jnp.arange(dim//4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = jnp.ravel(y)[:, None] * omega[None, :]
    x = jnp.ravel(x)[:, None] * omega[None, :]
    pos = jnp.concatenate((
        jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=1)
    return jnp.expand_dims(pos, 0)

def timestep_embedding(t: jax.Array, dim: int, max_period: int = 10000) -> jax.Array:
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate((jnp.cos(args), jnp.sin(args)), axis=-1)
    if dim % 2:
        embedding = jnp.concatenate(
            (embedding, jnp.zeros_like(embedding[:, :1])), axis=-1
        )
    return embedding


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1 + scale[:, jnp.newaxis]) + shift[:, jnp.newaxis]

class SwiGLU(nn.Module):
    dim: int
    mlp_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x_gate = nn.Dense(
            features=self.mlp_dim,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.Dense(
            features=self.mlp_dim,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.silu(x_gate) * x
        x = nn.Dense(
            features=self.dim,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class DiTBlock(nn.Module):
    hidden_size: int
    num_heads: int = 8
    mlp_ratio: int = 4
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        mlp_dim = self.mlp_ratio * self.hidden_size

        adaln = nn.Sequential(
            [
                nn.silu,
                nn.Dense(
                    6 * self.hidden_size,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype
                ),
            ]
        )(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            adaln, 6, axis=-1
        )

        x_hat = modulate(nn.RMSNorm()(x), shift_msa, scale_msa)
        #x_hat = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, attention_fn=talking_heads_dot_product_attention, dtype=self.dtype)(x_hat)
        x_hat = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dtype=self.dtype)(x_hat)
        #x_hat = FourierTransform(dim=self.hidden_size, seq_len=256)(x_hat)
        x = x + gate_msa[:, jnp.newaxis] * x_hat

        x_hat = modulate(nn.RMSNorm()(x), shift_mlp, scale_mlp)
        x_hat = SwiGLU(dim=self.hidden_size, mlp_dim=mlp_dim, dtype=self.dtype)(x_hat)
        x = x + gate_mlp[:, jnp.newaxis] * x_hat
        return x



class FinalLayer(nn.Module):
    hidden_size: int
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    channels: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        h, w = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        adaln = nn.Sequential(
            [
                nn.silu,
                nn.Dense(
                    2 * self.hidden_size,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype
                ),
            ]
        )(c)
        shift, scale = jnp.split(adaln, 2, axis=-1)

        x = modulate(nn.RMSNorm()(x), shift, scale)
        #x = nn.Dense(
        #    features=self.patch_size[0] * self.patch_size[1] * self.channels,
        #    #kernel_init=nn.initializers.zeros,
        #    kernel_init=nn.initializers.glorot_uniform(),
        #    bias_init=nn.initializers.zeros,
        #    dtype=self.dtype
        #)(x)

        #x = rearrange(
        #    x,
        #    "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
        #    h=h,
        #    w=w,
        #    p1=self.patch_size[0],
        #    p2=self.patch_size[1],
        #)

        x = nn.Dense(
            features=2048,
            #kernel_init=nn.initializers.zeros,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype
        )(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=self.image_size[0], w=self.image_size[1])
        return x

class DiT(nn.Module):
    patch_size: Union[int, tuple[int, int]] = 16
    in_channels: int = 1
    hidden_size: int = 1024
    depth: int = 16
    num_heads: int = 8
    mlp_ratio: int = 4
    frequency_embedding_size: int = 256
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x: jax.Array, time: jax.Array, condition: Optional[jax.Array] = None
    ) -> jax.Array:
        patch_size = pair(self.patch_size)
        #image_size = (x.shape[1], x.shape[2])
        #out_dim = 1

        if condition is not None:
            x = jnp.concatenate((x, condition), axis=-1)

        # DWT deconstruction
        wavelet = Wavelet(channels=x.shape[-1], levels=5, dtype=self.dtype)
        x = wavelet.encode(x)
        out_dim = x.shape[-1]
        image_size = (x.shape[1], x.shape[2])

        #print("dwt:", x.shape)
        
        # Patching
        #x = rearrange(
        #    x,
        #    "b (h p1) (w p2) c -> b h w (p1 p2 c)",
        #    p1=patch_size[0],
        #    p2=patch_size[1],
        #)
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype
        )(x)
        x = nn.LayerNorm()(x)

        #print("patch:", x.shape)

        # Add pos emb
        h,w,d = x.shape[1:]
        x = rearrange(x, "b h w c -> b (h w) c")
        pos_emb = sinusoidal_embedding_2d(h,w,d)
        x = x + pos_emb

        # Embed time
        t_freq = timestep_embedding(time, self.frequency_embedding_size)
        t_emb = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_size,
                    kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                    dtype=self.dtype
                ),
                nn.silu,
                nn.Dense(
                    self.hidden_size,
                    kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                    dtype=self.dtype
                ),
            ]
        )(t_freq)

        # Transformer blocks
        l = x.shape[1]
        for _ in range(self.depth):
            x = DiTBlock(
                self.hidden_size, self.num_heads, self.mlp_ratio, dtype=self.dtype
            )(x, t_emb)

        # Unpatching
        #x = FinalLayer(self.hidden_size, image_size, patch_size, out_dim, dtype=self.dtype)(
        #    x, t_emb
        #)
        #print("unpatch:", x.shape)

        x = FinalLayer(self.hidden_size, image_size, patch_size, out_dim, dtype=self.dtype)(
            x, t_emb
        )

        #print("final", x.shape)
        
        # DWT reconstruction + merge to one image
        x = wavelet.decode(x)
        x = nn.Dense(features=1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros, dtype=self.dtype)(x)
        #print("out:", x.shape)
        return x
