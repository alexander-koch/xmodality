import jax
import jax.numpy as jnp
import math
from einops import rearrange, repeat
from typing import NamedTuple, Optional, Any, Union
from functools import partial
from flax import linen as nn
from vit import Transformer

def pair(x: Union[int, tuple[int, int, int]]):
    return x if isinstance(x, tuple) else (x,x)

class SinusoidalPositionEmbeddings(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, time: jax.Array) -> jax.Array:
        half_dim = self.dim // 2
        embeddings_scale = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings_scale)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate((jnp.sin(embeddings), jnp.cos(embeddings)), axis=1)
        return embeddings


class Block(nn.Module):
    dim_out: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x: jax.Array, scale_shift: Optional[list[jax.Array]] = None
    ) -> jax.Array:
        x = nn.Conv(
            features=self.dim_out,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
        )(x)
        x = nn.RMSNorm()(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return nn.gelu(x)


class ResNetBlock(nn.Module):
    dim: int
    dim_out: int
    time_emb_dim: Optional[int] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: Optional[jax.Array] = None) -> jax.Array:
        scale_shift = None
        if time_emb is not None:
            t = nn.Sequential(
                [
                    nn.Dense(
                        features=self.dim_out * 2,
                        kernel_init=nn.initializers.glorot_uniform(),
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                    ),
                    nn.RMSNorm(),
                    nn.gelu,
                    nn.Dense(
                        features=self.dim_out * 2,
                        kernel_init=nn.initializers.glorot_uniform(),
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                    ),
                ]
            )(time_emb)
            t = rearrange(t, "b c -> b 1 1 c")
            scale_shift = jnp.split(t, 2, axis=-1)

        h = Block(dim_out=self.dim_out, dtype=self.dtype)(
            x, scale_shift=scale_shift
        )
        h = Block(dim_out=self.dim_out, dtype=self.dtype)(h)
        if self.dim != self.dim_out:
            h = h + nn.Dense(features=self.dim_out, kernel_init=nn.initializers.glorot_uniform(), bias_init=nn.initializers.zeros, dtype=self.dtype)(x)
        return h

class PixelShuffleUpsample(nn.Module):
    dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Sequential(
            [
                nn.Dense(
                    features=self.dim * 4,
                    kernel_init=nn.initializers.glorot_uniform(),
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype,
                ),
                nn.gelu,
            ]
        )(x)
        x = rearrange(x, "b h w (c s1 s2) -> b (h s1) (w s2) c", s1=2, s2=2)
        return x


class PixelShuffleDownsample(nn.Module):
    dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = rearrange(x, "b (h s1) (w s2) c -> b h w (c s1 s2)", s1=2, s2=2)
        x = nn.Dense(
            features=self.dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
        )(x)
        return x


class UViT(nn.Module):
    dim: int
    dim_mults: tuple[int, ...] = (1, 2, 4)
    channels: int = 3
    vit_num_heads: int = 4
    vit_depth: int = 6
    patch_size: Union[int, tuple[int, int]] = 16
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        time: Optional[jax.Array] = None,
        condition: Optional[jax.Array] = None,
    ) -> jax.Array:
        dims = [self.dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        patch_height, patch_width = pair(self.patch_size)

        if time is not None:
            time_emb_dim = self.dim * 4
            time_embed = nn.Sequential(
                [
                    SinusoidalPositionEmbeddings(self.dim),
                    nn.Dense(
                        features=time_emb_dim,
                        kernel_init=nn.initializers.glorot_uniform(),
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                    ),
                    nn.RMSNorm(),
                    nn.gelu,
                    nn.Dense(
                        features=time_emb_dim,
                        kernel_init=nn.initializers.glorot_uniform(),
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                    ),
                    nn.RMSNorm(),
                    nn.gelu,
                ]
            )(time)
        else:
            time_emb_dim = None
            time_embed = None

        if condition is not None:
            x = jnp.concatenate((condition, x), axis=-1)
        
        # Patching
        sh = x.shape[1:3]
        x = rearrange(x, "b (h p1) (w p2) c -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width)
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            features=self.dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
        )(x)
        x = nn.LayerNorm()(x)
        residual = x

        # Downsampling
        h = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            x = ResNetBlock(
                dim=dim_in,
                dim_out=dim_out,
                time_emb_dim=time_emb_dim,
                dtype=self.dtype,
            )(x, time_embed)
            x = ResNetBlock(
                dim=dim_out,
                dim_out=dim_out,
                time_emb_dim=time_emb_dim,
                dtype=self.dtype,
            )(x, time_embed)
            h.append(x)

            if not is_last:
                x = PixelShuffleDownsample(dim=dim_out, dtype=self.dtype)(x)

        # Transformer middle part
        mid_dim = dims[-1]

        img_h, img_w = x.shape[1], x.shape[2]
        seq_len = img_h * img_w
        x = rearrange(x, "b h w c -> b (h w) c")

        pos_emb_init = nn.initializers.truncated_normal(stddev=0.02)
        pos_emb_shape = (1, seq_len, mid_dim)
        pos_emb = self.param("pos_emb", pos_emb_init, pos_emb_shape, self.dtype)
        x = x + pos_emb

        x = Transformer(
            dim=mid_dim,
            mlp_dim=mid_dim * 4,
            num_heads=self.vit_num_heads,
            depth=self.vit_depth,
            dtype=self.dtype
        )(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=img_h, w=img_w)

        # Upsampling
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            h_l = h.pop()
            x = jnp.concatenate((x, h_l), axis=-1)

            is_last = ind >= (num_resolutions - 1)
            x = ResNetBlock(
                dim=dim_out * 2,
                dim_out=dim_in,
                time_emb_dim=time_emb_dim,
                dtype=self.dtype,
            )(x, time_embed)
            x = ResNetBlock(
                dim=dim_in,
                dim_out=dim_in,
                time_emb_dim=time_emb_dim,
                dtype=self.dtype,
            )(x, time_embed)

            if not is_last:
                x = PixelShuffleUpsample(dim=dim_in, dtype=self.dtype)(x)

        # Residual skip + final
        x = jnp.concatenate((x, residual), axis=-1)
        out_dim = self.channels * patch_height * patch_width
        x = nn.Sequential(
            [
                ResNetBlock(
                    dim=self.dim,
                    dim_out=self.dim,
                    dtype=self.dtype,
                ),
                nn.Dense(
                    features=out_dim,
                    kernel_init=nn.initializers.glorot_uniform(),
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype,
                ),
            ]
        )(x)

        # Un-patching
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=patch_height, p2=patch_width)
        return x
