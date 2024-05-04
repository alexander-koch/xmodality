import jax
import jax.numpy as jnp
import math
from einops import rearrange
from typing import NamedTuple, Optional, Any
from functools import partial
from flax import linen as nn


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
    groups: int = 8

    @nn.compact
    def __call__(
        self, x: jax.Array, scale_shift: Optional[list[jax.Array]] = None
    ) -> jax.Array:
        x = nn.Conv(features=self.dim_out, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.GroupNorm(num_groups=self.groups)(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return nn.gelu(x)


class ResNetBlock(nn.Module):
    dim: int
    dim_out: int
    time_emb_dim: Optional[int] = None
    groups: int = 8

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: Optional[jax.Array] = None) -> jax.Array:
        scale_shift = None
        if time_emb is not None:
            t = nn.Sequential(
                [
                    nn.Dense(self.dim_out * 2),
                    nn.GroupNorm(num_groups=self.groups),
                    nn.gelu,
                    nn.Dense(self.dim_out * 2),
                ]
            )(time_emb)
            t = rearrange(t, "b c -> b 1 1 c")
            scale_shift = jnp.split(t, 2, axis=-1)

        h = Block(self.dim_out, self.groups)(x, scale_shift=scale_shift)
        h = Block(self.dim_out, self.groups)(h)
        if self.dim != self.dim_out:
            h = h + nn.Dense(self.dim_out)(x)
        return h


class Upsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        s = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, s, method="nearest")
        x = nn.Conv(features=self.dim, kernel_size=(3, 3), strides=1, padding=1)(x)
        return x


Downsample = partial(nn.Conv, kernel_size=(4, 4), strides=2, padding=1)


class PixelShuffleUpsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Sequential([nn.Dense(self.dim * 4), nn.gelu])(x)
        x = rearrange(x, "b h w (c s1 s2) -> b (h s1) (w s2) c", s1=2, s2=2)
        return x


class PixelShuffleDownsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = rearrange(x, "b (h s1) (w s2) c -> b h w (c s1 s2)", s1=2, s2=2)
        x = nn.Dense(self.dim)(x)
        return x


class UNet(nn.Module):
    dim: int
    dim_mults: tuple[int, ...] = (1, 2, 4)
    channels: int = 3
    resnet_block_groups: int = 8
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        time: Optional[jax.Array] = None,
        condition: Optional[jax.Array] = None,
    ) -> jax.Array:
        init_dim = self.dim // 3 * 2
        dims = [init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        if time is not None:
            time_emb_dim = self.dim * 4
            time_embed = nn.Sequential(
                [
                    SinusoidalPositionEmbeddings(self.dim),
                    nn.Dense(time_emb_dim),
                    nn.GroupNorm(num_groups=32),
                    nn.gelu,
                    nn.Dense(time_emb_dim),
                    nn.GroupNorm(num_groups=32),
                    nn.gelu,
                ]
            )(time)
        else:
            time_emb_dim = None
            time_embed = None

        if condition is not None:
            x = jnp.concatenate((condition, x), axis=-1)

        x = nn.Conv(features=init_dim, kernel_size=7, padding=3)(x)

        h = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            x = ResNetBlock(
                dim_in, dim_out, time_emb_dim, self.resnet_block_groups
            )(x, time_embed)
            x = ResNetBlock(
                dim_out, dim_out, time_emb_dim, self.resnet_block_groups
            )(x, time_embed)
            h.append(x)

            if not is_last:
                x = PixelShuffleDownsample(dim_out)(x)

        mid_dim = dims[-1]
        x = ResNetBlock(
            mid_dim, mid_dim, time_emb_dim, self.resnet_block_groups
        )(x, time_embed)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            h_l = h.pop()
            x = jnp.concatenate((x, h_l), axis=-1)

            is_last = ind >= (num_resolutions - 1)
            x = ResNetBlock(
                dim_out * 2, dim_in, time_emb_dim, self.resnet_block_groups
            )(x, time_embed)
            x = ResNetBlock(
                dim_in, dim_in, time_emb_dim, self.resnet_block_groups
            )(x, time_embed)

            if not is_last:
                x = PixelShuffleUpsample(dim_in)(x)

        out_dim = self.out_dim if self.out_dim is not None else self.channels

        x = nn.Sequential(
            [
                ResNetBlock(self.dim, self.dim, self.resnet_block_groups),
                nn.Dense(out_dim),
            ]
        )(x)
        return x
