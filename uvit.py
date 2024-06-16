import jax
import jax.numpy as jnp
import math
from einops import rearrange, repeat
from typing import NamedTuple, Optional, Any, Union
from functools import partial
from flax import linen as nn
import jax_wavelets as jw
from flax.linen.dtypes import promote_dtype


def pair(x: Union[int, tuple[int, int, int]]):
    return x if isinstance(x, tuple) else (x, x)


def sinusoidal_embedding_2d(
    h: int, w: int, dim: int, temperature: int = 10000
) -> jax.Array:
    y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    omega = jnp.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = jnp.ravel(y)[:, None] * omega[None, :]
    x = jnp.ravel(x)[:, None] * omega[None, :]
    pos = jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=1)
    return jnp.expand_dims(pos, 0)


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


class SwiGLU(nn.Module):
    dim: int
    mlp_dim: int
    time_emb_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: Optional[jax.Array] = None) -> jax.Array:
        if time_emb is not None:
            t = nn.Sequential(
                [
                    nn.gelu,
                    nn.Dense(
                        features=self.mlp_dim * 2,
                        kernel_init=nn.initializers.glorot_uniform(),
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                    ),
                ]
            )(time_emb)
            t = rearrange(t, "b c -> b 1 c")
            scale_shift = jnp.split(t, 2, axis=-1)
        else:
            scale_shift = None

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

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = nn.Dense(
            features=self.dim,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class MLP(nn.Module):
    dim: int
    mlp_dim: int
    time_emb_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: Optional[jax.Array] = None) -> jax.Array:
        if time_emb is not None:
            t = nn.Sequential(
                [
                    nn.gelu,
                    nn.Dense(
                        features=self.mlp_dim * 2,
                        kernel_init=nn.initializers.glorot_uniform(),
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                    ),
                ]
            )(time_emb)
            t = rearrange(t, "b c -> b 1 c")
            scale_shift = jnp.split(t, 2, axis=-1)
        else:
            scale_shift = None

        x = nn.Dense(
            features=self.mlp_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = nn.gelu(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = nn.Dense(
            features=self.dim,
            kernel_init=nn.initializers.glorot_uniform(),
            use_bias=False,
            dtype=self.dtype,
        )(x)
        return x


class Transformer(nn.Module):
    dim: int
    mlp_dim: int
    num_heads: int
    depth: int
    time_emb_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: Optional[jax.Array] = None) -> jax.Array:
        for _ in range(self.depth):
            x_hat = nn.RMSNorm()(x)
            x_hat = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=self.dtype,
            )(x_hat)
            x = x + x_hat

            x_hat = nn.RMSNorm()(x)
            # x_hat = MLP(dim=self.dim, mlp_dim=self.mlp_dim, time_emb_dim=self.time_emb_dim, dtype=self.dtype)(x_hat, time_emb)
            x_hat = SwiGLU(
                dim=self.dim,
                mlp_dim=self.mlp_dim,
                time_emb_dim=self.time_emb_dim,
                dtype=self.dtype,
            )(x_hat, time_emb)
            x = x + x_hat
        return nn.RMSNorm()(x)


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

        h = Block(dim_out=self.dim_out, dtype=self.dtype)(x, scale_shift=scale_shift)
        h = Block(dim_out=self.dim_out, dtype=self.dtype)(h)
        if self.dim != self.dim_out:
            h = h + nn.Dense(
                features=self.dim_out,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=self.dtype,
            )(x)
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


class Wavelet(nn.Module):
    channels: int
    wavelet: str = "bior4.4"
    levels: int = 1
    mode: str = "wrap"
    dtype: Any = jnp.float32

    def setup(self):
        filt = jw.get_filter_bank(self.wavelet, self.dtype)
        self.kernel_dec, self.kernel_rec = jw.make_kernels(filt, self.channels)

    def encode(self, x):
        x = promote_dtype(x, dtype=self.dtype)[0]
        return jw.wavelet_dec(x, self.kernel_dec, levels=self.levels, mode=self.mode)

    def decode(self, x):
        x = promote_dtype(x, dtype=self.dtype)[0]
        return jw.wavelet_rec(x, self.kernel_rec, levels=self.levels, mode=self.mode)


class UViT(nn.Module):
    dim: int
    dim_mults: tuple[int, ...] = (1, 2, 4)
    channels: int = 1
    vit_num_heads: int = 4
    vit_depth: int = 16
    num_heads: int = 4
    # patch_size: Union[int, tuple[int, int]] = 4
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
        # patch_height, patch_width = pair(self.patch_size)

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
                    nn.gelu,
                    nn.Dense(
                        features=time_emb_dim,
                        kernel_init=nn.initializers.glorot_uniform(),
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                    ),
                ]
            )(time)
        else:
            time_emb_dim = None
            time_embed = None

        if condition is not None:
            x = jnp.concatenate((condition, x), axis=-1)

        wavelet = Wavelet(channels=x.shape[-1], dtype=self.dtype)

        # Patching
        # sh = x.shape[1:3]
        # x = rearrange(x, "b (h p1) (w p2) c -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width)
        # x = nn.LayerNorm()(x)
        # x = nn.Dense(
        #    features=self.dim,
        #    kernel_init=nn.initializers.glorot_uniform(),
        #    bias_init=nn.initializers.zeros,
        #    dtype=self.dtype,
        # )(x)
        # x = nn.LayerNorm()(x)

        x = wavelet.encode(x)
        out_dim = x.shape[-1]
        # out_dim = 1
        x = nn.Conv(
            features=self.dim,
            kernel_size=(7, 7),
            padding=3,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
        )(x)
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
            h.append(x)

            x = ResNetBlock(
                dim=dim_out,
                dim_out=dim_out,
                time_emb_dim=time_emb_dim,
                dtype=self.dtype,
            )(x, time_embed)
            x_hat = nn.RMSNorm()(x)
            x_hat = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=self.dtype,
            )(x_hat)
            x = x + x_hat
            h.append(x)

            if not is_last:
                x = PixelShuffleDownsample(dim=dim_out, dtype=self.dtype)(x)

        # Transformer middle part
        mid_dim = dims[-1]
        img_h, img_w, c = x.shape[1], x.shape[2], x.shape[3]
        seq_len = img_h * img_w
        x = rearrange(x, "b h w c -> b (h w) c")
        pos_emb = sinusoidal_embedding_2d(img_h, img_w, c)
        x = x + pos_emb

        x = Transformer(
            dim=mid_dim,
            mlp_dim=mid_dim * 4,
            num_heads=self.vit_num_heads,
            depth=self.vit_depth,
            time_emb_dim=time_emb_dim,
            dtype=self.dtype,
        )(x, time_embed)
        x = rearrange(x, "b (h w) c -> b h w c", h=img_h, w=img_w)

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

            h_l = h.pop()
            x = jnp.concatenate((x, h_l), axis=-1)
            x = ResNetBlock(
                dim=dim_in,
                dim_out=dim_in,
                time_emb_dim=time_emb_dim,
                dtype=self.dtype,
            )(x, time_embed)
            x_hat = nn.RMSNorm()(x)
            x_hat = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=self.dtype,
            )(x_hat)
            x = x + x_hat

            if not is_last:
                x = PixelShuffleUpsample(dim=dim_in, dtype=self.dtype)(x)

        # Residual skip + final
        x = jnp.concatenate((x, residual), axis=-1)

        x = ResNetBlock(
            dim=self.dim,
            dim_out=self.dim,
            dtype=self.dtype,
        )(x)
        x = nn.Dense(
            features=out_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
        )(x)

        x = wavelet.decode(x)
        x = nn.Dense(
            features=1,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
        )(x)

        # out_dim = self.channels * patch_height * patch_width
        # out_dim = 1
        # x = nn.Sequential(
        #    [
        #        ResNetBlock(
        #            dim=self.dim,
        #            dim_out=self.dim,
        #            dtype=self.dtype,
        #        ),
        #        nn.Dense(
        #            features=out_dim,
        #            kernel_init=nn.initializers.glorot_uniform(),
        #            bias_init=nn.initializers.zeros,
        #            dtype=self.dtype,
        #        ),
        #    ]
        # )(x)

        # Un-patching
        # x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=patch_height, p2=patch_width)
        return x
