# Taken from https://github.com/google-research/vision_transformer

from flax.training import checkpoints
import numpy as np
import jax
import re
import flax
from typing import Any, Callable, Optional, Tuple, Type
import collections
import scipy
import flax.linen as nn
import jax.numpy as jnp
from jax import random
import ml_collections


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.model_name = "ViT-B_16"
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.classifier = "token"
    config.representation_size = None
    return config


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param(
            "pos_embedding", self.posemb_init, pos_emb_shape, self.param_dtype
        )
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        nn.initializers.xavier_uniform()
    )
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(
        stddev=1e-6
    )

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            inputs
        )
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            x
        )
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate
        )(y, deterministic=deterministic)

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        if self.add_position_embedding:
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
                name="posembed_input",
            )(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{lyr}",
                num_heads=self.num_heads,
            )(x, deterministic=not train)
        encoded = nn.LayerNorm(name="encoder_norm")(x)

        return encoded


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    num_classes: int
    patches: Any
    transformer: Any
    hidden_size: int
    representation_size: Optional[int] = None
    classifier: str = "token"
    head_bias_init: float = 0.0
    encoder: Type[nn.Module] = Encoder
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, x, *, train):
        n, h, w, c = x.shape

        # We can merge s2d+emb into a single conv; it's the same.
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches.size,
            strides=self.patches.size,
            padding="VALID",
            name="embedding",
        )(x)

        # Here, x is a grid of embeddings.

        # (Possibly partial) Transformer.
        if self.transformer is not None:
            n, h, w, c = x.shape
            x = jnp.reshape(x, [n, h * w, c])

            # If we want to add a class token, add it here.
            if self.classifier in ["token", "token_unpooled"]:
                cls = self.param("cls", nn.initializers.zeros, (1, 1, c))
                cls = jnp.tile(cls, [n, 1, 1])
                x = jnp.concatenate([cls, x], axis=1)

            x = self.encoder(name="Transformer", **self.transformer)(x, train=train)

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
        elif self.classifier in ["unpooled", "token_unpooled"]:
            pass
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        # if self.representation_size is not None:
        #  x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
        #  x = nn.tanh(x)
        # else:
        #  x = IdentityLayer(name='pre_logits')(x)

        # if self.num_classes:
        #  x = nn.Dense(
        #      features=self.num_classes,
        #      name='head',
        #      kernel_init=nn.initializers.zeros,
        #      bias_init=nn.initializers.constant(self.head_bias_init))(x)
        return x


def _fix_groupnorm(params):
    # See https://github.com/google/flax/issues/1721
    regex = re.compile(r"gn(\d+|_root|_proj)$")

    def fix_gn(args):
        path, array = args
        if len(path) > 1 and regex.match(path[-2]) and path[-1] in ("bias", "scale"):
            array = array.squeeze()
        return (path, array)

    return flax.traverse_util.unflatten_dict(
        dict(map(fix_gn, flax.traverse_util.flatten_dict(params).items()))
    )


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.

    This function is useful to analyze checkpoints that are without need to access
    the exact source code of the experiment. In particular, it can be used to
    extract an reuse various subtrees of the scheckpoint, e.g. subtree of
    parameters.

    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.

    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def interpolate_posembed(posemb, num_tokens: int, has_class_token: bool):
    """Interpolate given positional embedding parameters into a new shape.

    Args:
      posemb: positional embedding parameters.
      num_tokens: desired number of tokens.
      has_class_token: True if the positional embedding parameters contain a
        class token.

    Returns:
      Positional embedding parameters interpolated into the new shape.
    """
    assert posemb.shape[0] == 1
    if has_class_token:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        num_tokens -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]

    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(num_tokens))
    print(f"interpolate_posembed: grid-size from {gs_old} to {gs_new}")
    assert gs_old**2 == len(posemb_grid), f"{gs_old ** 2} != {len(posemb_grid)}"
    assert gs_new**2 == num_tokens, f"{gs_new ** 2} != {num_tokens}"
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    return jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))


def load(path, num_tokens=257, has_class_token=True):
    """Loads params from a checkpoint previously stored with `save()`."""
    with open(path, "rb") as f:
        ckpt_dict = np.load(f, allow_pickle=False)
        keys, values = zip(*list(ckpt_dict.items()))
    params = checkpoints.convert_pre_linen(recover_tree(keys, values))
    if isinstance(params, flax.core.FrozenDict):
        params = params.unfreeze()
    params = _fix_groupnorm(params)
    posemb = params["Transformer"]["posembed_input"]["pos_embedding"]
    posemb = interpolate_posembed(posemb, num_tokens, has_class_token)
    params["Transformer"]["posembed_input"]["pos_embedding"] = posemb
    params = {"params": params}
    return flax.core.freeze(params)


def get_b16_model(ckpt_path="ViT-B_16.npz"):
    config = get_b16_config()
    model = VisionTransformer(**config, num_classes=1000)
    params = load(ckpt_path)
    return model, params
