import functools
import warnings
from typing import Any, Callable, Optional, Union, overload

import jax
import jax.numpy as jnp
from jax import lax, random

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import (
  DenseGeneral,
  default_kernel_init,
)
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import LayerNorm
from flax.typing import (
  Array,
  PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)

import flax.linen as nn
from einops import rearrange

from alibi import get_alibi_bias

def dot_product_attention_weights(
  query: Array,
  key: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[PRNGKey] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
) -> Array:
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of ``[batch..., q_length,
      num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is ``False``.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable via
      ``mutable=['intermediates']`` in order to have that collection returned.
      If ``module`` is None, the attention weights will not be sowed.

  Returns:
    Output of shape ``[batch..., num_heads, q_length, kv_length]``.
  """
  query, key = promote_dtype(query, key, dtype=dtype)
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
    '...qhd,...khd->...hqk', query, key, precision=precision
  )

  # pre-softmax talking heads
  heads = attn_weights.shape[1]
  attn_weights = rearrange(attn_weights, "... h q k -> ... q k h")
  attn_weights = nn.Conv(features=heads, kernel_size=(1,1), strides=(1,1), padding=0, use_bias=False)(attn_weights)
  attn_weights = rearrange(attn_weights, "... q k h -> ... h q k")

  # alibi relative positional embedding
  i, j = attn_weights.shape[-2:]
  attn_bias = get_alibi_bias(heads, i, j)
  attn_weights = attn_weights + attn_bias

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min # type: ignore
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # post-softmax talking heads
  attn_weights = rearrange(attn_weights, "... h q k -> ... q k h")
  attn_weights = nn.Conv(features=heads, kernel_size=(1,1), strides=(1,1), padding=0, use_bias=False)(attn_weights)
  attn_weights = rearrange(attn_weights, "... q k h -> ... h q k")

  if module:
    module.sow('intermediates', 'attention_weights', attn_weights)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights



def talking_heads_dot_product_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[PRNGKey] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
) -> Array:
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  .. note::
    ``query``, ``key``, ``value`` needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of ``[batch..., q_length,
      num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    value: values to be used in attention with shape of ``[batch..., kv_length,
      num_heads, v_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is ``False``.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see ``jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable via
      ``mutable=['intermediates']`` in order to have that collection returned.
      If ``module`` is None, the attention weights will not be sowed.

  Returns:
    Output of shape ``[batch..., q_length, num_heads, v_depth_per_head]``.
  """
  query, key, value = promote_dtype(query, key, value, dtype=dtype)
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    module,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
    '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )
