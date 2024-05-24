import jax
import jax.numpy as jnp
import flax.linen as nn
import scipy
from functools import partial

def two_dim_matmul(
    x,
    matrix_dim_one,
    matrix_dim_two,
    precision = jax.lax.Precision.DEFAULT):
  """Applies 2D matrix multiplication to 2D input arrays.

  Args:
    x: Input of shape [MAX_SEQ_LEN, HIDDEN_DIM]
    matrix_dim_one: [MAX_SEQ_LEN, MAX_SEQ_LEN] matrix to apply to first
      (sequence) dimension of input.
    matrix_dim_two: [HIDDEN_DIM, HIDDEN_DIM] matrix to apply to second (hidden)
      dimension of input.
    precision: XLA precision for matrix multiplication operation.

  Returns:
    [MAX_SEQ_LEN, HIDDEN_DIM] array resulting from application of two
      consecutive matrix multiplications.
  """
  return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two, precision)


@partial(jax.jit, static_argnums=3)
def _two_dim_matmul(x, matrix_dim_one,
                    matrix_dim_two,
                    precision):
  """Applies 2D matrix multiplication to 2D input arrays."""
  return jnp.einsum(  # pytype: disable=wrong-arg-types  # jnp-type
      "ij,jk,ni->nk",
      x,
      matrix_dim_two,
      matrix_dim_one,
      optimize=True,
      precision=precision)

class FourierTransform(nn.Module):
    dim: int
    seq_len: int

    def setup(self):
        dft_mat_hidden = scipy.linalg.dft(self.dim)
        dft_mat_seq = scipy.linalg.dft(self.seq_len)
        #print("constructing fourier for:", self.dim, self.seq_len)

        self.fourier_transform = partial(
          two_dim_matmul,
          matrix_dim_one=jnp.asarray(dft_mat_seq),
          matrix_dim_two=jnp.asarray(dft_mat_hidden))

    def __call__(self, x):
        return jax.vmap(self.fourier_transform)(x).real
