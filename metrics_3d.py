import chex
import jax
import jax.numpy as jnp
from typing import Optional, Callable, Union

def mae_3d(a: chex.Array, b: chex.Array) -> chex.Numeric:
  """Returns the Mean Absolute Error between `a` and `b`.

  Args:
    a: First image (or set of images).
    b: Second image (or set of images).

  Returns:
    MAE between `a` and `b`.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank([a, b], {4, 5})
  chex.assert_type([a, b], float)
  chex.assert_equal_shape([a, b])
  return jnp.abs(a - b).mean(axis=(-4, -3, -2, -1))

def mse_3d(a: chex.Array, b: chex.Array) -> chex.Numeric:
  """Returns the Mean Squared Error between `a` and `b`.

  Args:
    a: First image (or set of images).
    b: Second image (or set of images).

  Returns:
    MSE between `a` and `b`.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank([a, b], {4, 5})
  chex.assert_type([a, b], float)
  chex.assert_equal_shape([a, b])
  return jnp.square(a - b).mean(axis=(-4, -3, -2, -1))

def psnr_3d(a: chex.Array,
            b: chex.Array,
            data_range: Union[tuple[float, float], float] = 1.0) -> chex.Numeric:
  """Returns the Peak Signal-to-Noise Ratio between `a` and `b`.

  Assumes that the dynamic range of the images (the difference between the
  maximum and the minimum allowed values) is 1.0.

  Args:
    a: First image (or set of images).
    b: Second image (or set of images).
    data_range: Range of the data of a and b

  Returns:
    PSNR in decibels between `a` and `b`.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank([a, b], {4, 5})
  chex.assert_type([a, b], float)
  chex.assert_equal_shape([a, b])
    
  if isinstance(data_range, tuple):
    data_range = data_range[1] - data_range[0]

  return (2 * jnp.log(data_range) - jnp.log(mse_3d(a, b))) * (10 / jnp.log(10.0))

def ssim_3d(
    a: chex.Array,
    b: chex.Array,
    *,
    data_range: Union[float, tuple[float, float]] = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    precision=jax.lax.Precision.HIGHEST,
    filter_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
) -> chex.Numeric:
  """Computes the structural similarity index (SSIM) between image pairs.

  This function is based on the standard SSIM implementation from:
  Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
  "Image quality assessment: from error visibility to structural similarity",
  in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

  This function was modeled after tf.image.ssim, and should produce comparable
  output.

  Note: the true SSIM is only defined on grayscale. This function does not
  perform any colorspace transform. If the input is in a color space, then it
  will compute the average SSIM.

  Args:
    a: First image (or set of images).
    b: Second image (or set of images).
    data_range: Range of the data
    filter_size: Window size (>= 1). Image dims must be at least this small.
    filter_sigma: The bandwidth of the Gaussian used for filtering (> 0.).
    k1: One of the SSIM dampening parameters (> 0.).
    k2: One of the SSIM dampening parameters (> 0.).
    return_map: If True, will cause the per-pixel SSIM "map" to be returned.
    precision: The numerical precision to use when performing convolution.
    filter_fn: An optional argument for overriding the filter function used by
      SSIM, which would otherwise be a 2D Gaussian blur specified by filter_size
      and filter_sigma.

  Returns:
    Each image's mean SSIM, or a tensor of individual values if `return_map`.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank([a, b], {4, 5})
  chex.assert_type([a, b], float)
  chex.assert_equal_shape([a, b])

  if filter_fn is None:
    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = jnp.exp(-0.5 * f_i)
    filt /= jnp.sum(filt)

    # Construct a 1D convolution.
    def filter_fn_1(z):
      return jnp.convolve(z, filt, mode="valid", precision=precision)

    filter_fn_vmap = jax.vmap(filter_fn_1)

    # H W C
    # D H W C

    # Apply the vectorized filter along the z axis.
    def filter_fn_z(z):
      z_flat = jnp.moveaxis(z, -4, -1).reshape((-1, z.shape[-4]))
      z_filtered_shape = ((z.shape[-5],) if z.ndim == 5 else ()) + (
          z.shape[-3],
          z.shape[-2],
          z.shape[-1],
          -1,
      )
      z_filtered = jnp.moveaxis(
          filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -4
      )
      return z_filtered

    # H W C
    # D H W C

    # Apply the vectorized filter along the y axis.
    def filter_fn_y(z):
      z_flat = jnp.moveaxis(z, -3, -1).reshape((-1, z.shape[-3]))
      z_filtered_shape = ((z.shape[-5],) if z.ndim == 5 else ()) + (
          z.shape[-4],
          z.shape[-2],
          z.shape[-1],
          -1,
      )
      z_filtered = jnp.moveaxis(
          filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -3
      )
      return z_filtered

    # H W C
    # D H W C

    # Apply the vectorized filter along the x axis.
    def filter_fn_x(z):
      z_flat = jnp.moveaxis(z, -2, -1).reshape((-1, z.shape[-2]))
      z_filtered_shape = ((z.shape[-5],) if z.ndim == 5 else ()) + (
          z.shape[-4],
          z.shape[-3],
          z.shape[-1],
          -1,
      )
      z_filtered = jnp.moveaxis(
          filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -2
      )
      return z_filtered

    # Apply the blur in both x and y.
    filter_fn = lambda z: filter_fn_z(filter_fn_y(filter_fn_x(z)))

  mu0 = filter_fn(a)
  mu1 = filter_fn(b)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filter_fn(a**2) - mu00
  sigma11 = filter_fn(b**2) - mu11
  sigma01 = filter_fn(a * b) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  epsilon = jnp.finfo(jnp.float32).eps ** 2
  sigma00 = jnp.maximum(epsilon, sigma00)
  sigma11 = jnp.maximum(epsilon, sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
  )

  if isinstance(data_range, tuple):
    data_range = data_range[1] - data_range[0]

  c1 = (k1 * data_range) ** 2
  c2 = (k2 * data_range) ** 2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  ssim_value = jnp.mean(ssim_map, list(range(-4, 0)))
  return ssim_map if return_map else ssim_value

if __name__ == "__main__":
  x = jnp.zeros((32, 32, 32, 3))
  y = jnp.ones((32, 32, 32, 3))
  print(ssim_3d(x,y))
  print(psnr_3d(x,y))

  x = jnp.zeros((1, 32, 32, 32, 3))
  y = jnp.ones((1, 32, 32, 32, 3))
  print(ssim_3d(x,y))
  print(psnr_3d(x,y))
