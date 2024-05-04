import jax
import jax.numpy as jnp
import math
from einops import rearrange

def get_slopes(heads: int) -> list[float]:
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(heads).is_integer():
        return get_slopes_power_of_2(heads)

    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

def get_alibi_bias(heads: int, i: int, j: int) -> jax.Array:
    slopes = jnp.array(get_slopes(heads))
    slopes = rearrange(slopes, 'h -> h 1 1')

    def get_bias(i: int, j: int) -> jax.Array:
        i_arange = jnp.arange(j - i, j)
        j_arange = jnp.arange(j)
        bias = -jnp.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    bias = get_bias(i, j)
    bias = bias * slopes
    return bias
