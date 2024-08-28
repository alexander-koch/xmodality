import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
from jax import jit, random, numpy as jnp
import argparse
from functools import partial
import timeit
from models import get_model


def get_compiled(arch, bfloat16=False, disable_diffusion=False):
    dtype = jnp.bfloat16 if bfloat16 else jnp.float32
    module = get_model(arch, dtype=dtype)
    
    @jit
    def module_fn(params, x, **kwargs):
        return module.apply(params, x, **kwargs)

    key = random.key(0)
    batch_size = 16
    x = jnp.ones((batch_size, 256, 256, 1))
    condition = jnp.ones((batch_size, 256, 256, 1))

    if disable_diffusion:
        params = module.init(key, x)
        compiled = module_fn.lower(params, x).compile()
        return compiled, params, x
    else:
        time = jnp.ones((batch_size,))
        params = module.init(key, x, time=time, condition=condition)
        compiled = module_fn.lower(params, x, time=time, condition=condition).compile()
        fn = partial(compiled, time=time, condition=condition)
        return fn, params, x

