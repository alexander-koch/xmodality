import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

from models import ADM, UViT, UNet, DiT
import jax
from jax import jit, random, numpy as jnp
import argparse
from functools import partial
import timeit


def get_compiled(arch, bfloat16=False, disable_diffusion=False):
    dtype = jnp.bfloat16 if bfloat16 else jnp.float32
    if arch == "adm":
        module = ADM(dim=128, channels=1, dtype=dtype)
    elif arch == "uvit":
        module = UViT(dim=128, channels=1, dtype=dtype)
    elif arch == "unet":
        module = UNet(dim=128, channels=1, dtype=dtype)
    elif arch == "dit":
        module = DiT(
            patch_size=16,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            in_channels=1,
            dtype=dtype,
        )
    else:
        raise NotImplementedError()
    
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

