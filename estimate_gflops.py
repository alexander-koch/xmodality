#!/usr/bin/env python3
import jax
from jax import jit, random, numpy as jnp
import argparse
from functools import partial
from models import get_model

def main(args):
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    module = get_model(args.arch, dtype=dtype)
    
    @jit
    def module_fn(params, x, **kwargs):
        return module.apply(params, x, **kwargs)

    key = random.key(0)
    batch_size = 1
    x = jnp.ones((batch_size, 256, 256, 1))
    condition = jnp.ones((batch_size, 256, 256, 1))

    if args.disable_diffusion:
        params = module.init(key, x)
        compiled = module_fn.lower(params, x).compile()
    else:
        time = jnp.ones((batch_size,))
        params = module.init(key, x, time=time, condition=condition)
        compiled = module_fn.lower(params, x, time=time, condition=condition).compile()

    cost = compiled.cost_analysis()
    gflops = round(cost[0]["flops"] / 10**9)
    print(f"gflops: {gflops}")

    num_params = round(sum(x.size for x in jax.tree_util.tree_leaves(params)) / 10**6)
    print(f"num params: {num_params}M")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--disable_diffusion", action="store_true", help="disable diffusion")
    p.add_argument("--arch", type=str, choices=["unet", "adm", "uvit", "dit"], help="architecture", required=True)
    main(p.parse_args())
