#!/usr/bin/env python3

"""Performs inference on a cross-modality diffusion model."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
from jax import random, vmap
import jax.numpy as jnp
import pickle
from models import get_model
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import math
import argparse
from einops import rearrange
from external_validation import strip, vmap_transform
from sampling import get_sampler, get_sampler_names
import functools
import chex

def main(args):
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    module = get_model(args.arch, dtype=dtype)
    assert args.load.endswith(".pkl")
    print(args)

    with open(args.load, "rb") as f:
        state = pickle.load(f)

    factor = 16 if args.arch in ["dit", "test"] else 8
    generator = functools.partial(
        generate,
        module=module,
        params=state.params,
        batch_size=args.batch_size,
        seed=args.seed,
        use_diffusion=not args.disable_diffusion,
        sampler=args.sampler,
        num_sample_steps=args.num_sample_steps,
        factor=factor
    )

    if args.input.endswith(".nii.gz"):
        source = nib.load(args.input)
        header, affine = source.header, source.affine
        source_data = source.get_fdata().astype(np.float32)

        generated_data = generator(source_data)
        print("generated:", generated_data.shape, generated_data.min(), generated_data.max(), generated_data.mean())
        out_img = nib.Nifti1Image(generated_data, header=header, affine=affine)
        nib.save(out_img, args.output)
    elif args.input.endswith(".txt"):
        raise NotImplementedError()

def generate(
    img,
    module,
    params,
    batch_size=64,
    seed=0,
    use_diffusion=True,
    sampler="ddpm",
    num_sample_steps=128,
    factor=8,
):
    """Takes TOF-MRA image as input (unprocessed) and returns a CT in [-50,350] range"""
    chex.assert_rank(img, 3)
    chex.assert_type(img, float)

    #img = img[130:470,41:452, :196]
    #print("img:", img.shape)

    img, lshift, rshift = strip(img)

    # Rescale to [-1,1]
    img = rearrange(img, "h w b -> b h w 1")
    img = vmap_transform(img) * 2 - 1
    num_slices, h, w, _ = img.shape
    print("img:", img.shape)

    # Padding
    new_h = math.ceil(h / factor) * factor
    new_w = math.ceil(w / factor) * factor
    pad_h = new_h - h
    pad_w = new_w - w
    img = jnp.pad(img, ((0,0), (0, pad_h), (0, pad_w), (0,0)))
    print("padded:", img.shape)

    key = random.key(seed)
    num_iters = math.ceil(num_slices / batch_size)
    keys = random.split(key, num_iters * 2)
    sample_fn = get_sampler(sampler)

    out_slices = []
    for i in tqdm(range(num_iters)):
        start = i * batch_size
        if start + batch_size >= num_slices:
            end = num_slices
        else:
            end = start + batch_size
        m = end - start

        init_noise = random.normal(keys[i * 2], (m, new_h, new_w, 1))
        slices = img[start:end]

        if use_diffusion:
            samplekey = keys[i * 2 + 1]
            out = sample_fn(
                module=module,
                params=params,
                key=samplekey,
                img=init_noise,
                condition=slices,
                num_sample_steps=num_sample_steps,
            )
        else:
            out = module.apply(params, slices)
        out = out.astype(jnp.float32)
        out_slices.append(out)

    out = jnp.concatenate(out_slices, axis=0)  # [-1,1]
    out = rearrange(out, "d h w 1 -> h w d")
    out = out[:h, :w]

    out = jnp.clip(out, -1, 1)
    out = (out + 1) * 200 - 50  # [-50, 350]
    out = jnp.pad(out, ((0,0),(0,0),(lshift,rshift)),constant_values=-50)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from", required=True)
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--input", type=str, help="path to image or list of images", required=True)
    p.add_argument("--arch", type=str, choices=["unet", "adm", "uvit", "dit", "test"], help="architecture", required=True)
    p.add_argument("--disable_diffusion", action="store_true", help="disable diffusion")
    p.add_argument("--batch_size", type=int, default=64, help="how many slices to process in parallel")
    p.add_argument("--output", type=str, help="output path", default="out.nii.gz")
    p.add_argument("--seed", type=int, help="random seed to use", default=42)
    p.add_argument("--num_sample_steps", type=int, help="how many steps to sample for", default=128)
    p.add_argument("--sampler", type=str, choices=get_sampler_names(), help="the sampling method to use", default="ddpm")
    main(p.parse_args())
