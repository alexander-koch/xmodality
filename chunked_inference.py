#!/usr/bin/env python3

"""Performs inference on a cross-modality diffusion model."""

#import os
#os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
#os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
#os.environ["TF_DETERMINISTIC_ops"] = "1"

import jax
from jax import random, vmap
import jax.numpy as jnp
import pickle
from models import get_model
import numpy as np
import matplotlib.pyplot as plt
from sampling import ddpm_sample, ddim_sample
import nibabel as nib
from tqdm import tqdm
import math
import argparse
from einops import rearrange

def transform(img):
    min_v = img.min()
    max_v = img.max()
    return (img - min_v) / (max_v - min_v)

def vmap_transform(img):
    return vmap(transform)(img)

def main(args):
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    module = get_model(args.arch, dtype=dtype)
    assert args.load.endswith(".pkl")
    print(args)

    with open(args.load, "rb") as f:
        state = pickle.load(f)

    if args.input.endswith(".nii.gz"):
        run(module, state.params, not args.disable_diffusion, args.input, args.output, args.batch_size, args.seed, args.sampler, args.num_sample_steps)
    elif args.input.endswith(".txt"):
        raise NotImplementedError()

def run(module, params, use_diffusion, path, out_path, batch_size, seed=0, sampler="ddpm", num_sample_steps=128):
    tof_brain = nib.load(path)
    header, affine = tof_brain.header, tof_brain.affine
            
    tof_brain = jnp.array(tof_brain.get_fdata().astype(np.float32))
    tof_brain = rearrange(tof_brain, "h w b -> b h w 1")
    tof_brain = vmap_transform(tof_brain) * 2 - 1
    print("tof_brain (rescaled):", tof_brain.shape, tof_brain.min(), tof_brain.max())
    num_slices, h, w, _ = tof_brain.shape

    # Batch image
    key = random.key(seed)
    num_iters = math.ceil(num_slices / batch_size)
    keys = random.split(key, num_iters*2)

    sample_fn = ddpm_sample if sampler == "ddpm" else ddim_sample

    out_slices = []
    for i in tqdm(range(num_iters)):
        start = i*batch_size
        if start + batch_size >= num_slices:
            end = num_slices
        else:
            end = start + batch_size
        m = end - start

        img = random.normal(keys[i*2], (m, 256, 256, 1))
        tof_brain_slices = tof_brain[start:end]

        out = jnp.zeros((m, h, w, 1))
        num_u = math.ceil(h / 256)
        num_v = math.ceil(w / 256)

        for u in range(num_u):
            for v in range(num_v):
                u_start = u * 256
                u_end = u_start + 256
                v_start =v * 256
                v_end = v_start + 256

                if u == num_u - 1:
                    u_start = h - 256
                    u_end = h
                if v == num_v - 1:
                    v_start = w - 256
                    v_end = w

                print("i:", i, "u:", u_start, u_end, u_end-u_start)
                print("i:", i, "v:", v_start, v_end, v_end-v_start)
                tof_chunk = tof_brain_slices[:, u_start:u_end, v_start:v_end,:]

                if use_diffusion:
                    chunk = sample_fn(module=module, params=params, key=keys[i*2+1], img=img, condition=tof_chunk, num_sample_steps=num_sample_steps)
                else:
                    chunk = module.apply(params, tof_chunk)

                print("chunk:", chunk.shape)
                out = out.at[:, u_start:u_end, v_start:v_end, :].set(chunk)
        out_slices.append(out)
    
    out = jnp.concatenate(out_slices, axis=0) # [-1,1]
    out = rearrange(out, "d h w 1 -> h w d")
    out = out[:h, :w]
    out = (out + 1) * 200 - 50 # [-50, 350]
    print("final:", out.shape, out.min(), out.max())

    img = nib.Nifti1Image(out, header=header, affine=affine)
    nib.save(img, out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from", required=True)
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--input", type=str, help="path to image or list of images", required=True)
    p.add_argument("--arch", type=str, choices=["unet", "adm", "uvit", "dit"], help="architecture", required=True)
    p.add_argument("--disable_diffusion", action="store_true", help="disable diffusion")
    p.add_argument("--batch_size", type=int, default=64, help="how many slices to process in parallel")
    p.add_argument("--output", type=str, help="output path", default="out.nii.gz")
    p.add_argument("--seed", type=int, help="random seed to use", default=42)
    p.add_argument("--num_sample_steps", type=int, help="how many steps to sample for", default=128)
    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], help="the sampling method to use", default="ddpm")
    main(p.parse_args())
