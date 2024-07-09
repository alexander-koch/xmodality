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
from sampling import get_sampler_names, get_sampler
import nibabel as nib
from tqdm import tqdm
import math
import argparse
from einops import rearrange
from scipy.ndimage import zoom

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
        run(module, state.params, args.arch, not args.disable_diffusion, args.input, args.output, args.batch_size, args.seed, args.sampler, args.num_sample_steps)
    elif args.input.endswith(".txt"):
        raise NotImplementedError()

def run(module, params, arch, use_diffusion, path, out_path, batch_size, seed=0, sampler="ddpm", num_sample_steps=128):
    tof_brain = nib.load(path)
    print("brain:", tof_brain.shape)
    z = tof_brain.shape[-1]
    target_shape = (256,256,z)

    header, affine = tof_brain.header, tof_brain.affine

    #tof_brain
    tof_brain_data = tof_brain.get_fdata().astype(np.float32)
    dsfactor = [w/float(f) for w,f in zip(target_shape, tof_brain_data.shape)]
    tof_brain_data = zoom(tof_brain_data, zoom=dsfactor)
    print("brain (resampled):", tof_brain_data.shape)
    tof_brain = tof_brain_data
            
    tof_brain = rearrange(tof_brain, "h w b -> b h w 1")
    tof_brain = vmap_transform(tof_brain) * 2 - 1
    print("tof_brain (rescaled):", tof_brain.shape, tof_brain.min(), tof_brain.max())
    num_slices, h, w, _ = tof_brain.shape

    # Padding
    # 8 for unet, uvit, adm, 16 for dit (patch size)
    factor = 16 if arch in ["dit", "test"] else 8
    new_h = math.ceil(h / factor) * factor
    new_w = math.ceil(w / factor) * factor
    pad_h = new_h - h
    pad_w = new_w - w
    tof_brain = jnp.pad(tof_brain, ((0,0), (0, pad_h), (0, pad_w), (0,0)))

    # Batch image
    key = random.key(seed)
    num_iters = math.ceil(num_slices / batch_size)
    keys = random.split(key, num_iters*2)

    sample_fn = get_sampler(sampler)

    out_slices = []
    for i in tqdm(range(num_iters)):
        start = i*batch_size
        if start + batch_size >= num_slices:
            end = num_slices
        else:
            end = start + batch_size
        m = end - start

        img = random.normal(keys[i*2], (m, new_h, new_w, 1))
        tof_brain_slices = tof_brain[start:end]

        if use_diffusion:
            samplekey = keys[i*2+1]
            out = sample_fn(module=module, params=params, key=samplekey, img=img, condition=tof_brain_slices, num_sample_steps=num_sample_steps)

            #num_avg_iters = 4
            #samplekeys = random.split(samplekey, num_avg_iters)
            #out = jnp.zeros((m, new_h, new_w, 1))
            #for j in range(num_avg_iters):
            #    sample = sample_fn(module=module, params=params, key=samplekeys[j], img=img, condition=tof_brain_slices, num_sample_steps=num_sample_steps)
            #    out += sample / num_avg_iters

        else:
            out = module.apply(params, tof_brain_slices)
        out_slices.append(out)
    
    out = jnp.concatenate(out_slices, axis=0) # [-1,1]
    out = rearrange(out, "d h w 1 -> h w d")
    out = out[:h, :w]
    out = (out + 1) * 200 - 50 # [-50, 350]
    print("result:", out.shape, out.min(), out.max())

    print("resampling...")
    dsfactor = [1.0/f for f in dsfactor]
    out = zoom(out, zoom=dsfactor)
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
    p.add_argument("--arch", type=str, choices=["unet", "adm", "uvit", "dit", "test"], help="architecture", required=True)
    p.add_argument("--disable_diffusion", action="store_true", help="disable diffusion")
    p.add_argument("--batch_size", type=int, default=64, help="how many slices to process in parallel")
    p.add_argument("--output", type=str, help="output path", default="out.nii.gz")
    p.add_argument("--seed", type=int, help="random seed to use", default=42)
    p.add_argument("--num_sample_steps", type=int, help="how many steps to sample for", default=128)
    p.add_argument("--sampler", type=str, choices=get_sampler_names(), help="the sampling method to use", default="ddpm")
    main(p.parse_args())
