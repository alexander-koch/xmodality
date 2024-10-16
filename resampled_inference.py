#!/usr/bin/env python3

"""Performs inference on a cross-modality diffusion model."""

#import os
#os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
#os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
#os.environ["TF_DETERMINISTIC_ops"] = "1"

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import pickle
from models import get_model
import numpy as np
from sampling import get_sampler_names
import nibabel as nib
import argparse
import functools
from external_validation import generate
import utils

def main(args):
    print(args)

    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    module = get_model(args.arch, dtype=dtype)
    params = utils.load_params(args.load)

    generator = functools.partial(
        generate,
        module=module,
        params=params,
        batch_size=args.batch_size,
        seed=args.seed,
        use_diffusion=not args.disable_diffusion,
        sampler=args.sampler,
        num_sample_steps=args.num_sample_steps,
        #order=0,
        #prefilter=False
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
    p.add_argument("--sampler", type=str, choices=get_sampler_names(), help="the sampling method to use", default="ddpm")
    main(p.parse_args())
