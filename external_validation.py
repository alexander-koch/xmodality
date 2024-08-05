#!/usr/bin/env python3
import json
import argparse
import numpy as np
from einops import rearrange
import math
from tqdm import tqdm
import functools
import jax.numpy as jnp
from sampling import get_sampler, get_sampler_names
from models import get_model
import pickle
import nibabel as nib
from scipy.ndimage import zoom
from jax import vmap, random
import jax
import yaml


def get_metrics(y_hat: jax.Array, y: jax.Array) -> dict[str, float]:
    """Assumes y_hat and y are in [-50,350] range."""
    mse = jnp.mean(jnp.square(y - y_hat))
    mae = jnp.mean(jnp.abs(y - y_hat))
    return {
        "mse": mse.item(),
        "mae": mae.item(),
    }


def transform(img: jax.Array) -> jax.Array:
    min_v = img.min()
    max_v = img.max()
    return (img - min_v) / (max_v - min_v)


def vmap_transform(img: jax.Array) -> jax.Array:
    return vmap(transform)(img)

def top_strip(img):
    # Assume img (h,w,d)
    z = img.shape[-1]-1
    while z > 0 and jnp.all(jnp.isclose(img[:, :, z], img[:,:,z].min())):
        z = z - 1
    return img[:, :, :z+1], img.shape[-1] - z - 1

def bottom_strip(img):
    # Assume img (h,w,d)
    z = 0
    while z < img.shape[-1] and jnp.all(jnp.isclose(img[:, :, z], img[:,:,z].min())):
        z = z + 1
    return img[:, :, z:], z

def strip(img):
    img, shift_top = top_strip(img)
    img, shift_bottom = bottom_strip(img)
    return img, shift_bottom, shift_top

def generate(
    img,
    module,
    params,
    batch_size=64,
    seed=0,
    use_diffusion=True,
    sampler="ddpm",
    num_sample_steps=128,
):
    """Takes TOF-MRA image as input (unprocessed) and returns a CT in [-1,1] range"""
    img, lshift, rshift = strip(img)

    z = img.shape[-1]
    target_shape = (256, 256, z)

    # Resample to target resolution
    dsfactor = [w / float(f) for w, f in zip(target_shape, img.shape)]
    img_resampled = zoom(img, zoom=dsfactor)

    # Rescale to [-1,1]
    img_resampled = rearrange(img_resampled, "h w b -> b h w 1")
    img_resampled = vmap_transform(img_resampled) * 2 - 1
    num_slices, h, w, _ = img_resampled.shape

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

        init_noise = random.normal(keys[i * 2], (m, h, w, 1))
        slices = img_resampled[start:end]

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
        out_slices.append(out)

    out = jnp.concatenate(out_slices, axis=0)  # [-1,1]
    out = rearrange(out, "d h w 1 -> h w d")
    out = out[:h, :w]

    # Resample to original resolution
    dsfactor = [1.0 / f for f in dsfactor]
    out = zoom(out, zoom=dsfactor)

    out = jnp.clip(out, -1, 1)
    out = (out + 1) * 200 - 50  # [-50, 350]
    out = jnp.pad(out, ((0,0),(0,0),(lshift,rshift),constant_values=-50))
    return out


def main(args):
    # Load the data
    if args.external:
        sources, targets = [], []
        with open("dataset_external.txt", "r") as f:
            for line in f:
                src, tgt = line.strip().split(",")
                sources.append(src)
                targets.append(tgt)

        test_sources = sources
        test_targets = targets
    else:
        sources, targets = [], []
        with open("dataset.txt", "r") as f:
            for line in f:
                src, tgt = line.strip().split(",")
                sources.append(src)
                targets.append(tgt)

        with open("test_indices.json", "r") as f:
            test_indices = json.load(f)

        test_sources = [sources[i] for i in test_indices]
        test_targets = [targets[i] for i in test_indices]
    print("number of images:", len(test_sources))

    # Load the model
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    module = get_model(args.arch, dtype=dtype)
    assert args.load.endswith(".pkl")
    print(args)
    with open(args.load, "rb") as f:
        state = pickle.load(f)

    # Prepare generator and run
    generator = functools.partial(
        generate,
        module=module,
        params=state.params,
        batch_size=args.batch_size,
        seed=args.seed,
        use_diffusion=not args.disable_diffusion,
        sampler=args.sampler,
        num_sample_steps=args.num_sample_steps,
    )

    metrics_list = []
    for src_path, tgt_path in zip(test_sources, test_targets):
        # Prepare and load data
        src_img = nib.load(src_path)
        tgt_img = nib.load(tgt_path)
        src_img_data = src_img.get_fdata().astype(np.float32)
        tgt_img_data = tgt_img.get_fdata().astype(np.float32)
        tgt_img_data = jnp.clip(tgt_img_data, -50, 350)

        # Perform prediction
        tgt_img_data_hat = generator(src_img_data)

        # Get metrics
        metrics = get_metrics(tgt_img_data_hat, tgt_img_data)
        metrics_list.append(metrics)

    # Accumulate metrics
    mse = jnp.mean(jnp.stack([s["mse"] for s in metrics_list]))
    mae = jnp.mean(jnp.stack([s["mae"] for s in metrics_list]))
    metrics = {
        "mse": mse.item(),
        "mae": mae.item(),
    }
    with open(args.output, "w") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--load", type=str, help="path to load pretrained weights from", required=True
    )
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--external", action="store_true", help="use external data")
    p.add_argument(
        "--arch",
        type=str,
        choices=["unet", "adm", "uvit", "dit"],
        help="architecture",
        required=True,
    )
    p.add_argument("--disable_diffusion", action="store_true", help="disable diffusion")
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="how many slices to process in parallel",
    )
    p.add_argument("--seed", type=int, help="random seed to use", default=42)
    p.add_argument(
        "--num_sample_steps", type=int, help="how many steps to sample for", default=128
    )
    p.add_argument(
        "--sampler",
        type=str,
        choices=get_sampler_names(),
        help="the sampling method to use",
        default="ddpm",
    )
    p.add_argument(
        "--output", type=str, help="where to write the metrics to", required=True
    )
    main(p.parse_args())
