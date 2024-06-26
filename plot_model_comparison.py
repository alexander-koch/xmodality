#!/bin/bash

from glob import glob
from dataset import SliceDS
from sampling import ddpm_sample, ddim_sample
from grain.python import DataLoader, ReadOptions, Batch, ShardOptions, IndexSampler
from models import get_model

import matplotlib.pyplot as plt
from jax import random
import pickle
import numpy as np
import jax.numpy as jnp
import utils
import torch

SEED = 42

def main():
    batch_size = 8
    num_sample_steps = 128

    rng = np.random.default_rng(SEED)

    test_paths = sorted(list(glob("data/test*.npz")))
    test_ds = SliceDS(test_paths, rng=rng)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    val_sampler = IndexSampler(len(test_ds), shard_opts, shuffle=True, seed=SEED)
    val_dl = DataLoader(data_source=test_ds,
            sampler=val_sampler,
            worker_count=4,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=batch_size, drop_remainder=False)])
    key = random.key(SEED)

    dtype = jnp.float32
    print("loading models and weights")
    
    model_names = ["unet", "adm", "uvit", "dit"]
    modules = {}
    module_params = {}
    for model_name in model_names:
        modules[model_name] = get_model(model_name, dtype=dtype)

        with open(f"weights/{model_name}.pkl", "rb") as f:
            state = pickle.load(f)
            module_params[model_name] = state.params
    
    print("loading batch")
    x,y = next(iter(val_dl))
    x = x * 2 - 1
    y = y * 2 - 1

    samples = [x]
    print("sampling unet...")
    unet_sample = modules["unet"].apply(module_params["unet"], x)
    samples.append(unet_sample)

    # Intentionally use the same keys across all models
    initkey, samplekey = random.split(key)
    diff_models = model_names[1:]
    for model_name in diff_models:
        print(f"sampling {model_name}...")
        params = module_params[model_name]
        module = modules[model_name]

        img = random.normal(initkey, (batch_size, 256, 256, 1))
        sample = ddpm_sample(module=module, params=params, key=samplekey, img=img, condition=x, num_sample_steps=num_sample_steps)
        samples.append(sample)
    samples.append(y)

    plot_names = ["TOF (Source)", "U-Net", "ADM", "U-ViT", "DiT-L/16", "CTA (Ground truth)"]
    samples = jnp.concatenate(samples, axis=0)
    samples = jnp.clip((samples+1) * 0.5, 0., 1.)
    samples = samples.reshape(-1, 256, 256)
    img = utils.make_grid(samples, nrow=len(plot_names), ncol=batch_size)
    utils.save_image(img, "out.png")

if __name__ == "__main__":
    main()
