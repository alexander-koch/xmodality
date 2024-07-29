#!/bin/bash

from glob import glob
from dataset import SliceDS
from sampling import ddpm_sample, ddim_sample
from models import get_model

import matplotlib.pyplot as plt
from jax import random, numpy as jnp
import pickle
import numpy as np
import utils

SEED = 42

def main():
    num_sample_steps = 128
    rng = np.random.default_rng(SEED)
    key = random.key(SEED)

    test_paths = sorted(list(glob("data/test*.npz")))
    test_ds = SliceDS(test_paths, rng=rng)

    dtype = jnp.float32
    print("loading models and weights")
    
    model_names = ["adm", "uvit", "dit"]
    modules = {}
    module_params = {}
    for model_name in model_names:
        modules[model_name] = get_model(model_name, dtype=dtype)

        with open(f"weights/{model_name}.pkl", "rb") as f:
            state = pickle.load(f)
            module_params[model_name] = state.params
    
    print("loading sample")
    idx = rng.integers(0, len(test_ds))
    x,y = test_ds[idx]
    x = x.reshape(1, 256, 256, 1)
    y = y.reshape(1, 256, 256, 1)
    x = x * 2 - 1
    y = y * 2 - 1

    condition = x
    #steps_list = [1,4,8,32,256,1000]
    steps_list = [1,4,8,32,128,256,1000]
    #steps_list = [1,4,8,32,64,128,256,1000]
    #steps_list = [1,4,8,16,32,64,128,256,1000]
    samples = []

    # Intentionally use the same keys across all models
    initkey, samplekey = random.split(key)
    diff_models = model_names
    for model_name in diff_models:
        params = module_params[model_name]
        module = modules[model_name]

        for steps in steps_list:
            print(f"sampling {model_name} for {steps} steps...")
            img = random.normal(initkey, (1, 256, 256, 1))
            sample = ddpm_sample(module=module, params=params, key=samplekey, img=img, condition=condition, num_sample_steps=steps)
            samples.append(sample)
    plot_names = ["ADM", "U-ViT", "DiT-L/16"]
    samples = jnp.concatenate(samples, axis=0)
    samples = jnp.clip((samples+1) * 0.5, 0., 1.)
    samples = samples.reshape(-1, 256, 256)
    img = utils.make_grid(samples, nrow=len(plot_names), ncol=len(steps_list))
    utils.save_image(img, "ddpm.png")

    # Same for DDIM sampling
    samples = []
    diff_models = model_names
    for model_name in diff_models:
        params = module_params[model_name]
        module = modules[model_name]

        for steps in steps_list:
            print(f"sampling {model_name} for {steps} steps...")
            img = random.normal(initkey, (1, 256, 256, 1))
            sample = ddim_sample(module=module, params=params, key=samplekey, img=img, condition=condition, num_sample_steps=steps)
            samples.append(sample)
    samples = jnp.concatenate(samples, axis=0)
    samples = jnp.clip((samples+1) * 0.5, 0., 1.)
    samples = samples.reshape(-1, 256, 256)
    img = utils.make_grid(samples, nrow=len(plot_names), ncol=len(steps_list))
    utils.save_image(img, "ddim.png")

if __name__ == "__main__":
    main()
