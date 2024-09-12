import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
from jax import jit, random, numpy as jnp
import numpy as np
from models import get_model
import math
import argparse
import wandb
from glob import glob
import yaml
from typing import NamedTuple
from tqdm import tqdm
from grain.python import DataLoader, ReadOptions, Batch, ShardOptions, IndexSampler
from einops import reduce, repeat
from functools import partial
import cloudpickle
import pickle
import utils
from sampling_with_scheds import get_sampler
import matplotlib.pyplot as plt
from dataset import SliceDS
from tueplots import bundles
from tueplots import figsizes, fonts
plt.rcParams.update({"figure.dpi": 500, "axes.linewidth": 0.5})
#bundle = bundles.icml2022(family="sans-serif", usetex=False)
#plt.rcParams.update(bundle)

def logsnr_schedule_linear(t: jax.Array, clip_min = 1e-9):
    alpha = jnp.clip(1 - t, clip_min, 1.)
    alpha_squared = alpha ** 2
    sigma_squared = 1 - alpha_squared
    return jnp.log(alpha_squared / sigma_squared)

def logsnr_schedule_cosine(
    t: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15
) -> jax.Array:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * jnp.log(jnp.tan(t_min + t * (t_max - t_min))) #+ 2 * jnp.log(16 / 256)

def logsnr_schedule_cosine_shifted(
    t: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15
) -> jax.Array:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * jnp.log(jnp.tan(t_min + t * (t_max - t_min))) + 2 * jnp.log(32 / 256)

def logsnr_schedule_sigmoid(t: jax.Array, start=0, end=3, tau=.5):
    v_start = jax.nn.sigmoid(start / tau)
    v_end = jax.nn.sigmoid(end / tau)
    output = jax.nn.sigmoid(t * ((end - start) + start) / tau)

    output = (v_end - output) / (v_end - v_start)
    alpha = jnp.clip(output, 1e-9, 1)
    alpha_squared = alpha ** 2
    sigma_squared = 1 - alpha_squared
    return jnp.log(alpha_squared/sigma_squared)

def logsnr_schedule_laplace(t: jax.Array, mu = .5, b = .5):
    return mu - b * jnp.sign(0.5 - t) * jnp.log(1 - 2 * jnp.abs(t - 0.5))

def main():
    dtype = jnp.bfloat16
    batch_size = 4
    model_name = "adm"
    seed = 42
    num_sample_steps = 128
    module = get_model(name=model_name, dtype=dtype)
    unet = get_model(name="unet", dtype=dtype)

    rng = np.random.default_rng(seed)

    test_paths = sorted(list(glob("data/test*.npz")))
    test_ds = SliceDS(test_paths, rng=rng)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    test_sampler = IndexSampler(len(test_ds), shard_opts, shuffle=True, seed=seed)
    test_dl = DataLoader(
        data_source=test_ds,
        sampler=test_sampler,
        worker_count=4,
        shard_options=shard_opts,
        read_options=read_opts,
        operations=[Batch(batch_size=batch_size, drop_remainder=False)],
    )
    key = random.key(seed)
    key, initkey = random.split(key)

    weights_path = f"weights/{model_name}.pkl"
    with open(weights_path, "rb") as f:
        state = pickle.load(f)

    with open("weights/unet.pkl", "rb") as f:
        unet_state = pickle.load(f)

    x, y = next(iter(test_dl))
    x = x * 2 - 1

    batch_size = x.shape[0]
    params = state.params
    key, initkey, samplekey = random.split(key, 3)
    img = random.normal(initkey, (batch_size, 256, 256, 1))
    sample_fn = get_sampler("ddpm")

    schedules = [
        logsnr_schedule_cosine,
        #logsnr_schedule_linear,
        logsnr_schedule_cosine_shifted,
        partial(logsnr_schedule_sigmoid, tau=0.5),
        partial(logsnr_schedule_sigmoid, tau=0.1),
        partial(logsnr_schedule_sigmoid, tau=0.008),
        #partial(logsnr_schedule_sigmoid, tau=0.007),
        None
    ]

    schedule_names = [
        "cosine",
        "cosine shifted",
        "sigmoid (tau=.5)",
        "sigmoid (tau=.1)",
        "sigmoid (tau=.008)",
        #"sigmoid (tau = 0.007)",
        "reference (unet)"
    ]
    
    #for schedule_name in ["linear", "cosine", "sigmoid"]:
    #for tau in [0.5, 0.1, 0.05, 0.01]:
    #for end in [2.8, 3.2]:
    #for steps in [256, 512, 1024]:

    fig, ax = plt.subplots(4, len(schedules), dpi=200)

    for i, (schedule_name, schedule) in enumerate(zip(schedule_names,schedules)):
        if i == len(schedules) - 1:
            y_hat = unet.apply(unet_state.params, x)
        else:
            y_hat = sample_fn(
                module=module,
                params=params,
                key=samplekey,
                img=img,
                condition=x,
                num_sample_steps=num_sample_steps,
                #schedule=partial(logsnr_schedule_sigmoid, tau=0.007)
                schedule=schedule
                #schedule=logsnr_schedule_cosine
            )
        y_hat = jnp.clip((y_hat + 1) * 0.5, 0.0, 1.0)
        metrics = utils.get_metrics(y_hat, y)

        mse = metrics["mse"]
        mae = metrics["mae"]
        ssim = metrics["ssim"]
        psnr = metrics["psnr"]

        title = f"{schedule_name}\nmse={mse:.3f}\nmae={mae:.3f}\nssim={ssim:.3f}\npsnr={psnr:.3f}"

        ax[0, i].imshow(y_hat[0,:,:,0], cmap="grey")
        ax[0, i].set_title(title, fontsize=8)
        ax[0, i].axis("off")
        
        for j in range(1, 4):
            ax[j, i].imshow(y_hat[j,:,:,0], cmap="grey")
            ax[j, i].axis("off")

    plt.tight_layout()
    plt.savefig("out.pdf")
        
    #samples = jnp.concatenate((x, y, y_hat), axis=0)
    ##samples = jnp.clip((samples + 1) * 0.5, 0.0, 1.0)
    #samples = samples.reshape(-1, 256, 256)
    #img = utils.make_grid(samples, nrow=3, ncol=batch_size)
    #utils.save_image(img, "out.png")

if __name__ == "__main__":
    main()
