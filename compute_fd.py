#!/bin/bash

import os
# disable tensorflow spamming
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# force determinism
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_ops"] = "1"

import yaml
from glob import glob
from dataset import SliceDS
from sampling import get_sampler_names, get_sampler
from grain.python import DataLoader, ReadOptions, Batch, ShardOptions, IndexSampler
from models import get_model
import matplotlib.pyplot as plt
from jax import random
import pickle
import numpy as np
import jax.numpy as jnp
import utils
from tqdm import tqdm
from einops import repeat
from vit import get_b16_model
import math
import argparse

def main(args):
    use_ema = False
    batch_size = 16
    rng = np.random.default_rng(args.seed)
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    sample_fn = get_sampler(args.sampler)
    #sample_fn = ddpm_sample if args.sampler == "ddpm" else ddim_sample

    test_paths = sorted(list(glob("data/test*.npz")))
    test_ds = SliceDS(test_paths, rng=rng)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    test_sampler = IndexSampler(len(test_ds), shard_opts, shuffle=True, seed=args.seed)
    test_dl = DataLoader(data_source=test_ds,
            sampler=test_sampler,
            worker_count=4,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=batch_size, drop_remainder=False)])

    key = random.key(args.seed)
    module = get_model(args.arch, dtype=dtype)
    assert args.load.endswith(".pkl")
    with open(args.load, "rb") as f:
        state = pickle.load(f)
        params = state.ema_params if use_ema else state.params
    
    vit, vit_params = get_b16_model()

    def get_features(img):
        img = repeat(img, "b h w 1 -> b h w c", c=3)
        f = vit.apply(vit_params, img, train=False)
        return f

    metric_list = []
    if args.disable_diffusion:
        evaluator = utils.Evaluator(feature_extractor=get_features)
        test_length = math.ceil(len(test_ds) / batch_size)
        it = iter(test_dl)
        for i in tqdm(range(test_length)):
            x, y = next(it)
            x = x * 2 - 1
            y_hat_scaled = module.apply(params, x)
            y_hat = jnp.clip((y_hat_scaled + 1) * 0.5, 0, 1)
            evaluator.update(y_hat, y)
        metrics = evaluator.calculate()
        metric_list.append(metrics)
    else:
        xs = [16, 32, 64, 128, 256, 1000]
        for steps in xs:
            evaluator = utils.Evaluator(feature_extractor=get_features)
            test_length = math.ceil(len(test_ds) / batch_size)
            it = iter(test_dl)
            for i in tqdm(range(test_length)):
                x, y = next(it)
                x = x * 2 - 1
                batch_size, h, w, _ = x.shape

                key, initkey, samplekey = random.split(key, 3)
                img = random.normal(initkey, (batch_size, h, w, 1))
                y_hat_scaled = sample_fn(
                    module=module,
                    params=params,
                    key=samplekey,
                    img=img,
                    condition=x,
                    num_sample_steps=steps,
                )
                y_hat = jnp.clip((y_hat_scaled + 1) * 0.5, 0, 1)
                evaluator.update(y_hat, y)

            metrics = evaluator.calculate()
            print(steps, metrics)
            metric_list.append(metrics)

    with open(args.output, "w") as f:
        yaml.dump(metric_list, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from", required=True)
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--arch", type=str, choices=["unet", "adm", "uvit", "dit"], help="architecture", required=True)
    p.add_argument("--sampler", type=str, choices=get_sampler_names(), default="ddpm", help="sampler to use")
    p.add_argument("--output", type=str, help="output path for scores", required=True)
    p.add_argument("--disable_diffusion", action="store_true", help="disable diffusion")
    p.add_argument("--seed", type=int, default=42, help="global seed")
    main(p.parse_args())
