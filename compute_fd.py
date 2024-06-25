import os
# disable tensorflow spamming
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
from glob import glob
from dataset import SliceDS
from sampling import ddpm_sample, ddim_sample
from grain.python import DataLoader, ReadOptions, Batch, ShardOptions, IndexSampler
from models import UViT, UNet, ADM
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

SEED = 42

def main(args):
    use_ema = False
    batch_size = 16
    rng = np.random.default_rng(SEED)
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    sample_fn = ddpm_sample if args.sampler == "ddpm" else ddim_sample

    test_paths = sorted(list(glob("data/test*.npz")))
    test_ds = SliceDS(test_paths, rng=rng)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    test_sampler = IndexSampler(len(test_ds), shard_opts, shuffle=True, seed=SEED)
    test_dl = DataLoader(data_source=test_ds,
            sampler=test_sampler,
            worker_count=4,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=batch_size, drop_remainder=False)])

    key = random.key(SEED)
    if args.arch == "uvit":
        module = UViT(dim=128, channels=1, dtype=dtype)
    elif args.arch == "adm":
        module = ADM(dim=128, channels=1, dtype=dtype)
    elif args.arch == "unet":
        module = UNet(dim=128, channels=1, dtype=dtype)
    else:
        raise ValueError

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
    #xs = [10, 25, 50, 75, 100]
    xs = [16, 32, 64, 128, 256, 1000]
    # xs = [1000, 500, 250, 100, 50]
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

    scores = metric_list
    with open(args.output, "w") as f:
        yaml.dump(scores, f)

    #psnr_values = np.array([i["psnr"] for i in scores])
    #mse_values = np.array([i["mse"] for i in scores])
    #mae_values = np.array([i["mae"] for i in scores])
    #ssim_values = np.array([i["ssim"] for i in scores])
    #fd_values = np.array([i["fd"] for i in scores])

    #fig, ax = plt.subplots(5)
    #ax[0].plot(xs, psnr_values)
    #ax[0].set_ylabel("psnr")

    #ax[1].plot(xs, mse_values)
    #ax[1].set_ylabel("mse")

    #ax[2].plot(xs, mae_values)
    #ax[2].set_ylabel("mae")

    #ax[3].plot(xs, ssim_values)
    #ax[3].set_ylabel("ssim")

    #ax[4].plot(xs, fd_values)
    #ax[4].set_ylabel("fd")

    #plt.savefig("out.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from", required=True)
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--arch", type=str, choices=["unet", "adm", "uvit"], help="architecture", required=True)
    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddpm", help="sampler to use")
    p.add_argument("--output", type=str, help="output path for scores", required=True)
    main(p.parse_args())
