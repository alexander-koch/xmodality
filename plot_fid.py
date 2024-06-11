from glob import glob
from train import SliceDS, sample
from grain.python import DataLoader, ReadOptions, Batch, ShardOptions, IndexSampler
from uvit import UViT
from unet import UNet
from adm import ADM
import matplotlib.pyplot as plt
from jax import random
import pickle
import numpy as np
import jax.numpy as jnp
from torchvision import utils
import torch
import dm_pix as pix

SEED = 1024

def main():
    batch_size = 16
    rng = np.random.default_rng(SEED)

    val_paths = sorted(list(glob("data/val*.npz")))
    val_ds = SliceDS(val_paths, rng=rng)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    val_sampler = IndexSampler(len(val_ds), shard_opts, shuffle=True, seed=SEED)
    val_dl = DataLoader(data_source=val_ds,
            sampler=val_sampler,
            worker_count=4,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=batch_size, drop_remainder=False)])

    key = random.key(SEED)

    dtype = jnp.bfloat16
    print("creating models")
    uvit = UViT(dim=128, channels=1, dtype=dtype)
    #uvit = ADM(dim=128, channels=1, dtype=dtype)

    print("loading weights")
    with open("weights/uvit.pkl", "rb") as f:
        uvit_state = pickle.load(f)
    #with open("weights/adm.pkl", "rb") as f:
    #    uvit_state = pickle.load(f)

    #print("loading image")
    #samples = []
    #for i in range(4):
    x,y = next(iter(val_dl))
    x = x * 2 - 1
    print("x:", x.shape)

    #xs = [8, 32, 128, 1024, 4096, 16384]
    xs = [1,2,3,4,5,6,7,8,10]#[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    scores = []
    results = []
    for steps in xs:
        #steps = (i+1) * 10
        print("steps:", steps)

        #local_scores = []
        #for i in range(4):
        key, initkey, samplekey = random.split(key, 3)
        img = random.normal(initkey, (batch_size, 256, 256, 1))

        sample_result = sample(module=uvit, params=uvit_state.params, key=samplekey, img=img, condition=x, num_sample_steps=steps)
        sample_result = jnp.clip((sample_result+1) * 0.5, 0, 1)
        print("sample:", sample_result.shape)

        score = {
            "mse": jnp.mean(jnp.square(sample_result - y)),
            "mae": jnp.mean(jnp.abs(sample_result - y)),
            "ssim": jnp.mean(pix.ssim(sample_result, y)),
            "psnr": jnp.mean(pix.psnr(sample_result, y))
        }
        #    local_scores.append(score)

        #score = {
        #    "mse": jnp.mean(np.array([s["mse"] for s in local_scores])),
        #    "mae": jnp.mean(np.array([s["mae"] for s in local_scores])),
        #    "ssim": jnp.mean(np.array([s["ssim"] for s in local_scores])),
        #    "psnr": jnp.mean(np.array([s["psnr"] for s in local_scores])),
        #}
        print(score)
        scores.append(score)

        results.append(sample_result)

    psnr_values = np.array([i["psnr"].item() for i in scores])
    mse_values = np.array([i["mse"].item() for i in scores])
    mae_values = np.array([i["mae"].item() for i in scores])
    ssim_values = np.array([i["ssim"].item() for i in scores])

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(xs, psnr_values)
    ax[0,0].set_ylabel("psnr")

    ax[0,1].plot(xs, mse_values)
    ax[0,1].set_ylabel("mse")

    ax[1,0].plot(xs, mae_values)
    ax[1,0].set_ylabel("mae")

    ax[1,1].plot(xs, ssim_values)
    ax[1,1].set_ylabel("ssim")

    #plt.plot(xs, ys)
    plt.savefig("out.png")


if __name__ == "__main__":
    main()
