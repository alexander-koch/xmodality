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

SEED = 8

def main():
    rng = np.random.default_rng(SEED)
    key = random.key(SEED)

    val_paths = sorted(list(glob("data/val*.npz")))
    val_ds = SliceDS(val_paths, rng=rng)

    dtype = jnp.bfloat16
    print("creating models")
    #uvit = UViT(dim=128, channels=1, dtype=dtype)
    uvit = ADM(dim=128, channels=1, dtype=dtype)

    print("loading weights")
    #with open("weights/uvit.pkl", "rb") as f:
    #    uvit_state = pickle.load(f)
    with open("weights/adm.pkl", "rb") as f:
        uvit_state = pickle.load(f)

    print("loading image")
    samples = []
    for i in range(4):
        idx = rng.integers(0, len(val_ds))
        x,y = val_ds[idx]
        x = jnp.expand_dims(x, 0)
        y = jnp.expand_dims(y, 0)
        x = x * 2 - 1
        y = y * 2 - 1
        batch_size = 1
        print("x:", x.shape)

        key, initkey, samplekey = random.split(key, 3)
        img = random.normal(initkey, (batch_size, 256, 256, 1))
        uvit_result_1 = sample(module=uvit, params=uvit_state.params, key=samplekey, img=img, condition=x, num_sample_steps=1000)
        print("uvit result:", uvit_result_1.shape)

        key, initkey, samplekey = random.split(key, 3)
        img = random.normal(initkey, (batch_size, 256, 256, 1))
        uvit_result_2 = sample(module=uvit, params=uvit_state.params, key=samplekey, img=img, condition=x, num_sample_steps=1000)
        print("uvit result:", uvit_result_2.shape)

        key, initkey, samplekey = random.split(key, 3)
        img = random.normal(initkey, (batch_size, 256, 256, 1))
        uvit_result_3 = sample(module=uvit, params=uvit_state.params, key=samplekey, img=img, condition=x, num_sample_steps=1000)
        print("uvit result:", uvit_result_3.shape)

        key, initkey, samplekey = random.split(key, 3)
        img = random.normal(initkey, (batch_size, 256, 256, 1))
        uvit_result_4 = sample(module=uvit, params=uvit_state.params, key=samplekey, img=img, condition=x, num_sample_steps=1000)
        print("uvit result:", uvit_result_4.shape)
        samples += [x, uvit_result_1, uvit_result_2, uvit_result_3, uvit_result_4]

    #samples = jnp.concatenate((x, uvit_result_1, uvit_result_2, uvit_result_3, y), axis=0)
    samples = jnp.concatenate(samples, axis=0)
    samples = jnp.clip((samples+1) * 0.5, 0., 1.)
    samples = torch.from_numpy(np.array(samples)).reshape(-1, 1, 256, 256)
    utils.save_image(utils.make_grid(samples, nrow=5, normalize=False, padding=1, pad_value=1.0), "out.png")

if __name__ == "__main__":
    main()
