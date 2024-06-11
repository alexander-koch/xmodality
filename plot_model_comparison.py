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

SEED = 1729

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
    adm = ADM(dim=128, channels=1, dtype=dtype)
    unet = UNet(dim=128, channels=1, dtype=dtype)

    print("loading weights")
    with open("weights/uvit.pkl", "rb") as f:
        uvit_state = pickle.load(f)

    with open("weights/adm.pkl", "rb") as f:
        adm_state = pickle.load(f)

    #with open("weights/adm_both.pkl", "rb") as f:
    #    adm_state = pickle.load(f)

    with open("weights/unet.pkl", "rb") as f:
        unet_state = pickle.load(f)
    
    print("loading batch")
    x,y = next(iter(val_dl))
    x = x * 2 - 1
    y = y * 2 - 1

    unet_result = unet.apply(unet_state.params, x)
    print("unet:", unet_result.shape)
    
    key, initkey, samplekey = random.split(key, 3)
    img = random.normal(initkey, (batch_size, 256, 256, 1))
    adm_result = sample(module=adm, params=adm_state.params, key=samplekey, img=img, condition=x, num_sample_steps=100)
    print("adm result:", adm_result.shape)

    key, initkey, samplekey = random.split(key, 3)
    img = random.normal(initkey, (batch_size, 256, 256, 1))
    uvit_result = sample(module=uvit, params=uvit_state.params, key=samplekey, img=img, condition=x, num_sample_steps=100)
    print("uvit result:", uvit_result.shape)

    #samples = jnp.concatenate((x, unet_result, adm_result, uvit_result, y), axis=0)
    #samples = jnp.clip((samples+1) * 0.5, 0., 1.)
    #samples = jnp.rot90(samples, -1, (-3, -2))

    names = ["x", "unet", "adm", "uvit", "y"]
    imgs = [x, unet_result, adm_result, uvit_result, y]
    for name, img in zip(names, imgs):
        samples = img
        samples = jnp.clip((samples+1) * 0.5, 0., 1.)
        samples = jnp.rot90(samples, -1, (-3, -2))
        samples = torch.from_numpy(np.array(samples).astype(np.float32)).reshape(-1, 1, 256, 256)
        path = name + ".png"
        utils.save_image(utils.make_grid(samples, nrow=batch_size, normalize=False, padding=1, pad_value=1.0), path)

    #fig, ax = plt.subplots(len(imgs), batch_size, dpi=200)

    #for i in range(len(imgs)):
    #    for j in range(batch_size):
    #        data = (np.array(imgs[i][j,:,:,0]).astype(np.float32) + 1) * 0.5
    #        data = np.clip(data, 0, 1)
    #        ax[i, j].imshow(data, cmap="gray")

    #x = torch.from_numpy(np.array((x + 1) * 0.5).astype(np.float32))
    #unet_result = torch.from_numpy(np.array((unet_result + 1) * 0.5).astype(np.float32))
    #adm_result = torch.from_numpy(np.array((adm_result + 1) * 0.5).astype(np.float32))
    #uvit_result = torch.from_numpy(np.array((uvit_result + 1) * 0.5).astype(np.float32))
    #y = torch.from_numpy(np.array(y).astype(np.float32))

    #grid_x = utils.make_grid(x, nrow=batch_size, normalize=False, padding=1, pad_value=1.0)
    #ax[0].imshow(grid_x)

    #grid_unet = utils.make_grid(unet_result, nrow=batch_size, normalize=False, padding=1, pad_value=1.0)
    #ax[1].imshow(grid_unet)

    #grid_adm = utils.make_grid(adm_result, nrow=batch_size, normalize=False, padding=1, pad_value=1.0)
    #ax[2].imshow(grid_adm)

    #grid_uvit = utils.make_grid(uvit_result, nrow=batch_size, normalize=False, padding=1, pad_value=1.0)
    #ax[3].imshow(grid_uvit)

    #grid_y = utils.make_grid(y, nrow=batch_size, normalize=False, padding=1, pad_value=1.0)
    #ax[4].imshow(grid_y)

    #plt.savefig("out.png")

    #utils.save_image(utils.make_grid(samples, nrow=batch_size, normalize=False, padding=1, pad_value=1.0), "out.png")
if __name__ == "__main__":
    main()
