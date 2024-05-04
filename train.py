import jax
from jax import jit, random, numpy as jnp
import numpy as np
import optax
from einops import rearrange
from unet import UNet
from dit import DiT
from maxim import maxim
import math
import argparse
import wandb
import dm_pix as pix

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from typing import NamedTuple, Optional, Any, Iterable
from tqdm import tqdm
from grain.python import DataLoader, SequentialSampler, ReadOptions, MapTransform, Batch, ShardOptions, IndexSampler
from einops import reduce, repeat
from functools import partial
import cloudpickle
import pickle

def cycle(dl: Iterable[Any]) -> Any:
    while True:
        it = iter(dl)
        for x in it:
            yield x

def logsnr_schedule_cosine(t: jax.Array, logsnr_min: float =-15, logsnr_max: float =15) -> jax.Array:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * jnp.log(jnp.tan(t_min + t * (t_max - t_min)))

def right_pad_dims_to(x: jax.Array, t: jax.Array) -> jax.Array:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))

def q_sample(x_start: jax.Array, times: jax.Array, noise: jax.Array):
    log_snr = logsnr_schedule_cosine(times)
    log_snr_padded = right_pad_dims_to(x_start, log_snr)
    alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(jax.nn.sigmoid(-log_snr_padded))
    x_noised = x_start * alpha + noise * sigma
    return x_noised, log_snr
 
def p_sample(model_fn, key, x, time, time_next, **kwargs):
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(log_snr_next)
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(-log_snr_next)

    alpha, sigma, alpha_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = model_fn(x, time=batch_log_snr, **kwargs)

    x_start = jnp.clip(alpha * x - sigma * pred, -1, 1)
    model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
    posterior_variance = squared_sigma_next * c

    noise = random.normal(key, x.shape)
    return jax.lax.cond(time_next == 0, lambda m: m, lambda m: m + jnp.sqrt(posterior_variance) * noise, model_mean)


class SliceDS(Dataset):
    def __init__(self, path):
        self.tensor = torch.load(path)
        self.src = self.tensor["src"].cpu().numpy()
        self.tgt = self.tensor["tgt"].cpu().numpy()

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        src_slice, tgt_slice = self.src[index], self.tgt[index]
        return src_slice.reshape(256, 256, 1), tgt_slice.reshape(256, 256, 1)


def tof_transform(x):
    # assumes x is not an empty image
    x = x.reshape(-1, 256 * 256)
    upper_bound = jnp.quantile(x, 0.999, axis=1).reshape(-1, 1)
    x = jnp.clip(x, a_min=0)
    x = jnp.minimum(x, upper_bound) / upper_bound
    x = x.reshape(-1, 256, 256, 1)
    return x

def cta_transform(x):
    # [-50, 350]
    background_mask = (x == 0)
    out = (jnp.clip(x, -50, 350) + 50) / 400
    out = jnp.where(background_mask, 0., out)
    return out

class TrainingState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    train_loss: jax.Array
    key: jax.Array

def main(args):
    if args.arch == "unet":
        module = UNet(128, channels=1)
    elif args.arch == "dit":
        module = DiT(patch_size=8, in_channels=1, dtype=jnp.bfloat16)
    elif args.arch == "maxim":
        module = maxim(variant="S-2", use_bias=False, dtype=jnp.bfloat16)
    else:
        raise ValueError("invalid arch")

    if args.wandb:
        wandb.init(project="modality", config=args)

    src_transform = tof_transform
    tgt_transform = cta_transform
    min_snr_gamma = int(args.min_snr_gamma)
    validate_every_n_steps = int(args.validate_every_n_steps)
    opt = optax.adam(1e-4)

    @jit
    def loss(params, batch):
        x, y = batch
        x = src_transform(x) * 2 - 1
        y = tgt_transform(y) * 2 - 1

        y_hat = module.apply(params, x)
        loss = jnp.mean(jnp.square(y-y_hat))
        return loss

    @jit
    def diff_loss(params, batch, keys):
        x, y = batch
        x = src_transform(x) * 2 - 1
        y = tgt_transform(y) * 2 - 1

        times = random.uniform(keys[0], (y.shape[0],))
        x_start = y

        noise = random.normal(keys[1], x_start.shape)
        x_noised, log_snr = q_sample(x_start=x_start, times=times, noise=noise)

        # Mixup source
        #lam = random.beta(keys[2], 0.8, 0.8, shape=(x.shape[0],))
        #indices = jnp.arange(x.shape[0])
        #perm = random.permutation(keys[3], indices)
        #x_noised = (1-lam[:, None, None, None]) * x_noised[perm] + lam[:, None, None, None] * x_noised
        #log_snr = (1-lam) * log_snr[perm] + lam * log_snr
        #condition = (1-lam[:, None, None, None]) * condition[perm] + lam[:, None, None, None] * condition

        model_out = module.apply(params, x_noised, time=log_snr, condition=x)

        padded_log_snr = right_pad_dims_to(x_noised, log_snr)
        alpha, sigma = (
            jnp.sqrt(jax.nn.sigmoid(padded_log_snr)),
            jnp.sqrt(jax.nn.sigmoid(-padded_log_snr)),
        )
        target = alpha * noise - sigma * x_start


        # Mixup targets
        #target = (1-lam[:, None, None, None]) * target[perm] + lam[:, None, None, None] * target

        loss = jnp.square(model_out - target)
        loss = reduce(loss, "b ... -> b", "mean")

        snr = jnp.exp(log_snr)
        clipped_snr = jnp.clip(snr, a_max=min_snr_gamma)
        loss_weight = clipped_snr / (snr + 1)
        loss = (loss * loss_weight).mean()
        return loss
    
    @jit
    def sample(params, key, batch_size=4, num_sample_steps=100, **kwargs):
        keys = random.split(key, num_sample_steps+1)

        img = random.normal(keys[0], (batch_size, 256, 256, 1))
        steps = jnp.linspace(1.0, 0.0, num_sample_steps+1)
        xs = jnp.stack((steps[:-1], steps[1:]), axis=1)
        model_fn = partial(module.apply, params)

        def f(carry, x):
            img, index = carry
            time, time_next = x
            img = p_sample(model_fn, keys[index+1], img, time, time_next, **kwargs)
            return (img, index+1), img

        # "always scan when you can!"
        init = (img, 0)
        (img, _), process = jax.lax.scan(f, init=init, xs=xs)

        img = jnp.clip(img, -1, 1)
        process = jnp.clip(process, -1, 1)
        return img, process

    @jit
    def update(state, batch):
        keys = random.split(state.key, 3)
        next_state_key = keys[0]

        train_loss_value, grads = jax.value_and_grad(diff_loss)(state.params, batch, keys[1:])
        updates, opt_state = opt.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return TrainingState(params, opt_state, train_loss_value, next_state_key)

    train_ds = SliceDS("../modality/data/train_slices.pt")
    val_ds = SliceDS("../modality/data/val_slices.pt")

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    train_sampler = IndexSampler(len(train_ds), shard_opts, shuffle=True, seed=42)
    val_sampler = IndexSampler(len(val_ds), shard_opts, shuffle=True, seed=42)
    train_dl = DataLoader(data_source=train_ds,
            sampler=train_sampler,
            worker_count=0,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=4, drop_remainder=False)])
    val_dl = DataLoader(data_source=val_ds,
            sampler=val_sampler,
            worker_count=0,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=4, drop_remainder=False)])
    key = random.key(0)

    # Setup state
    if args.load is not None:
        with open(args.load, "rb") as f:
            state = pickle.load(f)
        print("train loss:", state.train_loss)
    else:
        x,y = next(iter(train_dl))
        dummy_time = jnp.array([0.0])
        initial_params = module.init(key, x, time=dummy_time, condition=y)
        initial_opt_state = opt.init(initial_params)
        state = TrainingState(initial_params, initial_opt_state, None, key)

    if args.train:
        train_length = math.ceil(len(train_ds) / 4)
        total_steps = train_length * 400
        log_every_n_steps = int(args.log_every_n_steps)
        print("total_steps:", total_steps)

        train_dl = cycle(train_dl)
        for step in (p := tqdm(range(total_steps))):
            batch = next(train_dl)
            state = update(state, batch)
            p.set_description(f"train_loss: {state.train_loss}")

            if args.wandb and (step % log_every_n_steps == log_every_n_steps-1):
                wandb.log({"train_loss": state.train_loss})

            if step % validate_every_n_steps == validate_every_n_steps-1:
                names = ["train", "val"]
                dls = [train_dl, val_dl]
                log = {}
                for name, dl in zip(names, dls):
                    x,y = next(iter(dl))

                    x = src_transform(x) * 2 - 1
                    y = tgt_transform(y) * 2 - 1
                    y_hat, process = sample(state.params, key, condition=x)

                    mse = jnp.mean(jnp.square(y - y_hat))
                    ssim = jnp.mean(pix.ssim(y_hat, y))
                    psnr = jnp.mean(pix.psnr(y_hat, y))

                    log[f"{name}/mse"] = mse
                    log[f"{name}/ssim"] = ssim
                    log[f"{name}/psnr"] = psnr

                if args.wandb:
                    wandb.log(log)
                else:
                    print(log)
                
                if args.save is not None:
                    with open(args.save, "wb") as f:
                        cloudpickle.dump(state, f)
    elif args.sample:
        x,y = next(iter(val_dl))
        x = src_transform(x) * 2 - 1
        y = tgt_transform(y) * 2 - 1
        y_hat, process = sample(state.params, key, condition=x)

        samples = jnp.concatenate((x,y,y_hat), axis=0)
        samples = jnp.clip((samples+1) * 0.5, 0., 1.)
        samples = torch.from_numpy(np.array(samples)).reshape(-1, 1, 256, 256)
        utils.save_image(utils.make_grid(samples, nrow=4, normalize=False, padding=1, pad_value=1.0), "out.png")

        process = rearrange(process, "t b h w c -> b t h w c")
        process = jnp.clip((process+1) * 0.5, 0., 1.)
        process = process[0]

        process = torch.from_numpy(np.array(process)).reshape(-1, 1, 256, 256)
        utils.save_image(utils.make_grid(process, nrow=25, normalize=False, padding=1, pad_value=1.0), "process.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from")
    p.add_argument("--save", type=str, help="path to save weight to", default="checkpoint.pkl")
    p.add_argument("--train", action="store_true", help="train the model")
    p.add_argument("--sample", action="store_true", help="sample the model")
    p.add_argument("--validate_every_n_steps", type=int, default=1000, help="validate the model every n steps")
    p.add_argument("--log_every_n_steps", type=int, default=20, help="log the metrics every n steps")
    p.add_argument("--min_snr_gamma", type=float, default=5.0, help="set loss weighting for min snr")
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--arch", type=str, choices=["unet", "dit", "maxim"], default="unet", help="which architecture to use")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    main(p.parse_args())
