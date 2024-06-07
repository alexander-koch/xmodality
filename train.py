import jax
from jax import jit, random, numpy as jnp
import numpy as np
import optax
from einops import rearrange
from unet import UNet
from adm import ADM
from dit import DiT
from uvit import UViT
import math
import argparse
import wandb
import dm_pix as pix
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import utils
from typing import NamedTuple, Optional, Any, Iterable
from tqdm import tqdm
from grain.python import DataLoader, ReadOptions, Batch, ShardOptions, IndexSampler
from einops import reduce, repeat
from functools import partial
import cloudpickle
import pickle
from jax_tqdm import loop_tqdm

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
 
def p_sample(module, params, key, x, time, time_next, objective="v", **kwargs):
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(log_snr_next)
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(-log_snr_next)

    alpha, sigma, alpha_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred

    x_start = jnp.clip(x_start, -1, 1)

    model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
    posterior_variance = squared_sigma_next * c

    noise = random.normal(key, x.shape)
    return jax.lax.cond(time_next == 0, lambda m: m, lambda m: m + jnp.sqrt(posterior_variance) * noise, model_mean)


class SliceDS(Dataset):
    def __init__(self, paths, rng):
        self.paths = paths
        self.rng = rng

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        item = np.load(self.paths[index])
        src_slice = item["src"]
        tgt_slice = item["tgt"]

        h = src_slice.shape[0]
        w = src_slice.shape[1]
        pad_h = 256 - h if h < 256 else 0
        pad_w = 256 - w if w < 256 else 0
        if pad_h > 0 or pad_w > 0:
            src_slice = np.pad(src_slice, ((0,pad_h), (0, pad_w)))
            tgt_slice = np.pad(tgt_slice, ((0,pad_h), (0, pad_w)))
            h = src_slice.shape[0]
            w = src_slice.shape[1]

        x = self.rng.integers(0, h-256) if h-256 > 0 else 0
        y = self.rng.integers(0, w-256) if w-256 > 0 else 0

        src_slice = src_slice[x:x+256, y:y+256]
        tgt_slice = tgt_slice[x:x+256, y:y+256]

        src_slice = np.expand_dims(src_slice, -1)
        tgt_slice = np.expand_dims(tgt_slice, -1)

        return src_slice, tgt_slice

@partial(jit, static_argnums=(0,4))
def sample(module, params, key, img, num_sample_steps=100, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps+1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i+1]
        return p_sample(module, params, keys[i], img, time, time_next, **kwargs)
    
    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

def augment(batch, key):
    lrkey, udkey = random.split(key)
    x,y = batch
    both = jnp.concatenate((x,y), axis=-1)
    batch_size = x.shape[0]

    lr_prob = random.bernoulli(key=lrkey, shape=(batch_size,1,1,1))
    ud_prob = random.bernoulli(key=udkey, shape=(batch_size,1,1,1))

    both = both * lr_prob + (1-lr_prob) * jnp.flip(both, -2)
    both = both * ud_prob + (1-ud_prob) * jnp.flip(both, -3)
    return jnp.split(both, 2, axis=-1)


class TrainingState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    train_loss: jax.Array

SEED = 42

def main(args):
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    print("dtype:", dtype)
    print("config:", args)

    if args.arch == "unet":
        module = UNet(dim=128, channels=1, dtype=dtype)
    elif args.arch == "adm":
        module = ADM(dim=128, channels=1, dtype=dtype)
    elif args.arch == "dit":
        # DiT-L/16
        module = DiT(patch_size=2, hidden_size=1024, depth=16, num_heads=16, in_channels=1, dtype=dtype)
    elif args.arch == "uvit":
        module = UViT(dim=128, channels=1, dtype=dtype)
    else:
        raise ValueError("invalid arch")

    if args.wandb:
        wandb.init(project="xmodality_paper", config=args)

    rng = np.random.default_rng(SEED)
    min_snr_gamma = int(args.min_snr_gamma)
    validate_every_n_steps = int(args.validate_every_n_steps)
    opt = optax.adam(1e-4)

    def base_loss(params, batch, keys):
        x, y = batch
        x = x * 2 - 1
        y = y * 2 - 1
        y_hat = module.apply(params, x)
        return jnp.mean(jnp.square(y_hat - y))

    def diff_loss(params, batch, keys, objective="v", use_minsnr=True):
        x, y = batch
        x = x * 2 - 1
        y = y * 2 - 1

        times = random.uniform(keys[0], (y.shape[0],))
        x_start = y

        noise = random.normal(keys[1], x_start.shape)
        x_noised, log_snr = q_sample(x_start=x_start, times=times, noise=noise)
        model_out = module.apply(params, x_noised, time=log_snr, condition=x)

        if objective == "v":
            padded_log_snr = right_pad_dims_to(x_noised, log_snr)
            alpha, sigma = (
                jnp.sqrt(jax.nn.sigmoid(padded_log_snr)),
                jnp.sqrt(jax.nn.sigmoid(-padded_log_snr)),
            )
            target = alpha * noise - sigma * x_start
        elif objective == "eps":
            target = noise
        elif objective == "start":
            target = x_start

        loss = jnp.square(model_out - target)
        loss = reduce(loss, "b ... -> b", "mean")

        snr = jnp.exp(log_snr)
        if use_minsnr:
            clipped_snr = jnp.clip(snr, a_max=min_snr_gamma)
        else:
            clipped_snr = snr

        if objective == "v":
            loss_weight = clipped_snr / (snr + 1)
        elif objective == "eps":
            loss_weight = clipped_snr / snr
        elif objective == "start":
            loss_weight = clipped_snr

        loss = (loss * loss_weight).mean()
        return loss

    loss_fn = base_loss if args.baseline else partial(diff_loss, objective=args.objective, use_minsnr=not args.disable_minsnr)
    
    @jit
    def update(state, batch, key):
        keys = random.split(key, 2)
        train_loss_value, grads = jax.value_and_grad(loss_fn)(state.params, batch, keys)
        updates, opt_state = opt.update(grads, state.opt_state, params=state.params)
        params = optax.apply_updates(state.params, updates)
        return TrainingState(params, opt_state, train_loss_value)

    train_paths = sorted(list(glob("data/train*.npz")))
    val_paths = sorted(list(glob("data/val*.npz")))
    test_paths = sorted(list(glob("data/test*.npz")))

    train_ds = SliceDS(train_paths, rng=rng)
    val_ds = SliceDS(val_paths, rng=rng)
    test_ds = SliceDS(test_paths, rng=rng)

    batch_size = int(args.batch_size)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    train_sampler = IndexSampler(len(train_ds), shard_opts, shuffle=True, seed=SEED)
    val_sampler = IndexSampler(len(val_ds), shard_opts, shuffle=True, seed=SEED)
    test_sampler = IndexSampler(len(test_ds), shard_opts, shuffle=True, seed=SEED)
    train_dl = DataLoader(data_source=train_ds,
            sampler=train_sampler,
            worker_count=4,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=batch_size, drop_remainder=False)])
    val_dl = DataLoader(data_source=val_ds,
            sampler=val_sampler,
            worker_count=4,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=batch_size, drop_remainder=False)])
    test_dl = DataLoader(data_source=test_ds,
            sampler=test_sampler,
            worker_count=4,
            shard_options=shard_opts,
            read_options=read_opts,
            operations=[Batch(batch_size=batch_size, drop_remainder=False)])
    key = random.key(SEED)
    key, initkey = random.split(key)

    # Setup state
    if args.load is not None:
        with open(args.load, "rb") as f:
            state = pickle.load(f)
        print("train loss:", state.train_loss)
    else:
        x,y = next(iter(train_dl))
        if args.baseline:
            initial_params = module.init(initkey, x)
        else:
            dummy_time = jnp.array([0.0])
            initial_params = module.init(initkey, x, time=dummy_time, condition=y)

        initial_opt_state = opt.init(initial_params)
        state = TrainingState(initial_params, initial_opt_state, None)

    if args.train:
        num_epochs = 400
        train_length = math.ceil(len(train_ds) / batch_size)
        total_steps = train_length * num_epochs
        log_every_n_steps = int(args.log_every_n_steps)
        print("total_steps:", total_steps)

        train_dl = cycle(train_dl)
        for step in (p := tqdm(range(total_steps))):
            key, updatekey, augmentkey = random.split(key, 3)

            batch = next(train_dl)
            batch = augment(batch, augmentkey)

            state = update(state, batch, updatekey)
            p.set_description(f"train_loss: {state.train_loss}")

            if args.wandb and (step % log_every_n_steps == log_every_n_steps-1):
                wandb.log({"train_loss": state.train_loss}, step=step)

            if step % validate_every_n_steps == validate_every_n_steps-1:
                names = ["train", "val"]
                dls = [train_dl, val_dl]
                log = {}
                for name, dl in zip(names, dls):
                    x,y = next(iter(dl))
                    x_scaled = x * 2 - 1
                    y_scaled = y * 2 - 1

                    if args.baseline:
                        y_hat_scaled = module.apply(state.params, x_scaled)
                    else:
                        b = x.shape[0]
                        key, initkey, samplekey = random.split(key, 3)
                        img = random.normal(initkey, (b, 256, 256, 1))
                        y_hat_scaled = sample(module=module, params=state.params, key=samplekey, img=img, condition=x_scaled)
                    
                    y_hat = jnp.clip((y_hat_scaled+1)*0.5, 0, 1)
                    mse = jnp.mean(jnp.square(y - y_hat))
                    mae = jnp.mean(jnp.abs(y - y_hat))
                    ssim = jnp.mean(pix.ssim(y_hat, y))
                    psnr = jnp.mean(pix.psnr(y_hat, y))

                    log[f"{name}/mae"] = mae.item()
                    log[f"{name}/mse"] = mse.item()
                    log[f"{name}/ssim"] = ssim.item()
                    log[f"{name}/psnr"] = psnr.item()

                if args.wandb:
                    wandb.log(log, step=step)
                else:
                    print(log)
                
                if args.save is not None:
                    with open(args.save, "wb") as f:
                        cloudpickle.dump(state, f)
    elif args.sample:
        x,y = next(iter(val_dl))
        x = x * 2 - 1
        y = y * 2 - 1

        batch_size = x.shape[0]
        if args.baseline:
            y_hat = module.apply(state.params, x)
        else:
            key, initkey, samplekey = random.split(key, 3)
            img = random.normal(initkey, (batch_size, 256, 256, 1))
            y_hat = sample(module=module, params=state.params, key=samplekey, img=img, condition=x)

        samples = jnp.concatenate((x,y,y_hat), axis=0)
        samples = jnp.clip((samples+1) * 0.5, 0., 1.)
        samples = torch.from_numpy(np.array(samples)).reshape(-1, 1, 256, 256)
        utils.save_image(utils.make_grid(samples, nrow=batch_size, normalize=False, padding=1, pad_value=1.0), "out.png")


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
    p.add_argument("--arch", type=str, choices=["unet", "adm", "dit", "uvit"], default="adm", help="which architecture to use")
    p.add_argument("--batch_size", type=int, default=16, help="batch size")
    p.add_argument("--objective", type=str, default="v", choices=["v", "eps", "start"], help="diffusion objective")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--baseline", action="store_true", help="use unet baseline")
    p.add_argument("--disable_minsnr", action="store_true", help="disable min snr loss weighting")
    main(p.parse_args())
