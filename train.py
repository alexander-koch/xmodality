#!/usr/bin/env python3

"""Trains and evaluates a cross-modality diffusion model."""

import jax
from jax import jit, random, numpy as jnp
import numpy as np
import optax
from models import UNet, ADM, DiT, UViT
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
from sampling import q_sample, ddpm_sample, right_pad_dims_to, ddim_sample
from dataset import SliceDS
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"


class TrainingState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    train_loss: jax.Array
    ema_params: dict
    step: int


def main(args):
    print("config:", args)
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    batch_size = int(args.batch_size)
    use_ema = args.ema_decay > 0

    if args.arch == "unet":
        module = UNet(dim=128, channels=1, dtype=dtype)
    elif args.arch == "adm":
        module = ADM(dim=128, channels=1, dtype=dtype)
    elif args.arch == "dit":
        module = DiT(
            patch_size=16,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            in_channels=1,
            dtype=dtype,
        )
    elif args.arch == "uvit":
        module = UViT(dim=128, channels=1, dtype=dtype)
    else:
        raise ValueError("invalid arch")

    if args.wandb:
        wandb.init(project="xmodality_paper", config=args)

    rng = np.random.default_rng(args.seed)
    opt = optax.adam(learning_rate=1e-4)

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
            clipped_snr = jnp.clip(snr, a_max=args.min_snr_gamma)
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

    loss_fn = (
        base_loss
        if args.baseline
        else partial(
            diff_loss, objective=args.objective, use_minsnr=not args.disable_minsnr
        )
    )

    @jit
    def update(state, batch, key):
        keys = random.split(key, 2)
        train_loss_value, grads = jax.value_and_grad(loss_fn)(state.params, batch, keys)
        updates, opt_state = opt.update(grads, state.opt_state, params=state.params)
        params = optax.apply_updates(state.params, updates)

        if use_ema:
            # Only update every 10 steps
            ema_params = jax.lax.cond(state.step % 10 == 9,
                    lambda ema_params: ema_params,
                    lambda ema_params: optax.incremental_update(params, ema_params, step_size=1 - args.ema_decay),
                    state.ema_params)
        else:
            ema_params = params

        return TrainingState(
            params=params,
            opt_state=opt_state,
            train_loss=train_loss_value,
            ema_params=ema_params,
            step=state.step+1
        )

    train_paths = sorted(list(glob("data/train*.npz")))
    val_paths = sorted(list(glob("data/val*.npz")))
    test_paths = sorted(list(glob("data/test*.npz")))

    train_ds = SliceDS(train_paths, rng=rng, aug=args.augment)
    val_ds = SliceDS(val_paths, rng=rng)
    test_ds = SliceDS(test_paths, rng=rng)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)
    train_sampler = IndexSampler(
        len(train_ds), shard_opts, shuffle=True, seed=args.seed
    )
    val_sampler = IndexSampler(len(val_ds), shard_opts, shuffle=True, seed=args.seed)
    test_sampler = IndexSampler(len(test_ds), shard_opts, shuffle=True, seed=args.seed)
    train_dl = DataLoader(
        data_source=train_ds,
        sampler=train_sampler,
        worker_count=4,
        shard_options=shard_opts,
        read_options=read_opts,
        operations=[Batch(batch_size=batch_size, drop_remainder=False)],
    )
    val_dl = DataLoader(
        data_source=val_ds,
        sampler=val_sampler,
        worker_count=4,
        shard_options=shard_opts,
        read_options=read_opts,
        operations=[Batch(batch_size=batch_size, drop_remainder=False)],
    )
    test_dl = DataLoader(
        data_source=test_ds,
        sampler=test_sampler,
        worker_count=4,
        shard_options=shard_opts,
        read_options=read_opts,
        operations=[Batch(batch_size=batch_size, drop_remainder=False)],
    )
    key = random.key(args.seed)
    key, initkey = random.split(key)

    #x,y = next(iter(train_dl))
    #batch_size=x.shape[0]
    #samples = jnp.concatenate((x, y), axis=0)
    #samples = samples.reshape(-1, 256, 256)
    #img = utils.make_grid(samples, nrow=2, ncol=batch_size)
    #utils.save_image(img, "out.png")
    #import sys
    #sys.exit(0)

    # Setup state
    if args.load is not None:
        with open(args.load, "rb") as f:
            state = pickle.load(f)
        print("train loss:", state.train_loss)
        print("steps:", state.step)
    else:
        x, y = next(iter(train_dl))
        if args.baseline:
            initial_params = module.init(initkey, x)
        else:
            dummy_time = jnp.array([0.0])
            initial_params = jit(module.init)(initkey, x, time=dummy_time, condition=y)

        initial_opt_state = opt.init(initial_params)
        state = TrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            train_loss=None,
            ema_params=initial_params,
            step=0,
        )

    if args.train:
        train_dl = utils.cycle(train_dl)
        for step in (p := tqdm(range(args.total_steps))):
            key, updatekey = random.split(key)
            batch = next(train_dl)
            state = update(state, batch, updatekey)
            p.set_description(f"train_loss: {state.train_loss}")

            if args.wandb and (
                step % args.log_every_n_steps == args.log_every_n_steps - 1
            ):
                wandb.log({"train_loss": state.train_loss}, step=step)

            if step % args.validate_every_n_steps == args.validate_every_n_steps - 1:
                names = ["train", "val"]
                dls = [train_dl, val_dl]
                log = {}
                for name, dl in zip(names, dls):
                    x, y = next(iter(dl))
                    x_scaled = x * 2 - 1

                    params = state.ema_params if use_ema else state.params
                    if args.baseline:
                        y_hat_scaled = module.apply(params, x_scaled)
                    else:
                        b = x.shape[0]
                        key, initkey, samplekey = random.split(key, 3)
                        img = random.normal(initkey, (b, 256, 256, 1))
                        y_hat_scaled = ddpm_sample(
                            module=module,
                            params=params,
                            key=samplekey,
                            img=img,
                            condition=x_scaled,
                        )

                    y_hat = jnp.clip((y_hat_scaled + 1) * 0.5, 0, 1)
                    metrics = utils.get_metrics(y_hat, y)
                    for metric_key, metric_value in metrics.items():
                        log[f"{name}/{metric_key}"] = metric_value

                if args.wandb:
                    wandb.log(log, step=step)
                else:
                    print(log)

                if args.save is not None:
                    with open(args.save, "wb") as f:
                        cloudpickle.dump(state, f)
    elif args.sample:
        x, y = next(iter(val_dl))
        x = x * 2 - 1
        y = y * 2 - 1

        batch_size = x.shape[0]
        params = state.ema_params if use_ema else state.params
        if args.baseline:
            y_hat = module.apply(params, x)
        else:
            key, initkey, samplekey = random.split(key, 3)
            img = random.normal(initkey, (batch_size, 256, 256, 1))
            y_hat = ddpm_sample(
                module=module,
                params=params,
                key=samplekey,
                img=img,
                condition=x,
                num_sample_steps=100,
            )

        samples = jnp.concatenate((x, y, y_hat), axis=0)
        samples = jnp.clip((samples + 1) * 0.5, 0.0, 1.0)
        samples = samples.reshape(-1, 256, 256)
        img = utils.make_grid(samples, nrow=3, ncol=batch_size)
        utils.save_image(img, "out.png")

    elif args.evaluate:
        from vit import get_b16_model

        vit, vit_params = get_b16_model()

        def get_features(img):
            img = repeat(img, "b h w 1 -> b h w c", c=3)
            f = vit.apply(vit_params, img, train=False)
            return f

        evaluator = utils.Evaluator(feature_extractor=get_features)

        metric_list = []
        params = state.ema_params if use_ema else state.params
        test_length = math.ceil(len(test_ds) / batch_size)
        it = iter(test_dl)
        for i in tqdm(range(test_length)):
            x, y = next(it)
            x = x * 2 - 1
            batch_size, h, w, _ = x.shape

            if args.baseline:
                y_hat_scaled = module.apply(params, x)
            else:
                key, initkey, samplekey = random.split(key, 3)
                img = random.normal(initkey, (batch_size, h, w, 1))
                y_hat_scaled = ddpm_sample(
                    module=module,
                    params=params,
                    key=samplekey,
                    img=img,
                    condition=x,
                    num_sample_steps=100,
                )

            y_hat = jnp.clip((y_hat_scaled + 1) * 0.5, 0, 1)
            evaluator.update(y_hat, y)

        metrics = evaluator.calculate()
        print("metrics:", metrics)
        with open(args.eval_out_path, "w") as f:
            yaml.dump(metrics, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from")
    p.add_argument(
        "--save", type=str, help="path to save weight to", default="checkpoint.pkl"
    )
    p.add_argument("--train", action="store_true", help="train the model")
    p.add_argument(
        "--evaluate", action="store_true", help="evaluate the model on the test set"
    )
    p.add_argument("--eval_out_path", type=str, default="metrics.yaml", help="where to write evaluation metrics to")
    p.add_argument("--sample", action="store_true", help="sample the model")
    p.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=1000,
        help="validate the model every n steps",
    )
    p.add_argument(
        "--log_every_n_steps",
        type=int,
        default=20,
        help="log the metrics every n steps",
    )
    p.add_argument(
        "--min_snr_gamma",
        type=float,
        default=5.0,
        help="set loss weighting for min snr",
    )
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument(
        "--arch",
        type=str,
        choices=["unet", "adm", "dit", "uvit"],
        default="adm",
        help="which architecture to use",
    )
    p.add_argument("--batch_size", type=int, default=16, help="batch size")
    p.add_argument(
        "--total_steps",
        type=int,
        default=150000,
        help="total number of steps to train for",
    )
    p.add_argument("--seed", type=int, default=42, help="global seed")
    p.add_argument(
        "--objective",
        type=str,
        default="v",
        choices=["v", "eps", "start"],
        help="diffusion objective",
    )
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--baseline", action="store_true", help="use unet baseline")
    p.add_argument(
        "--disable_minsnr", action="store_true", help="disable min snr loss weighting"
    )
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="use exponential moving average of weights",
    )
    p.add_argument("--augment", action="store_true", help="use data augmentation")
    main(p.parse_args())
