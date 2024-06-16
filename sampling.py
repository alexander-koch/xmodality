import jax
from jax import random, jit, numpy as jnp
from einops import repeat
import math
from jax_tqdm import loop_tqdm
from tqdm import tqdm
from functools import partial

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

@partial(jit, static_argnums=(0,4))
def ddpm_sample(module, params, key, img, num_sample_steps=100, **kwargs):
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

def slow_sample(module, params, key, img, num_sample_steps=100, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps+1)
    keys = random.split(key, num_sample_steps)
    for i in tqdm(range(num_sample_steps)):
        time = steps[i]
        time_next = steps[i+1]
        img = p_sample(module, params, keys[i], img, time, time_next, **kwargs)
    
    img = jnp.clip(img, -1, 1)
    return img

def ddim_sample_step(module, params, key, x, time, time_next, objective="v", eta=0, **kwargs):
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(log_snr_next)
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(-log_snr_next)

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    ddim_sigma = eta * jnp.sqrt(sigma_next**2 / sigma**2) * jnp.sqrt(1 - alpha**2 / alpha_next ** 2)
    adjusted_sigma = jnp.sqrt(sigma_next ** 2 - ddim_sigma ** 2)

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    v = module.apply(params, x, time=batch_log_snr, **kwargs)

    pred = x * alpha - v * sigma
    eps = x * sigma + v * alpha

    #x = pred * alpha_next + eps * adjusted_sigma
    #if objective == "v":
    #    x_start = alpha * x - sigma * pred
    #elif objective == "eps":
    #    x_start = (x - sigma * pred) / alpha
    #elif objective == "start":
    #    x_start = pred

    #x_start = jnp.clip(x_start, -1, 1)

    #model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
    #posterior_variance = squared_sigma_next * c

    noise = random.normal(key, x.shape)
    return jax.lax.cond(time_next == 0, lambda pred: pred, lambda pred: pred * alpha_next + eps * adjusted_sigma + noise * ddim_sigma, pred)

@partial(jit, static_argnums=(0,4,5))
def ddim_sample(module, params, key, img, num_total_steps=100, num_sample_steps=50, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps+1)
    #steps = jnp.linspace(0, num_total_steps-1, steps=num_sample_steps)
    #print(steps)
    #steps = list(reversed(steps.int().tolist()))
    #step_pairs = list(zip(steps[:-1], steps[1:])) 
    #print(step_pairs)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i+1]
        return ddim_sample_step(module, params, keys[i], img, time, time_next, **kwargs)
    
    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img
