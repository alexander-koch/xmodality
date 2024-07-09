import jax
from jax import random, jit, numpy as jnp
from einops import repeat
import math
from jax_tqdm import loop_tqdm
from tqdm import tqdm
from functools import partial


def logsnr_schedule_cosine(
    t: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15
) -> jax.Array:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * jnp.log(jnp.tan(t_min + t * (t_max - t_min)))

def inv_logsnr_schedule_cosine(lam: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return (jnp.arctan(jnp.exp(-lam*0.5)) - t_min) / (t_max - t_min)

def right_pad_dims_to(x: jax.Array, t: jax.Array) -> jax.Array:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))


def q_sample(x_start: jax.Array, times: jax.Array, noise: jax.Array) -> tuple[jax.Array, jax.Array]:
    log_snr = logsnr_schedule_cosine(times)
    log_snr_padded = right_pad_dims_to(x_start, log_snr)
    alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
        jax.nn.sigmoid(-log_snr_padded)
    )
    x_noised = x_start * alpha + noise * sigma
    return x_noised, log_snr


def p_sample(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

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
    return jax.lax.cond(
        time_next == 0,
        lambda m: m,
        lambda m: m + jnp.sqrt(posterior_variance) * noise,
        model_mean,
    )


def _ddpm_sample(module, params, key, img, num_sample_steps=100, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return p_sample(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


@partial(jit, static_argnums=(0, 4))
def ddpm_sample(module, params, key, img, num_sample_steps=100, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return p_sample(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


def slow_sample(module, params, key, img, num_sample_steps=100, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)
    for i in tqdm(range(num_sample_steps)):
        time = steps[i]
        time_next = steps[i + 1]
        img = p_sample(module, params, keys[i], img, time, time_next, **kwargs)

    img = jnp.clip(img, -1, 1)
    return img

@partial(jit, static_argnums=(0, 4))
def ddim_sample(
    module, params, key, img, num_sample_steps=100, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return ddim_sample_step(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


def ddim_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
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
    model_mean = alpha_next * x_start + (sigma_next / sigma) * (x - alpha * x_start)
    return model_mean


def adjusted_ddim_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
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

    xvar = 0.1 / (2 + squared_alpha / squared_sigma)
    eps = (x - alpha * x_start) / sigma
    zs_var = (alpha_next - alpha * sigma_next / sigma) ** 2 * xvar
    
    # Change this for your resolution
    # TOOD: just read out x.shape[1,2]
    d = 256 * 256

    model_mean = alpha_next * x_start + jnp.sqrt(squared_sigma_next + (d/jnp.linalg.norm(eps)**2) * zs_var) * eps
    return model_mean


@partial(jit, static_argnums=(0, 4))
def adjusted_ddim_sample(
    module, params, key, img, num_sample_steps=100, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return adjusted_ddim_sample_step(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

def dpm1_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)
    if objective == "v":
        eps = sigma * x + alpha * pred
    elif objective == "eps":
        eps = pred
    elif objective == "start":
        eps = (x - alpha * pred) / sigma
    h_i = (log_snr_next - log_snr) * 0.5
    return (alpha_next / alpha) * x - sigma_next * jnp.expm1(h_i) * eps


@partial(jit, static_argnums=(0, 4))
def dpm1_sample(
    module, params, key, img, num_sample_steps=100, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return dpm1_sample_step(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


def dpm2_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    v = module.apply(params, x, time=batch_log_snr, **kwargs)
    eps = sigma * x + alpha * v

    #pred = module.apply(params, x, time=batch_log_snr, **kwargs)
    #if objective == "v":
    #    x_start = alpha * x - sigma * pred
    #elif objective == "eps":
    #    x_start = (x - sigma * pred) / alpha
    #elif objective == "start":
    #    x_start = pred
    #x_start = jnp.clip(x_start, -1, 1)
    #eps = (x - alpha * x_start) / sigma

    s_i = (log_snr + log_snr_next) * 0.5
    h_i = (log_snr_next - log_snr) * 0.5

    alpha_si = jnp.sqrt(jax.nn.sigmoid(s_i))
    sigma_si = jnp.sqrt(jax.nn.sigmoid(-s_i))

    #print("alpha:", alpha_si, sigma_si)

    u_i = (alpha_si / alpha) * x - sigma_si * jnp.expm1(h_i * 0.5) * eps
    
    batch_si_logsnr = repeat(s_i, " -> b", b=x.shape[0])
    v = module.apply(params, u_i, time=batch_si_logsnr, **kwargs)
    eps_si = sigma_si * u_i + alpha_si * v

    #if objective == "v":
    #    x_start = alpha_si * u_i - sigma_si * pred
    #elif objective == "eps":
    #    x_start = (u_i - sigma_si * pred) / alpha_si
    #elif objective == "start":
    #    x_start = pred
    #x_start = jnp.clip(x_start, -1, 1)
    #eps_si = (u_i - alpha_si * x_start) / sigma_si
    
    return (alpha_next / alpha) * x - sigma_next * jnp.expm1(h_i) * eps_si


@partial(jit, static_argnums=(0, 4))
def dpm2_sample(
    module, params, key, img, num_sample_steps=100, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return dpm2_sample_step(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


def dpm3_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
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
    eps = (x - alpha * x_start) / sigma

    r1 = 1./3.
    r2 = 2./3.
    h_i = (log_snr_next - log_snr) # in logsnr space
    h_i_half = h_i * 0.5 # in lambda space (half logsnr)
    
    s2_i = log_snr + r1 * h_i
    s2_i_next = log_snr + r2 * h_i
    
    alpha_s2i = jnp.sqrt(jax.nn.sigmoid(s2_i))
    sigma_s2i = jnp.sqrt(jax.nn.sigmoid(-s2_i))

    u2_i = (alpha_s2i / alpha) * x - sigma_s2i * (jnp.exp(r1 * hi_half) - 1) * eps

    
    #return (alpha_next / alpha) * x - sigma_next * (jnp.exp(h_i) - 1) * eps_si
    return None
