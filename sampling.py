import jax
from jax import random, jit, numpy as jnp
from einops import repeat
import math
from jax_tqdm import loop_tqdm
from tqdm import tqdm
from functools import partial

def logsnr_schedule_linear(t: jax.Array, clip_min = 1e-9):
    alpha = jnp.clip(1 - t, clip_min, 1.)
    alpha_squared = alpha ** 2
    sigma_squared = 1 - alpha_squared
    return jnp.log(alpha_squared / sigma_squared)

def logsnr_schedule_cosine(
    t: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15
) -> jax.Array:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * jnp.log(jnp.tan(t_min + t * (t_max - t_min))) #+ 2 * jnp.log(64 / 256)

def logsnr_schedule_sigmoid(t: jax.Array, start=0, end=3, tau=.5):
    v_start = jax.nn.sigmoid(start / tau)
    v_end = jax.nn.sigmoid(end / tau)
    output = jax.nn.sigmoid(t * ((end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    alpha = jnp.clip(output, 1e-9, 1)
    alpha_squared = alpha ** 2
    sigma_squared = 1 - alpha_squared
    return jnp.log(alpha_squared/sigma_squared)


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


@partial(jit, static_argnums=(0, 4, 5))
def ddpm_sample(module, params, key, img, num_sample_steps=100, objective="v", **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return p_sample(module, params, keys[i], img, time, time_next, objective=objective, **kwargs)

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
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)
    #eps = sigma * x + alpha * v

    x_start = alpha * x - sigma * pred
    x_start = jnp.clip(x_start, -1, 1)
    eps = (x - alpha * x_start) / sigma

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
    pred = module.apply(params, u_i, time=batch_si_logsnr, **kwargs)
    #eps_si = sigma_si * u_i + alpha_si * v

    x_start = alpha_si * u_i - sigma_si * pred
    x_start = jnp.clip(x_start, -1, 1)
    eps_si = (u_i - alpha_si * x_start) / sigma_si

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
    v = module.apply(params, x, time=batch_log_snr, **kwargs)
    eps = sigma * x + alpha * v

    r1 = 1./3.
    r2 = 2./3.
    h_i = (log_snr_next - log_snr) # in logsnr space
    
    s2im1 = log_snr + r1 * h_i
    s2i = log_snr + r2 * h_i
    
    alpha_s2im1 = jnp.sqrt(jax.nn.sigmoid(s2im1))
    sigma_s2im1 = jnp.sqrt(jax.nn.sigmoid(-s2im1))

    u2im1 = (alpha_s2im1 / alpha) * x - sigma_s2im1 * jnp.expm1(r1 * h_i * 0.5) * eps

    batch_s2im1 = repeat(s2im1, " -> b", b=x.shape[0])
    v = module.apply(params, u2im1, time=batch_s2im1, **kwargs)
    eps2im1 = sigma_s2im1 * x + alpha_s2im1 * v
    d2im1 = eps2im1 - eps

    alpha_s2i = jnp.sqrt(jax.nn.sigmoid(s2i))
    sigma_s2i = jnp.sqrt(jax.nn.sigmoid(-s2i))
    u2i = (alpha_s2i / alpha) * x - sigma_s2i * jnp.expm1(r2 * h_i * 0.5) * eps - (sigma_s2i * r2 / r1) * (jnp.expm1(r2 * h_i * 0.5) / (r2 * h_i) - 1) * d2im1

    batch_s2i = repeat(s2i, " -> b", b=x.shape[0])
    v = module.apply(params, u2i, time=batch_s2i, **kwargs)
    eps2i = sigma_s2i * x + alpha_s2i * v
    d2i = eps2i - eps
    
    return (alpha_next / alpha) * x - sigma_next * jnp.expm1(h_i * 0.5) * eps - (sigma_next / r2) * (jnp.expm1(h_i * 0.5) / (h_i * 0.5) - 1) * d2i


@partial(jit, static_argnums=(0, 4))
def dpm3_sample(
    module, params, key, img, num_sample_steps=100, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return dpm3_sample_step(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

def dpmpp_2s_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
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

    hi = (log_snr_next - log_snr) * 0.5
    si = logsnr_schedule_cosine(time + (time_next - time) * 0.5)
    lam_si = si * 0.5
    alpha_si, sigma_si = jnp.sqrt(jax.nn.sigmoid(si)), jnp.sqrt(jax.nn.sigmoid(-si))

    ri = (lam_si - log_snr * 0.5) / hi
    ui = (sigma_si / sigma) * x - alpha_si * jnp.expm1(-ri * hi) * x_start
    
    # Get start at ui,si
    batch_si = repeat(si, " -> b", b=x.shape[0])
    pred = module.apply(params, ui, time=batch_si, **kwargs)
    if objective == "v":
        x_start_ui = alpha_si * ui - sigma_si * pred
    elif objective == "eps":
        x_start_ui = (ui - sigma_si * pred) / alpha_si
    elif objective == "start":
        x_start_ui = pred
    x_start_ui = jnp.clip(x_start_ui, -1, 1)
    di = (1 - 1 / (2 * ri)) * x_start + 1/(2*ri) * x_start_ui
    return (sigma_next / sigma) * x - alpha_next * jnp.expm1(-hi) * di

@partial(jit, static_argnums=(0, 4))
def dpmpp_2s_sample(
    module, params, key, img, num_sample_steps=100, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return dpmpp_2s_sample_step(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

_samplers = {
    "ddpm": ddpm_sample,
    "ddim": ddim_sample,
    "addim": adjusted_ddim_sample,
    "dpm1": dpm1_sample,
    "dpm2": dpm2_sample,
    "dpm3": dpm3_sample,
    "dpm++2s": dpmpp_2s_sample
}

def get_sampler_names():
    return _samplers.keys()

def get_sampler(name):
    return _samplers[name]


