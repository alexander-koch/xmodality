#!/usr/bin/env python3
"""
Performs TOF2CTA inference using Apple MLX on diffusion transformer model DiT-L/16
"""

import mlx.core as mx
from mlx import nn
import math
from einops.array_api import rearrange, reduce, repeat, pack, unpack
from typing import Union
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
import nibabel as nib
import argparse

def pair(x: Union[int, tuple[int, int]]) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)

def modulate(x, shift, scale):
    return x * (1 + scale.reshape(scale.shape[0], 1, scale.shape[-1])) + shift.reshape(shift.shape[0], 1, shift.shape[-1])

def sinusoidal_embedding_2d(h: int, w: int, dim: int, temperature: int = 10000):
    y, x = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    omega = mx.arange(dim//4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)

    pos = mx.concatenate((
        mx.sin(x), mx.cos(x), mx.sin(y), mx.cos(y)), axis=1)
    return pos.reshape((1,) + pos.shape)

def timestep_embedding(t, dim, max_period = 10000):
    half = dim // 2
    freqs = mx.exp(
        -math.log(max_period) * mx.arange(start=0, stop=half).astype(mx.float32) / half
    )
    args = t.reshape(-1, 1) * freqs
    embedding = mx.concatenate((mx.cos(args), mx.sin(args)), axis=-1)
    if dim % 2:
        embedding = mx.concatenate(
            (embedding, mx.zeros_like(embedding[:, :1])), axis=-1
        )
    return embedding

class SwiGLU(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.gate = nn.Linear(self.dim, self.mlp_dim)
        self.dense_0 = nn.Linear(self.dim, self.mlp_dim)
        self.dense_1 = nn.Linear(self.mlp_dim, self.dim)

    def __call__(self, x):
        x_gate = self.gate(x)
        return self.dense_1(nn.silu(x_gate) * self.dense_0(x))

class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def __call__(self, x):
        b,seq_len,_ = x.shape
        q = self.query(x).reshape(b, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(b, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(b, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        dots = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(dots, axis=-1)
        out = mx.matmul(attn, v).transpose(0, 2, 1, 3).reshape(b, seq_len, self.dim)
        return self.to_out(out)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_dim = hidden_size * mlp_ratio
        self.adaln = nn.Sequential(
            nn.silu,
            nn.Linear(self.hidden_size, self.hidden_size * 6))
        self.rmsnorm_0 = nn.RMSNorm(dims=2, eps=1e-6)
        self.rmsnorm_1 = nn.RMSNorm(dims=2, eps=1e-6)
        self.mha = MultiHeadDotProductAttention(self.hidden_size, self.num_heads)
        self.swiglu = SwiGLU(self.hidden_size, self.mlp_dim)

    def __call__(self, x, c):
        adaln = self.adaln(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln.split(6, axis=1)
        x_hat = modulate(self.rmsnorm_0(x), shift_msa, scale_msa)
        x_hat = self.mha(x_hat)
        x = x + gate_msa.reshape(gate_msa.shape[0], 1, gate_msa.shape[1]) * x_hat
        x_hat = modulate(self.rmsnorm_1(x), shift_mlp, scale_mlp)
        x_hat = self.swiglu(x_hat)
        x = x + gate_mlp.reshape(gate_mlp.shape[0], 1, gate_mlp.shape[1]) * x_hat
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, image_size, patch_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        patch_dim = self.patch_size[0] * self.patch_size[1]
        self.adaln = nn.Sequential(
            nn.silu,
            nn.Linear(self.hidden_size, self.hidden_size * 2))
        self.rmsnorm_0 = nn.RMSNorm(dims=2, eps=1e-6)
        self.dense_0 = nn.Linear(self.hidden_size, patch_dim)

    def __call__(self, x, c):
        adaln = self.adaln(c)
        shift, scale = adaln.split(2, axis=1)
        x = modulate(self.rmsnorm_0(x), shift, scale)
        x = self.dense_0(x)
        h,w = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=h,
            w=w,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        return x

class DiT(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size, depth, num_heads, mlp_ratio=4, frequency_embedding_size=256, out_channels=1):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.frequency_embedding_size = frequency_embedding_size
        self.patch_dim = self.patch_size[0] * self.patch_size[1] * 2
        self.ln_0 = nn.LayerNorm(dims=2, eps=1e-6)
        self.ln_1 = nn.LayerNorm(dims=2, eps=1e-6)
        self.dense_0 = nn.Linear(self.patch_dim, self.hidden_size)
        self.blocks = [DiTBlock(self.hidden_size, self.num_heads, mlp_ratio) for _ in range(self.depth)]
        self.time_proj = nn.Sequential(
            nn.Linear(frequency_embedding_size, self.hidden_size),
            nn.silu,
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.final_layer = FinalLayer(self.hidden_size, self.image_size, self.patch_size)

    def __call__(self, x, time, condition):
        if condition is not None:
            x = mx.concatenate((x, condition), axis=-1)

        b, h, w, c = x.shape
        nh = h // self.patch_size[0]
        nw = w // self.patch_size[1]
        x = rearrange(
            x,
            "b (h p1) (w p2) c -> b h w (p1 p2 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )

        x = self.ln_0(x)
        x = self.dense_0(x)
        x = self.ln_1(x)
        h,w,d = x.shape[1:]
        x = rearrange(x, "b h w c -> b (h w) c")

        pos_emb = sinusoidal_embedding_2d(h,w,d)
        x = x + pos_emb
        t_freq = timestep_embedding(time, self.frequency_embedding_size)
        t_emb = self.time_proj(t_freq)
        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_layer(x, t_emb)
        return x

def logsnr_schedule_cosine(t, logsnr_min: float = -15, logsnr_max: float = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * mx.log(mx.tan(t_min + t * (t_max - t_min)))

def ddpm_sample(dit, x, time, time_next, objective="v", **kwargs):
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -mx.expm1(log_snr - log_snr_next)
    squared_alpha, squared_alpha_next = nn.sigmoid(log_snr), nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = nn.sigmoid(-log_snr), nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        mx.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = dit(x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = mx.clip(x_start, -1, 1)

    model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
    posterior_variance = squared_sigma_next * c
    noise = mx.random.normal(x.shape)

    if time_next == 0:
        return model_mean
    else:
        return model_mean + mx.sqrt(posterior_variance) * noise

def ddim_sample(dit, x, time, time_next, objective="v", **kwargs):
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    squared_alpha, squared_alpha_next = nn.sigmoid(log_snr), nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = nn.sigmoid(-log_snr), nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        mx.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )
    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = dit(x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = mx.clip(x_start, -1, 1)
    model_mean = alpha_next * x_start + (sigma_next / sigma) * (x - alpha * x_start)
    return model_mean

def sample(dit, img, num_sample_steps=100, **kwargs):
    steps = mx.linspace(1.0, 0.0, num_sample_steps + 1)
    for i in tqdm(range(num_sample_steps)):
        time = steps[i]
        time_next = steps[i + 1]
        img = ddpm_sample(dit, img, time, time_next, **kwargs)
        mx.eval(img)

    img = mx.clip(img, -1, 1)
    return img

def top_strip(img):
    # Assume img (h,w,d)
    z = img.shape[-1]-1
    while z > 0 and mx.all(mx.isclose(img[:, :, z], img[:,:,z].min())):
        z = z - 1
    return img[:, :, :z+1], img.shape[-1] - z - 1

def bottom_strip(img):
    # Assume img (h,w,d)
    z = 0
    while z < img.shape[-1] and mx.all(mx.isclose(img[:, :, z], img[:,:,z].min())):
        z = z + 1
    return img[:, :, z:], z

def strip(img):
    img, shift_top = top_strip(img)
    img, shift_bottom = bottom_strip(img)
    return img, shift_bottom, shift_top

def generate(
    dit,
    img,
    batch_size=64,
    num_sample_steps=128,
    order=3,
    prefilter=True
):
    """Takes TOF-MRA image as input (unprocessed) and returns a CT in [-50,350] range"""
    img, lshift, rshift = strip(img)
    z = img.shape[-1]
    target_shape = (256, 256, z)

    # Resample to target resolution
    dsfactor = [w / float(f) for w, f in zip(target_shape, img.shape)]
    img_resampled = zoom(img, zoom=dsfactor, order=order, prefilter=prefilter)
    img_resampled = mx.array(img_resampled)

    # Rescale to [-1,1]
    img_resampled = rearrange(img_resampled, "h w d -> d h w 1")

    min_v = img_resampled.min(axis=(1,2,3), keepdims=True)
    max_v = img_resampled.max(axis=(1,2,3), keepdims=True)
    img_resampled = (img_resampled - min_v) / (max_v - min_v)
    img_resampled = img_resampled * 2 - 1
    num_slices, h, w, _ = img_resampled.shape
    num_iters = math.ceil(num_slices / batch_size)

    out_slices = []
    for i in tqdm(range(num_iters)):
        start = i * batch_size
        if start + batch_size >= num_slices:
            end = num_slices
        else:
            end = start + batch_size
        m = end - start

        init_noise = mx.random.normal(shape=(m, h, w, 1))
        slices = img_resampled[start:end]

        out = sample(dit=dit, img=init_noise, condition=slices, num_sample_steps=num_sample_steps)
        out = out.astype(mx.float32)
        out_slices.append(out)

    out = mx.concatenate(out_slices, axis=0)  # [-1,1]
    out = rearrange(out, "d h w 1 -> h w d")
    out = out[:h, :w]

    # Resample to original resolution
    dsfactor = [1.0 / f for f in dsfactor]
    out = zoom(out, zoom=dsfactor)

    out = np.clip(out, -1, 1)
    out = (out + 1) * 200 - 50  # [-50, 350]
    out = np.pad(out, ((0,0),(0,0),(lshift,rshift)),constant_values=-50)
    return out

def load_from_safetensors(dit, path):
    from safetensors import safe_open
    params = {}
    with safe_open(path, framework="numpy", device="cpu") as f:
        for key in f.keys():
            params[key] = f.get_tensor(key)

    hidden_size = dit.hidden_size
    dit.ln_0.weight = mx.array(params["LayerNorm_0/scale"])
    dit.ln_0.bias = mx.array(params["LayerNorm_0/bias"])
    dit.ln_1.weight = mx.array(params["LayerNorm_1/scale"])
    dit.ln_1.bias = mx.array(params["LayerNorm_1/bias"])

    dit.dense_0.weight = mx.array(params["Dense_0/kernel"]).T
    dit.dense_0.bias = mx.array(params["Dense_0/bias"])
    dit.time_proj.layers[0].weight = mx.array(params["Dense_1/kernel"]).T
    dit.time_proj.layers[0].bias = mx.array(params["Dense_1/bias"])
    dit.time_proj.layers[2].weight = mx.array(params["Dense_2/kernel"]).T
    dit.time_proj.layers[2].bias = mx.array(params["Dense_2/bias"])

    for i, block in enumerate(dit.blocks):
        block.adaln.layers[1].weight = mx.array(params["DiTBlock_{}/Dense_0/kernel".format(i)]).T
        block.adaln.layers[1].bias = mx.array(params["DiTBlock_{}/Dense_0/bias".format(i)])

        block.rmsnorm_0.weight = mx.array(params["DiTBlock_{}/RMSNorm_0/scale".format(i)])
        block.rmsnorm_1.weight = mx.array(params["DiTBlock_{}/RMSNorm_1/scale".format(i)])

        block.mha.query.weight = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/query/kernel".format(i)]).reshape(-1, hidden_size).T
        block.mha.query.bias = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/query/bias".format(i)]).reshape(hidden_size)
        block.mha.key.weight = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/key/kernel".format(i)]).reshape(-1, hidden_size).T
        block.mha.key.bias = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/key/bias".format(i)]).reshape(hidden_size)
        block.mha.value.weight = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/value/kernel".format(i)]).reshape(-1, hidden_size).T
        block.mha.value.bias = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/value/bias".format(i)]).reshape(hidden_size)
        block.mha.to_out.weight = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/out/kernel".format(i)]).reshape(-1, hidden_size).T
        block.mha.to_out.bias = mx.array(params["DiTBlock_{}/MultiHeadDotProductAttention_0/out/bias".format(i)]).reshape(hidden_size)

        block.swiglu.gate.weight = mx.array(params["DiTBlock_{}/SwiGLU_0/Dense_0/kernel".format(i)]).T
        block.swiglu.gate.bias = mx.array(params["DiTBlock_{}/SwiGLU_0/Dense_0/bias".format(i)])
        block.swiglu.dense_0.weight = mx.array(params["DiTBlock_{}/SwiGLU_0/Dense_1/kernel".format(i)]).T
        block.swiglu.dense_0.bias = mx.array(params["DiTBlock_{}/SwiGLU_0/Dense_1/bias".format(i)])
        block.swiglu.dense_1.weight = mx.array(params["DiTBlock_{}/SwiGLU_0/Dense_2/kernel".format(i)]).T
        block.swiglu.dense_1.bias = mx.array(params["DiTBlock_{}/SwiGLU_0/Dense_2/bias".format(i)])

    dit.final_layer.adaln.layers[1].weight = mx.array(params["FinalLayer_0/Dense_0/kernel".format(i)]).T
    dit.final_layer.adaln.layers[1].bias = mx.array(params["FinalLayer_0/Dense_0/bias".format(i)])
    dit.final_layer.rmsnorm_0.weight = mx.array(params["FinalLayer_0/RMSNorm_0/scale".format(i)])
    dit.final_layer.dense_0.weight = mx.array(params["FinalLayer_0/Dense_1/kernel".format(i)]).T
    dit.final_layer.dense_0.bias = mx.array(params["FinalLayer_0/Dense_1/bias".format(i)])

def load_from_hdf5(dit, path):
    # Weights converted from pickle to hdf5
    hidden_size = dit.hidden_size
    with h5py.File(path, "r") as f:
        dit.ln_0.weight = mx.array(f["LayerNorm_0/scale"][:])
        dit.ln_0.bias = mx.array(f["LayerNorm_0/bias"][:])
        dit.ln_1.weight = mx.array(f["LayerNorm_1/scale"][:])
        dit.ln_1.bias = mx.array(f["LayerNorm_1/bias"][:])

        dit.dense_0.weight = mx.array(f["Dense_0/kernel"][:]).T
        dit.dense_0.bias = mx.array(f["Dense_0/bias"][:])
        dit.time_proj.layers[0].weight = mx.array(f["Dense_1/kernel"][:]).T
        dit.time_proj.layers[0].bias = mx.array(f["Dense_1/bias"][:])
        dit.time_proj.layers[2].weight = mx.array(f["Dense_2/kernel"][:]).T
        dit.time_proj.layers[2].bias = mx.array(f["Dense_2/bias"][:])

        for i, block in enumerate(dit.blocks):
            block.adaln.layers[1].weight = mx.array(f["DiTBlock_{}/Dense_0/kernel".format(i)][:]).T
            block.adaln.layers[1].bias = mx.array(f["DiTBlock_{}/Dense_0/bias".format(i)][:])

            block.rmsnorm_0.weight = mx.array(f["DiTBlock_{}/RMSNorm_0/scale".format(i)][:])
            block.rmsnorm_1.weight = mx.array(f["DiTBlock_{}/RMSNorm_1/scale".format(i)][:])

            block.mha.query.weight = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/query/kernel".format(i)][:]).reshape(-1, hidden_size).T
            block.mha.query.bias = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/query/bias".format(i)][:]).reshape(hidden_size)
            block.mha.key.weight = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/key/kernel".format(i)][:]).reshape(-1, hidden_size).T
            block.mha.key.bias = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/key/bias".format(i)][:]).reshape(hidden_size)
            block.mha.value.weight = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/value/kernel".format(i)][:]).reshape(-1, hidden_size).T
            block.mha.value.bias = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/value/bias".format(i)][:]).reshape(hidden_size)
            block.mha.to_out.weight = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/out/kernel".format(i)][:]).reshape(-1, hidden_size).T
            block.mha.to_out.bias = mx.array(f["DiTBlock_{}/MultiHeadDotProductAttention_0/out/bias".format(i)][:]).reshape(hidden_size)

            block.swiglu.gate.weight = mx.array(f["DiTBlock_{}/SwiGLU_0/Dense_0/kernel".format(i)][:]).T
            block.swiglu.gate.bias = mx.array(f["DiTBlock_{}/SwiGLU_0/Dense_0/bias".format(i)][:])
            block.swiglu.dense_0.weight = mx.array(f["DiTBlock_{}/SwiGLU_0/Dense_1/kernel".format(i)][:]).T
            block.swiglu.dense_0.bias = mx.array(f["DiTBlock_{}/SwiGLU_0/Dense_1/bias".format(i)][:])
            block.swiglu.dense_1.weight = mx.array(f["DiTBlock_{}/SwiGLU_0/Dense_2/kernel".format(i)][:]).T
            block.swiglu.dense_1.bias = mx.array(f["DiTBlock_{}/SwiGLU_0/Dense_2/bias".format(i)][:])

        dit.final_layer.adaln.layers[1].weight = mx.array(f["FinalLayer_0/Dense_0/kernel".format(i)][:]).T
        dit.final_layer.adaln.layers[1].bias = mx.array(f["FinalLayer_0/Dense_0/bias".format(i)][:])
        dit.final_layer.rmsnorm_0.weight = mx.array(f["FinalLayer_0/RMSNorm_0/scale".format(i)][:])
        dit.final_layer.dense_0.weight = mx.array(f["FinalLayer_0/Dense_1/kernel".format(i)][:]).T
        dit.final_layer.dense_0.bias = mx.array(f["FinalLayer_0/Dense_1/bias".format(i)][:])

def main(args):
    hidden_size = 1024
    dit = DiT(image_size=256, patch_size=16, hidden_size=hidden_size, depth=24, num_heads=16)
    if args.load.endswith(".safetensors"):
        load_from_safetensors(dit, args.load)
    elif args.load.endswith(".hdf5") or args.load.endswith(".h5"):
        load_from_hdf5(dit, args.load)
    else:
        raise ValueError("unsupported weights file")

    #nn.quantize(dit, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=128, bits=8)

    # Materializes the model
    mx.eval(dit.parameters())

    source = nib.load(args.input)
    header, affine = source.header, source.affine
    source_data = source.get_fdata().astype(np.float32)
    source_data = mx.array(source_data)

    generated_data = generate(dit=dit, img=source_data, batch_size=args.batch_size, num_sample_steps=args.num_sample_steps)
    print("generated:", generated_data.shape, generated_data.min(), generated_data.max(), generated_data.mean())
    out_img = nib.Nifti1Image(generated_data, header=header, affine=affine)
    nib.save(out_img, args.output)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from", required=True)
    p.add_argument("--input", type=str, help="path to image or list of images", required=True)
    p.add_argument("--batch_size", type=int, default=64, help="how many slices to process in parallel")
    p.add_argument("--output", type=str, help="output path", default="out.nii.gz")
    p.add_argument("--num_sample_steps", type=int, help="how many steps to sample for", default=128)
    main(p.parse_args())
