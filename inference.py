import jax
from jax import random, vmap
import jax.numpy as jnp
import pickle
from uvit import UViT
import numpy as np
import matplotlib.pyplot as plt
from train import sample
import nibabel as nib
from tqdm import tqdm
import math
from einops import rearrange

def transform(img):
    #img = np.maximum(0, img)
    #max_v = np.quantile(img, 0.999)
    #return img / max_v
    
    min_v = img.min()
    max_v = img.max()
    return (img - min_v) / (max_v - min_v)

def vmap_transform(img):
    return vmap(transform)(img)

def main():
    dtype = jnp.bfloat16
    module = UViT(dim=128, channels=1, dtype=dtype)
    with open("weights/uvit.pkl", "rb") as f:
        state = pickle.load(f)

    path = "/fast/work/users/kochal_c/modality_data/final/topcow_mr_whole_016_0000_Warped.nii.gz"
    out_path = "test.nii.gz"

    tof_brain = nib.load(path)
    header, affine = tof_brain.header, tof_brain.affine
            
    tof_brain = jnp.array(tof_brain.get_fdata().astype(np.float32))
    #tof_brain = vmap_transform(tof_brain) * 2 - 1
    tof_brain = transform(tof_brain) * 2 - 1
    tof_brain = rearrange(tof_brain, "h w b -> b h w 1")
    print("tof_brain:", tof_brain.shape, tof_brain.min(), tof_brain.max())
    num_slices, h, w, _ = tof_brain.shape

    # Padding
    factor = 8
    new_h = math.ceil(h / factor) * factor
    new_w = math.ceil(w / factor) * factor
    pad_h = new_h - h
    pad_w = new_w - w
    tof_brain = jnp.pad(tof_brain, ((0,0), (0, pad_h), (0, pad_w), (0,0)))

    # Batch image
    key = random.key(0)
    num_iters = math.ceil(num_slices / 64)
    keys = random.split(key, num_iters*2)

    out_slices = []
    for i in tqdm(range(num_iters)):
        start = i*64
        if start + 64 >= num_slices:
            end = num_slices
        else:
            end = start + 64
        m = end - start

        img = random.normal(keys[i*2], (m, new_h, new_w, 1))
        tof_brain_slices = tof_brain[start:end]
        out = sample(module=module, params=state.params, key=keys[i*2+1], img=img, condition=tof_brain_slices, num_sample_steps=100)
        out_slices.append(out)
    
    out = jnp.concatenate(out_slices, axis=0) # [-1,1]
    out = rearrange(out, "d h w 1 -> h w d")
    out = out[:h, :w]
    out = (out + 1) * 200 - 50 # [-50, 350]
    print("final:", out.shape, out.min(), out.max())

    img = nib.Nifti1Image(out, header=header, affine=affine)
    nib.save(img, out_path)

if __name__ == "__main__":
    #p = argparse.ArgumentParser(
    #    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    #)
    #p.add_argument("--load", type=str, help="path to load pretrained weights from")
    #p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    #p.add_argument("--input", type=str, help="path to image or list of images")
    #p.add_argument("--output", type=str, help="output path")
    #main(p.parse_args())
    main()
