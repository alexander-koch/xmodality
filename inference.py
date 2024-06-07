import jax
from jax import random
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
    img = np.maximum(0, img)
    max_v = np.quantile(img, 0.999)
    return img / max_v

def main():
    #dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    #dtype = jnp.float32
    dtype = jnp.bfloat16
    module = UViT(dim=128, channels=1, dtype=dtype)
    with open("old_weights/uvit.pkl", "rb") as f:
        state = pickle.load(f)

    #path = "/fast/work/users/kochal_c/modality_data/final/topcow_mr_whole_016_0000_Warped.nii.gz"
    #tof_brain = nib.load(path).get_fdata().astype(np.float32)
    #tof_brain_slice = transform(tof_brain[:256, :256, 80]) * 2 - 1
    #tof_brain_slice = jnp.expand_dims(tof_brain_slice, (0, -1))
    #print("tof_brain_slice:", tof_brain_slice.shape)

    #key = random.key(0)
    #key, subkey = random.split(key, 2)
    #img = random.normal(subkey, (1, 256, 256, 1))
    #out = sample(module=module, params=state.params, key=key, img=img, condition=tof_brain_slice, num_sample_steps=50)
    #out = (out + 1) * 0.5
    #
    ## Plot
    #out = np.array(out).reshape(256, 256)
    #fig,ax = plt.subplots(1,2)
    #ax[0].imshow(np.array(tof_brain_slice).reshape(256, 256), cmap="gray")
    #ax[1].imshow(out, cmap="gray")
    #plt.savefig("out.png")

    path = "/fast/work/users/kochal_c/modality_data/final/topcow_mr_whole_016_0000_Warped.nii.gz"
    out_path = "test.nii.gz"

    tof_brain = nib.load(path)
    header, affine = tof_brain.header, tof_brain.affine
            
    tof_brain = jnp.array(tof_brain.get_fdata().astype(np.float32))
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
    main()
    
