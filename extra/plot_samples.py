#!/bin/bash
from nilearn.plotting import plot_img, plot_epi
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

from tueplots import figsizes, fonts
from tueplots import bundles
plt.rcParams.update({"figure.dpi": 200, "axes.linewidth": 0.5})
bundle = bundles.icml2022(family="sans-serif", column="half", usetex=False, nrows=3)
plt.rcParams.update(bundle)

# Cut image at the top
ruler = -140

def load_img(path, is_ct=True):
    img = nib.load(path)
    x = img.get_fdata().astype(np.float32)
    #x = x[:,:,:ruler]
    if is_ct:
        x = x.clip(-50, 350)
        x = x + 50
    else:
        x = np.clip(x, a_min=x.min(), a_max=np.quantile(x, 0.999))
    x = nib.Nifti1Image(x, header=img.header, affine=img.affine)
    return x

subject="043"
src_path=f"../modality_data/registered/{subject}_Warped.nii.gz"
tgt_path=f"../modality_data/external/{subject}/CTA.nii.gz"
unet_path = f"extra/samples/{subject}/sample_unet.nii.gz"
adm_path  = f"extra/samples/{subject}/sample_adm.nii.gz"
uvit_path = f"extra/samples/{subject}/sample_uvit.nii.gz"
dit_path  = f"extra/samples/{subject}/sample_dit.nii.gz"

src_img = load_img(src_path, is_ct=False)
tgt_img = load_img(tgt_path)

print("src:", src_img.shape)

#cut_coords = (-5, 14, 25) # 87

cut_coords = (-24, -1, -570) #(305, 225, 129)
annotate = True

plot_img(src_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file=f"extra/samples/{subject}/sample_src.png")
plot_img(tgt_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file=f"extra/samples/{subject}/sample_tgt.png")

unet_img = load_img(unet_path)
plot_img(unet_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file=f"extra/samples/{subject}/sample_unet.png")

adm_img = load_img(adm_path)
plot_img(adm_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file=f"extra/samples/{subject}/sample_adm.png")

uvit_img = load_img(uvit_path)
plot_img(uvit_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file=f"extra/samples/{subject}/sample_uvit.png")

dit_img = load_img(dit_path)
plot_img(dit_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file=f"extra/samples/{subject}/sample_dit.png")
