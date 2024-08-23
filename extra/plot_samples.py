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
ruler = -260

def load_img(path, is_ct=True):
    img = nib.load(path)
    x = img.get_fdata().astype(np.float32)
    x = x[:,:,:ruler]
    if is_ct:
        x = x.clip(-50, 350)
        x = x + 50
    else:
        x = np.clip(x, a_min=x.min(), a_max=np.quantile(x, 0.999))
    x = nib.Nifti1Image(x, header=img.header, affine=img.affine)
    return x

src_path="../modality_data/registered/040_Warped.nii.gz"
tgt_path="../modality_data/external/040/CTA.nii.gz"
unet_path = "extra/samples/sample_unet.nii.gz"
adm_path  = "extra/samples/sample_adm.nii.gz"
uvit_path = "extra/samples/sample_uvit.nii.gz"
dit_path  = "extra/samples/sample_dit.nii.gz"

src_img = load_img(src_path, is_ct=False)
tgt_img = load_img(tgt_path)

print("src:", src_img.shape)

cut_coords = (18, 210, 892)
annotate = False

plot_img(src_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file="extra/samples/sample_src.png")
plot_img(tgt_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file="extra/samples/sample_tgt.png")

unet_img = load_img(unet_path)
plot_img(unet_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file="extra/samples/sample_unet.png")

adm_img = load_img(adm_path)
plot_img(adm_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file="extra/samples/sample_adm.png")

uvit_img = load_img(uvit_path)
plot_img(uvit_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file="extra/samples/sample_uvit.png")

dit_img = load_img(dit_path)
plot_img(dit_img, cmap="grey", cut_coords=cut_coords, radiological=False, black_bg=True, draw_cross=False, annotate=annotate, output_file="extra/samples/sample_dit.png")
