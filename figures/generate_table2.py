#!/usr/bin/env python3
import yaml
import sys

header="""\\begin{tabular}{@{}lllll@{}}
\\toprule
Model & MSE ($\\downarrow$) & MAE ($\\downarrow$) & PSNR ($\\uparrow$) & SSIM ($\\uparrow$) \\\\ \\midrule
"""
footer="""
\\bottomrule
\\end{tabular}"""

text=header

names = ["U-Net", "ADM", "U-ViT", "DiT-L/16"]
internal_paths = ["scores_3d/scores_unet.yaml",
        "scores_3d/scores_adm_ddpm.yaml",
        "scores_3d/scores_uvit_ddpm.yaml",
        "scores_3d/scores_dit_ddpm.yaml"]

external_paths = ["scores_3d/scores_unet_external.yaml",
        "scores_3d/scores_adm_ddpm_external.yaml",
        "scores_3d/scores_uvit_ddpm_external.yaml",
        "scores_3d/scores_dit_ddpm_external.yaml"]

if sys.argv[1] == "internal":
    paths = internal_paths
elif sys.argv[1] == "external":
    paths = external_paths
else:
    raise ValueError

for i, (name, path) in enumerate(zip(names, paths)):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    mse = data["mse"]
    mae = data["mae"]
    psnr = data["psnr"]
    ssim = data["ssim"]
    text += f"{name} & {mse:.3f} & {mae:.3f} & {psnr:.3f} & {ssim:.3f} \\\\"
    if i < len(names) - 1:
        text += "\n"

text += footer
print(text)
