import yaml
import sys

if sys.argv[1] == "internal":
    print("internal")
    paths = ["scores_3d_with_fd/scores_adm_ddpm.yaml", "scores_3d_with_fd/scores_uvit_ddpm.yaml", "scores_3d_with_fd/scores_dit_ddpm.yaml"]
    paths_ddim = ["scores_3d_with_fd/scores_adm_ddim.yaml", "scores_3d_with_fd/scores_uvit_ddim.yaml", "scores_3d_with_fd/scores_dit_ddim.yaml"]
elif sys.argv[1] == "external":
    print("external")
    paths = ["scores_3d_with_fd/scores_adm_ddpm_external.yaml", "scores_3d_with_fd/scores_uvit_ddpm_external.yaml", "scores_3d_with_fd/scores_dit_ddpm_external.yaml"]
    paths_ddim = ["scores_3d_with_fd/scores_adm_ddim_external.yaml", "scores_3d_with_fd/scores_uvit_ddim_external.yaml", "scores_3d_with_fd/scores_dit_ddim_external.yaml"]
else:
    raise ValueError

names = ["ADM", "U-ViT", "DiT-L/16"]

header="""\\begin{tabular}{@{}lllllll@{}}
\\toprule
Model & Sampler & MSE ($\\downarrow$) & MAE ($\\downarrow$) & PSNR ($\\uparrow$) & SSIM ($\\uparrow$) & FD ($\\downarrow$)  \\\\ \\midrule
"""
footer="""
\\bottomrule
\\end{tabular}"""
text=header

name = "U-Net"
if sys.argv[1] == "internal":
    path = "scores_3d_with_fd/scores_unet.yaml"
elif sys.argv[1] == "external":
    path = "scores_3d_with_fd/scores_unet_external.yaml"
else:
    raise ValueError
with open(path, "r") as f:
    d = yaml.safe_load(f)
mse = d['mse']
mae = d['mae']
psnr = d['psnr']
ssim = d['ssim']
fd = d['fd']
text += f"{name} & N/A & {mse:.3f} & {mae:.3f} & {psnr:.3f} & {ssim:.3f} & {fd:.3f} \\\\ \\cmidrule{{1-7}}"
text+="\n"

for i, (name, path) in enumerate(zip(names, paths)):
    with open(path, "r") as f:
        d = yaml.safe_load(f)

        mse = d['mse']
        mae = d['mae']
        psnr = d['psnr']
        ssim = d['ssim']
        fd = d['fd']

        text += f"{name} & DDPM & {mse:.3f} & {mae:.3f} & {psnr:.3f} & {ssim:.3f} & {fd:.3f} \\\\"
        if i == len(names)-1:
            text += " \\cmidrule{1-7}"
        text+="\n"


for i, (name, path) in enumerate(zip(names, paths_ddim)):
    with open(path, "r") as f:
        d = yaml.safe_load(f)

        mse = d['mse']
        mae = d['mae']
        psnr = d['psnr']
        ssim = d['ssim']
        fd = d['fd']

        text += f"{name} & DDIM & {mse:.3f} & {mae:.3f} & {psnr:.3f} & {ssim:.3f} & {fd:.3f} \\\\"
    if i < len(names)-1:
        text+="\n"

text += footer
print(text)
