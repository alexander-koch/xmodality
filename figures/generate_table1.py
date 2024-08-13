import yaml

#paths = ["metrics_unet.yaml", "metrics_adm.yaml", "metrics_uvit.yaml", "metrics_dit.yaml"]
paths = ["scores/scores_adm_ddpm.yaml", "scores/scores_uvit_ddpm.yaml", "scores/scores_dit_ddpm.yaml"]
paths_ddim = ["scores/scores_adm_ddim.yaml", "scores/scores_uvit_ddim.yaml", "scores/scores_dit_ddim.yaml"]

names = ["ADM", "U-ViT", "DiT-L/16"]

#header="""\\begin{tabular}{@{}llllll@{}}
#\\toprule
#Model & MSE ($\\downarrow$) & MAE ($\\downarrow$) & PSNR ($\\uparrow$) & SSIM ($\\uparrow$) & FD ($\\downarrow$) \\\\ \\midrule
#"""
#footer="""
#\\bottomrule
#\\end{tabular}"""
#
#text=header
#for i, (name, path) in enumerate(zip(names, paths)):
#    with open(path, "r") as f:
#        d = yaml.safe_load(f)[-1]
#
#        mse = d['mse']
#        mae = d['mae']
#        psnr = d['psnr']
#        ssim = d['ssim']
#        fd = d['fd']
#
#        text += f"{name} & {mse:.3f} & {mae:.3f} & {psnr:.3f} & {ssim:.3f} & {fd:.3f} \\\\"
#    if i < len(names)-1:
#        text+="\n"
#
#text += footer
#print(text)

#import pandas as pd
#all_names = names + ["U-Net"]
#all_paths = paths + ["scores/scores_unet.yaml"]
#entries = {}
#for name, path in zip(all_names, all_paths):
#    with open(path, "r") as f:
#        d = yaml.safe_load(f)[-1]
#    entries[name] = d
#
#df = pd.DataFrame.from_dict(entries, orient="index")
#print(df)
#for metric in ["mse", "mae", "psnr", "ssim"]:
#    index =df[metric].argmin()
#    print(df.iloc[index])
#
#import sys
#sys.exit(0)


header="""\\begin{tabular}{@{}lllllll@{}}
\\toprule
Model & Sampler & MSE ($\\downarrow$) & MAE ($\\downarrow$) & PSNR ($\\uparrow$) & SSIM ($\\uparrow$) & FD ($\\downarrow$) \\\\ \\midrule
"""
footer="""
\\bottomrule
\\end{tabular}"""
text=header

name = "U-Net"
path = "scores/scores_unet.yaml"
with open(path, "r") as f:
    d = yaml.safe_load(f)[-1]
mse = d['mse']
mae = d['mae']
psnr = d['psnr']
ssim = d['ssim']
fd = d['fd']
text += f"{name} & N/A & {mse:.3f} & {mae:.3f} & {psnr:.3f} & {ssim:.3f} & {fd:.3f} \\\\ \\cmidrule{{1-7}}"
text+="\n"

for i, (name, path) in enumerate(zip(names, paths)):
    with open(path, "r") as f:
        d = yaml.safe_load(f)[-1]

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
        d = yaml.safe_load(f)[-1]

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
