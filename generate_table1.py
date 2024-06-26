import yaml

#paths = ["metrics_unet.yaml", "metrics_adm.yaml", "metrics_uvit.yaml", "metrics_dit.yaml"]
paths = ["scores/scores_unet.yaml"]

names = ["U-Net", "ADM", "U-ViT", "DiT-L/16"]

header="""\\begin{tabular}{@{}llllll@{}}
\\toprule
Model & MSE ($\\downarrow$) & MAE ($\\downarrow$) & PSNR ($\\uparrow$) & SSIM ($\\uparrow$) & FD ($\\downarrow$) \\\\ \\midrule
"""
footer="""
\\bottomrule
\\end{tabular}"""

text=header
for i, (name, path) in enumerate(zip(names, paths)):
    with open(path, "r") as f:
        d = yaml.safe_load(f)

        mse = d['mse']
        mae = d['mae']
        psnr = d['psnr']
        ssim = d['ssim']
        fd = d['fd']

        text += f"{name} & {mse:.2f} & {mae:.2f} & {psnr:.2f} & {ssim:.2f} & {fd:.2f} \\\\"
    if i < len(names)-1:
        text+="\n"

text += footer

print(text)
