import matplotlib.pyplot as plt
from tueplots import bundles
import yaml
import numpy as np
import seaborn as sns
from tueplots import figsizes, fonts

plt.rcParams.update({"figure.dpi": 200, "axes.linewidth": 0.5})
bundle = bundles.icml2022(family="sans-serif", column="half", usetex=False, nrows=2)
plt.rcParams.update(bundle)

blue="#5e8dfd"
red="#f6433d"
green="#88b16d"
yellow="#f5a93d"

import matplotlib
import colorsys
def scale_lightness(rgb, amount):
    rgb = matplotlib.colors.ColorConverter.to_rgb(rgb)
    c = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

light_red = scale_lightness(red, .7)
light_blue = scale_lightness(blue, .7)
light_yellow = scale_lightness(yellow, .7)

colors=[red, blue, yellow, light_red, light_blue, light_yellow, green]
linestyles=["-", "-", "-", "--", "--", "--", "-"]
markers=["o", "o", "o", "h", "h", "h", "o"]

#with open("scores/scores_unet.yaml", "r") as f:
#    scores_unet = yaml.safe_load(f)

#paths = ["scores_uvit_ddpm.yaml", "scores_adm.yaml", "scores_uvit_ddim.yaml", "scores_adm_ddim.yaml"]
#names = ["U-ViT", "ADM", "U-ViT (ddim)", "ADM (ddim)"]

paths = ["scores/scores_adm_ddpm.yaml",
        "scores/scores_uvit_ddpm.yaml",
        "scores/scores_dit_ddpm.yaml",
        "scores/scores_adm_ddim.yaml",
        "scores/scores_uvit_ddim.yaml",
        "scores/scores_uvit_dpm++2s.yaml"]
        #"scores/scores_dit_ddim.yaml"]
        #"scores/scores_uvit_addim.yaml"]
        #"scores/scores_dit_ddim.yaml"]
        #"scores/scores_dit_ddim.yaml"]

names = ["ADM", "U-ViT", "DiT-L/16", "ADM (DDIM)", "U-ViT (DDIM)", "U-ViT (DPM++2s)"]#, "DiT-L/16 (DDIM)"]
#names = ["ADM (DDPM)", "U-ViT (DDPM)", "DiT-L/16 (DDPM)", "U-ViT (DDIM)"]#, "DiT-L/16 (DDIM)"]

for i, (name, path) in enumerate(zip(names, paths)):
    with open(path, "r") as f:
        scores = yaml.safe_load(f)

    fd = [s["fd"] for s in scores]
    xs = [16, 32, 64, 128, 256, 1000]
    ys = fd
    plt.plot(xs, ys, marker=markers[i], label=name, color=colors[i], markersize=6, linestyle=linestyles[i])

plt.xscale("log")
plt.grid(alpha=0.5, linewidth=0.5)


ax = plt.gca()
ax.tick_params(axis='both', which='both',length=0)
ax.set_xlabel("Number of sampling steps")
ax.set_ylabel("Fréchet distance")

#ax.set_ylim(0.75, 2)

plt.legend(loc="upper right")
plt.savefig("out.png")
