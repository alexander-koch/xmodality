import matplotlib.pyplot as plt
from tueplots import bundles
import yaml
import numpy as np
import seaborn as sns
from tueplots import figsizes, fonts

plt.rcParams.update({"figure.dpi": 200, "axes.linewidth": 0.5})
bundle = bundles.icml2022(family="sans-serif", usetex=False, nrows=3, ncols=2)
plt.rcParams.update(bundle)

blue="#5e8dfd"
red="#f6433d"
green="#88b16d"
yellow="#f5a93d"

colors=[red, blue, yellow, blue, yellow]

#with open("scores/scores_unet.yaml", "r") as f:
#    scores_unet = yaml.safe_load(f)

#paths = ["scores_uvit_ddpm.yaml", "scores_adm.yaml", "scores_uvit_ddim.yaml", "scores_adm_ddim.yaml"]
#names = ["U-ViT", "ADM", "U-ViT (ddim)", "ADM (ddim)"]

paths = ["scores/scores_adm_ddpm.yaml",
        "scores/scores_uvit_ddpm.yaml",
        "scores/scores_dit_ddpm.yaml"]
        #"scores/scores_uvit_ddim.yaml"]
        #"scores/scores_dit_ddim.yaml"]

names = ["ADM", "U-ViT", "DiT-L/16"]
#names = ["ADM (DDPM)", "U-ViT (DDPM)", "DiT-L/16 (DDPM)", "U-ViT (DDIM)"]#, "DiT-L/16 (DDIM)"]

fig, ax = plt.subplots(2,2, sharex=True)
metric_names = ["mse", "mae", "psnr", "ssim"]
metric_display = ["MSE", "MAE", "PSNR", "SSIM"]

for j,metric_name in enumerate(metric_names):
    u = j % 2
    v = j // 2

    for i, (name, path) in enumerate(zip(names, paths)):
        with open(path, "r") as f:
            scores = yaml.safe_load(f)

        fd = [s[metric_name] for s in scores]
        xs = [16, 32, 64, 128, 256, 1000]
        ys = fd
        ax[u,v].plot(xs, ys, marker="o", label=name, color=colors[i])
        ax[u,v].tick_params(axis='both', which='both',length=0)
        
        if u == 1:
            ax[u,v].set_xlabel("Sampling steps")
        ax[u,v].set_ylabel(metric_display[j])
        ax[u,v].set_xscale("log")
        ax[u,v].grid(alpha=0.5, linewidth=0.5)

#plt.xscale("log")

handles, labels = ax[u,v].get_legend_handles_labels()
#fig.subplots_adjust(bottom=0.3, wspace=0.33)
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 0))

#ax = plt.gca()
#plt.legend(loc="upper right")
#ax.legend(loc='center left', bbox_to_anchor=(1, 1.5))

plt.savefig("out.png")
