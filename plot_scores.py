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

colors=[blue, red, yellow, green]

paths = ["scores_uvit.yaml", "scores_adm.yaml", "scores_uvit_ddim.yaml", "scores_adm_ddim.yaml"]
names = ["U-ViT", "ADM", "U-ViT (ddim)", "ADM (ddim)"]

for i, (name, path) in enumerate(zip(names, paths)):
    with open(path, "r") as f:
        scores = yaml.safe_load(f)

    fd = [s["fd"] for s in scores]
    xs = [16, 32, 64, 128, 256, 1000]
    ys = fd
    plt.plot(xs, ys, marker="o", label=name, color=colors[i])

plt.xscale("log")
plt.grid(alpha=0.5, linewidth=0.5)

ax = plt.gca()
ax.tick_params(axis='both', which='both',length=0)
ax.set_xlabel("Number of sampling steps")
ax.set_ylabel("Fréchet distance")

plt.legend(loc="upper right")
plt.savefig("out.png")
