# Evaluation

Download the ViT-B/16 weights from Google's vision transformer repository.

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

## Slice-wise model evaluation

Run the metric score evaluation using the following:

```
python3 compute_fd.py --arch adm --load weights/adm.pkl --output scores.yaml --bfloat16
```

This computes all metrics for the specified model and writes them for each number of sampling steps to a file.
The default sampling steps are [16,32,64,128,256,1000].

For the U-Net baseline use the flag `--disable_diffusion`.

See `compute_fd.py --help` for more options.
See the SLURM `sh/evaluate_*.sh` scripts for example usage.

## Full volume evaluation

To run internal or external full volume validation run the following:

```bash
python3 external_validation.py --arch adm --load weights/adm.pkl --output scores.yaml --batch_size 32 --bfloat16
```

See `external_validation.py --help` for more options.
See the SLURM `sh/evaluate_*_3d.sh` scripts for example usage.

## Plotting

Run either of the following to obtain plots

```bash
python3 figures/plot_fd.py
python3 figures/plot_metrics.py
```

to generate 3D plots run
```
bash figures/generate_samples.sh
python3 figures/plot_samples.py
```

the same applies for the appendix plot, see the  `extra/` folder.

