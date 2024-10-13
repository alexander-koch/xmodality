# Cross-modality image synthesis from TOF-MRA to CTA

This repository contains the code for the paper.

![Figure 1](imgs/figure1.png)

## Installation

We use Python 3.9 for the development.
The code is written in [JAX](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/).
You can run the Diffusion Transformer locally on your Mac via [Apple MLX](https://github.com/ml-explore/mlx).

```bash
pip install git+https://github.com/alexander-koch/xmodality.git
cd xmodality
pip install -r requirements.txt
```

For MLX support, you need an M series chip (Apple silicon) and macOS >= 13.5.

```bash
pip install mlx
```

## Download the weights

Pre-trained weights will be made available soon.

## Converting some TOF-MRAs

You can either use `inference.py` or `resampled_inference.py`.
Resampling additionally resamples to be in 256x256 range during generation.
Use it like so:

```bash
python3 resampled_inference.py --input <path/to/tof> --output <path/to/cta> --load weights/uvit.pkl --arch uvit --bfloat16
```

```bash
python3 inference.py --input <path/to/tof> --output <path/to/cta> --load weights/uvit.pkl --arch uvit --bfloat16
```

For Mac users, you can run the Diffusion Transformer model locally like this:

```bash
python3 mlx_inference.py --input <path/to/tof> --output <path/to/cta> --load <path/to/weights/ --num_sample_steps <num_sample_steps>
```

## Training and Evaluation

See [Training](docs/Training.md) for more information on how to train your model.
See [Evaluation](docs/Evaluation.md) for more information on how to evaluate a model.

## Speed

Inference speed on a 512x512x160 TOF-MRA image using a batch size of 64, float32 precision, DDPM sampling, 100 sample steps using resampled inference

| Device       | Time   |
|--------------|--------|
| Apple M3 Pro | ~17min |
| NVIDIA A40   |  ~2min |

## Organization

* `figures/` - Files to reproduce and generate data for the figures of the paper
* `extra/` - Files to reproduce and generate data for the appendix
* `sh/` - Scripts to run the experiments on the HPC
* `docs/` - Files for documentation
* `scripts/` - Additional scripts
* `compute_fd.py` - Computes the Frechet Distance on ViT
* `external_validation.py` - Computes 3D metrics on internal and external dataset

For anything evaluation related, you will need to download the ViT-B/16 weights.

## Contributing

You found a mistake or have an improvement?
Happy to hear, send us your pull requests or file an issue! 

