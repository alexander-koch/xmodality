# Cross-modality image synthesis from TOF-MRA to CTA

This repository contains the code for the [paper](https://arxiv.org/abs/2409.10089).
We explore diffusion-based image-to-image translation models to generate synthetic CTA images from TOF-MRA input.

![Figure 1](imgs/figure1.png)

## Installation

We use Python 3.9 for the development.
The code is written in [JAX](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/).

```bash
pip install git+https://github.com/alexander-koch/xmodality.git
cd xmodality
pip install -r requirements.txt
```

You can run the Diffusion Transformer locally on your Mac via [Apple MLX](https://github.com/ml-explore/mlx).
For MLX support, you need an M series chip (Apple silicon) and macOS >= 13.5.

```bash
pip install mlx
```

## Download the weights

Pre-trained weights will be made available soon.

## Converting some TOF-MRAs

You can either use `inference.py` or `resampled_inference.py`.
Resampling additionally resamples to be in 256x256 range during generation.
Use it like this:

```bash
python3 resampled_inference.py --input <path/to/tof> --output <path/to/cta> --load weights/uvit.pkl --arch uvit --bfloat16
```

or run the model on full resolution like this:

```bash
python3 inference.py --input <path/to/tof> --output <path/to/cta> --load weights/uvit.pkl --arch uvit --bfloat16
```

For Mac users, you can run the Diffusion Transformer model locally like this:

```bash
python3 mlx_inference.py --input <path/to/tof> --output <path/to/cta> --load <path/to/weights> --num_sample_steps <num_sample_steps>
```

All scripts print more detailed information on possible settings and parameters by using the `--help` flag, i.e.

```bash
python3 resampled_inference.py --help
```

## Training and Evaluation

See [Training](docs/Training.md) for more information on how to train your model.
See [Evaluation](docs/Evaluation.md) for more information on how to evaluate a model.

## Speed

Inference speed on a 512x512x160 TOF-MRA image using a batch size of 64, float32 precision, DDPM sampling, using the resampled inference script

| Device       | Time (4 steps) | Time (32 steps)   | Time (100 steps)  |
|--------------|----------------|-------------------|-------------------|
| NVIDIA A40   | ~15s           | ~50s              |  ~2min            |
| Apple M3 Pro | ~40s           | ~6min             | ~17min            |

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

## Citation

If you use this project, cite us as follows:

```bibtex
@misc{koch2024xmodality,
      title={Cross-modality image synthesis from {TOF-MRA} to {CTA} using diffusion-based models}, 
      author={Alexander Koch and Orhun Utku Aydin and Adam Hilbert and Jana Rieger and Satoru Tanioka and Fujimaro Ishida and Dietmar Frey},
      year={2024},
      eprint={2409.10089},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2409.10089}, 
}
```

