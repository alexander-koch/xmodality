# Cross-modality image synthesis from TOF-MRA to CTA

This repository contains the code for the paper.

## Installation
```
pip install git+https://github.com/alexander-koch/xmodality.git
cd xmodality
pip install -r requirements.txt
```

## Download the weights

TODO

## Converting some TOF-MRAs

You can either use `inference.py` or `resampled_inference.py`.
Resampling additionally resamples to be in 256x256 range during generation.
Use it like so:

```bash
python3 resampled_inference.py --input <path/to/tof> --output <path/to/cta> --load weights/uvit.pkl --arch uvit --bfloat16
python3 inference.py --input <path/to/tof> --output <path/to/cta> --load weights/uvit.pkl --arch uvit --bfloat16
```

## Training and Evaluation

See [Training](docs/Training.md) for more information on how to train your model.
See [Evaluation](docs/Evaluation.md) for more information on how to evaluate a model.

## Organization

* `figures/` - Files to reproduce and generate data for the figures of the paper
* `extra/` - Files to reproduce and generate data for the appendix
* `sh` - Scripts to run the experiments on the HPC
* `docs` - Files for documentation
* `compute_fd.py` - Computes the Frechet Distance on ViT
* `external_validation.py` - Computes 3D metrics on internal and external dataset

For anything evaluation related, you will need to download the ViT-B/16 weights.

## Contributing

You found a mistake or have an improvement?
Happy to hear, send us your pull requests or file an issue! 

## Citation

If you use our work, here is the citation (TODO)

```bibtex
@misc{koch24xmodality,
    title   = {Cross-modality image synthesis from TOF-MRA to CTA using diffusion-based models}, 
    author  = {Alexander Koch and Orhun Utku Aydin and Adam Hilbert and Jana Rieger and Satoru Tanioka and Fujimaro Ishida and Dietmar Frey},
    year    = {2024},
    eprint  = {TODO},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
