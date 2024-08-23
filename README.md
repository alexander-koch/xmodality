# Cross-modality image synthesis from TOF-MRA to CTA

This repository contains the code for the paper.

## Installation
```
pip install -r requirements.txt
```

## Organization

* `figures/` - Files to reproduce and generate data for the figures of the paper
* `extra/` - Files to reproduce and generate data for the appendix
* `sh` - Scripts to run the experiments on the HPC
* `compute_fd.py` - Computes the Frechet Distance on ViT
* `external_validation.py` - Computes 3D metrics on internal and external dataset

For anything evaluation related, you will need to download the ViT-B/16 weights.
