# Training

## Data preparation

Download the [TopCoW challenge dataset](https://topcow23.grand-challenge.org/data/) or use a custom dataset.

For training, please create a file named `dataset.txt`
with similar format, as the example provided.
The first column is the TOF-MRA path, the second the CTA path.

```csv

topcow_mr_whole_016_0000_Warped.nii.gz,topcow_ct_whole_016_0000.nii.gz
topcow_mr_whole_010_0000_Warped.nii.gz,topcow_ct_whole_010_0000.nii.gz
topcow_mr_whole_027_0000_Warped.nii.gz,topcow_ct_whole_027_0000.nii.gz
...
```

Then, you can run the following
```bash
python3 prepare.py
```

this will create a directory `data` containing the individual slice pairings.
Please adjust the script accordingly, to accomodate your data.

## Training

Verify that you have the correct JAX version installed and that CUDA can be found.
Next you can launch the train script.
All options with explanations and their defaults can be viewed via like so:

```bash
python3 train.py --help
```

To train an ADM model using bfloat16 precision run the following

```bash
python3 train.py --train --arch adm --bfloat16
```

You can log to Weights & Biases using the `--wandb` flag.
Weights can be loaded via the `--load` flag.

See the SLURM scripts in `sh/` for examples.

## Sampling

To check some samples, run with the `--sample` flag instead of `--train`.

```
python3 train.py --sample --load weights/adm.pkl --arch adm --bfloat16
```

