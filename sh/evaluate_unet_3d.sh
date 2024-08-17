#!/bin/bash

#SBATCH --job-name=eval-unet-3d
#SBATCH --output=logs/%x-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate jax
#srun python3 external_validation.py --arch unet --load weights/unet.pkl --output scores_3d/scores_unet.yaml --disable_diffusion --bfloat16
srun python3 external_validation.py --arch unet --load weights/unet.pkl --output scores_3d/scores_unet_external.yaml --disable_diffusion --external --bfloat16
