#!/bin/bash

#SBATCH --job-name=eval-unet
#SBATCH --output=logs/%x-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate jax
srun python3 compute_fd.py --arch unet --load weights/unet.pkl --disable_diffusion --output scores/scores_unet.yaml --bfloat16
