#!/bin/bash

#SBATCH --job-name=eval-dit
#SBATCH --output=logs/%x-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate jax
srun python3 compute_fd.py --arch dit --load weights/dit.pkl --sampler ddpm --output scores/scores_dit_ddpm.yaml --bfloat16
srun python3 compute_fd.py --arch dit --load weights/dit.pkl --sampler ddim --output scores/scores_dit_ddim.yaml --bfloat16
