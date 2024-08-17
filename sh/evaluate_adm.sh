#!/bin/bash

#SBATCH --job-name=eval-adm
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
#srun python3 compute_fd.py --arch adm --load weights/adm.pkl --sampler ddpm --output scores/scores_adm_ddpm.yaml --bfloat16
srun python3 compute_fd.py --arch adm --load weights/adm.pkl --sampler ddim --output scores/scores_adm_ddim.yaml --bfloat16

