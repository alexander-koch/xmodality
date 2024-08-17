#!/bin/bash

#SBATCH --job-name=eval-uvit
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
#srun python3 compute_fd.py --arch uvit --load weights/uvit.pkl --sampler ddpm --output scores/scores_uvit_ddpm.yaml --bfloat16
#srun python3 compute_fd.py --arch uvit --load weights/uvit.pkl --sampler ddim --output scores/scores_uvit_ddim.yaml --bfloat16
#srun python3 compute_fd.py --arch uvit --load weights/uvit.pkl --sampler addim --output scores/scores_uvit_addim.yaml --bfloat16
#srun python3 compute_fd.py --arch uvit --load weights/uvit.pkl --sampler dpm++2s --output scores/scores_uvit_dpm++2s.yaml --bfloat16

srun python3 compute_fd.py --arch uvit --load weights/uvit.pkl --sampler ddpm --output sigmoid_scores/scores_uvit_ddpm.yaml --bfloat16
srun python3 compute_fd.py --arch uvit --load weights/uvit.pkl --sampler ddim --output sigmoid_scores/scores_uvit_ddim.yaml --bfloat16
