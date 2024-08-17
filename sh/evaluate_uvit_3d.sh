#!/bin/bash

#SBATCH --job-name=eval-uvit-3d
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
#srun python3 external_validation.py --arch uvit --load weights/uvit.pkl --output scores_3d/scores_uvit_ddpm.yaml --bfloat16
#srun python3 external_validation.py --arch uvit --load weights/uvit.pkl --output scores_3d/scores_uvit_ddim.yaml --bfloat16 --sampler ddim
srun python3 external_validation.py --arch uvit --load weights/uvit.pkl --output scores_3d/scores_uvit_ddpm_external.yaml --external --bfloat16
srun python3 external_validation.py --arch uvit --load weights/uvit.pkl --output scores_3d/scores_uvit_ddim_external.yaml --external --bfloat16 --sampler ddim
