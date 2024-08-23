#!/bin/bash

#SBATCH --job-name=eval-dit-3d
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
#srun python3 external_validation.py --arch dit --load weights/dit.pkl --output scores_3d_with_fd/scores_dit_ddpm.yaml --bfloat16
#srun python3 external_validation.py --arch dit --load weights/dit.pkl --output scores_3d_with_fd/scores_dit_ddim.yaml --bfloat16 --sampler ddim
srun python3 external_validation.py --arch dit --load weights/dit.pkl --output scores_3d_with_fd/scores_dit_ddpm_external.yaml --external --bfloat16
srun python3 external_validation.py --arch dit --load weights/dit.pkl --output scores_3d_with_fd/scores_dit_ddim_external.yaml --external --bfloat16 --sampler ddim
