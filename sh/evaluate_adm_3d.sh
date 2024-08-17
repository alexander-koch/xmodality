#!/bin/bash

#SBATCH --job-name=eval-adm-3d
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
#srun python3 external_validation.py --arch adm --load weights/adm.pkl --output scores_3d/scores_adm_ddpm.yaml --batch_size 32 --bfloat16
#srun python3 external_validation.py --arch adm --load weights/adm.pkl --output scores_3d/scores_adm_ddim.yaml --batch_size 32 --bfloat16 --sampler ddim
srun python3 external_validation.py --arch adm --load weights/adm.pkl --output scores_3d/scores_adm_ddpm_external.yaml --batch_size 32 --external --bfloat16
srun python3 external_validation.py --arch adm --load weights/adm.pkl --output scores_3d/scores_adm_ddim_external.yaml --batch_size 32 --external --bfloat16 --sampler ddim
