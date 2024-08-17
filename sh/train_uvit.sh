#!/bin/bash

#SBATCH --job-name=uvit
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
srun python3 train.py --train --wandb --bfloat16 --arch uvit --save weights/uvit.pkl --log_every_n_steps 100
