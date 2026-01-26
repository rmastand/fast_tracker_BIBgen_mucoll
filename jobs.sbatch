#!/bin/bash
#SBATCH --nodes=1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH --gpus-per-task=1
#SBATCH -q regular
#SBATCH --job-name=test
#SBATCH --mail-user=rmastand@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH -t 48:00:00
#SBATCH -A m3246
#SBATCH --array=0-1

# Define an array of commands
commands=(
    "python train_generator.py -c CATHODE_5 -flow -id all"          # [0]
    "python train_generator.py -c CATHODE_5c -flow -id all -context 1" # [1]
)

# Run the command corresponding to this array task
eval "${commands[$SLURM_ARRAY_TASK_ID]}"
