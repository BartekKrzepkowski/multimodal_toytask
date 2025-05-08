#!/bin/bash
##ATHENA
#SBATCH --job-name=mm-toy-task
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdnnp2-gpu-a100
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
#source src/configs/env_variables.sh

WANDB__SERVICE_WAIT=300 python -m main_JTT