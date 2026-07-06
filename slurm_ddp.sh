#!/bin/bash

#SBATCH --partition=all
#SBATCH --gres=gpu:8                      # Request N GPUs (change to 2 or 6 as needed)
#SBATCH --time=10-00:00:00
#SBATCH --job-name=U_GRPO_DDP
#SBATCH --output=/mnt/vast/workspaces/Calibrated_Uncertainty/logs/slurm_logs/zeroshot_uncertainty_decomposition/GRPO_DDP-%j.out
#SBATCH --error=/mnt/vast/workspaces/Calibrated_Uncertainty/logs/slurm_logs/zeroshot_uncertainty_decomposition/GRPO_DDP-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -x
echo "$(date '+%d/%m/%Y %H:%M:%S') >>> $(hostname)"

cd /mnt/vast/home/aj45nyja/zeroshot_uncertainty_decomposition/GRPO/
source /mnt/vast/home/aj45nyja/zeroshot_uncertainty_decomposition/.venv/bin/activate
export $(grep -v '^#' .env | xargs)


nvidia-smi

# Number of GPUs must match --gres above
NUM_GPUS=8

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    grpo_ddp.py --error_type "absolute"
