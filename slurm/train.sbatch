#!/bin/bash
# LoRA SFT with one DDP process per allocated GPU. Hydra overrides are passed through:
#   sbatch slurm/train.sbatch training=smoke
#   sbatch --gres=gpu:2 slurm/train.sbatch training=full
#SBATCH --job-name=falldet-sft
#SBATCH --partition=gpu-h100
#SBATCH --constraint=LSDF
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

source slurm/env.sh

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
num_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs: $num_gpus, overrides: $*"

accelerate launch --config_file config/accelerate/ddp_bf16.yaml \
    --num_processes "$num_gpus" scripts/train_sft.py "$@"
