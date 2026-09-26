#!/bin/bash
# LoRA DPO with one DDP process per allocated GPU. Hydra overrides are passed through:
#   sbatch --time=00:30:00 slurm/dpo.sbatch dpo=smoke
#   sbatch slurm/dpo.sbatch dpo=full preference=confusion 'preference.train_predictions_paths=[...]'
#SBATCH --job-name=falldet-dpo
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
    --num_processes "$num_gpus" scripts/train_dpo.py "$@"
