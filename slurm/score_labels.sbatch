#!/bin/bash
# Teacher-forced scores of every label per clip (scripts/score_labels.py) on one H100.
# Hydra overrides are passed through:
#   sbatch slurm/score_labels.sbatch data.mode=train lora.path=<sft adapter>
#SBATCH --job-name=falldet-score
#SBATCH --partition=gpu-h100
#SBATCH --constraint=LSDF
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

source slurm/env.sh

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "Overrides: $*"

python scripts/score_labels.py num_workers=12 vllm.mm_processor_cache_gb=4 \
    wandb.project=fall-detection-label-scores "$@"
