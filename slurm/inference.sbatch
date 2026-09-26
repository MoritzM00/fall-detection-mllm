#!/bin/bash
# vLLM inference / evaluation on one H100. Hydra overrides are passed through:
#   sbatch slurm/inference.sbatch experiment=zeroshot model.params=8B
#SBATCH --job-name=falldet-infer
#SBATCH --partition=gpu-h100
#SBATCH --constraint=LSDF
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

source slurm/env.sh

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "Overrides: $*"

python scripts/vllm_inference.py num_workers=12 "$@"
