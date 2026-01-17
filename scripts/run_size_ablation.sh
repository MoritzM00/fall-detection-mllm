#!/bin/bash
# Input size ablation study for QwenVL and InternVL models

set -e

SIZES=(448)
MODELS=("qwenvl" "internvl")

for model in "${MODELS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Running $model with size=$size"
        python scripts/vllm_inference.py \
            experiment=zeroshot \
            prompt=baseline \
            model=$model \
            model.params=8B \
            data.size=$size \
            "wandb.tags=[ablation,input_size]"
    done
done
