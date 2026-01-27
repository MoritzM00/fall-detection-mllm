#!/bin/bash
# Input size ablation study for QwenVL and InternVL models

set -e

SIZES=(224 448 768)
MODELS=("qwenvl")

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
