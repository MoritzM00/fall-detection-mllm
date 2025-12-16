#!/bin/bash

# hydra multirun does not work well with vLLM yet (causes engine issues), start them separately for now

# Define model parameters to iterate over
params=(2B 4B 8B 32B)

# Define datasets to iterate over
datasets=(oops le2i up-fall)

experiment=zeroshot

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    echo "========================================"
    echo "Running experiments for dataset: $dataset"
    echo "========================================"

    # Iterate over each model parameter
    for param in "${params[@]}"; do
        echo "Running model.params=$param on $dataset"
        python scripts/vllm_inference.py model.params=$param experiment=$experiment dataset/omnifall/video@dataset=$dataset
        # wait for vllm engine shutdown
        sleep 5
    done
    python scripts/vllm_inference.py model=qwen/moe experiment=$experiment dataset/omnifall/video@dataset=$dataset
    sleep 5
done
