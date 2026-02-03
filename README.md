# Evaluation of Multimodal Large Language Models for Infrequent Action Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/) [![vLLM](https://img.shields.io/badge/inference-vLLM-1e3a5f?logo=python)](https://docs.vllm.ai) [![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd)](https://hydra.cc) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com)

## Project Overview
This project provides code for the master thesis on evaluation of Multimodal Large Language Models (MLLMs) for Infrequent Action Recognition, specifically the detection of Human Falls and the subsequent state of being fallen. We also evaluate MLLMs jointly with general Human Activity classes like `walking` or `standing` to assess models on Human Activity Recognition (HAR).

The main experiments we conduct are:
- Zero-shot: No exemplars are given, just the task instruction
- Few-shot: Few (usually 1-10) video exemplars with associated ground truth label are supplied for In-Context Learning (ICL)
- Chain-of-Thought (CoT): Specifically, Zero-Shot CoT, i.e. no exemplars with reasoning trace are given. The model can come up with its own reasoning trace.

## Quick Start

Requirements:
1. [Setup Environment](#create-the-environment) with conda/uv
2. Set recommended [environment variables](#environment-variables)

The main entrypoint is `scripts/vllm_inference.py` and experiments can be configured using e.g.,`experiment=zeroshot` (the default is `debug`)

To run zero-shot experiments with InternVL3.5-8B, execute
```shell
python scripts/vllm_inference.py experiment=zeroshot model=internvl model.params=8B
```
To run few-shot experiments with Qwen3-VL-4B, execute
```shell
python scripts/vllm_inference.py experiment=fewshot  model=qwenvl model.params=4B
```
To run CoT experiments with the default model, execute
```shell
python scripts/vllm_inference.py experiment=cot
```


### Configuration options

Besides settings experiments, the main configuration options are

1. vLLM configs in `config/vllm` (default: `default`, for faster warmup times, use `debug`)
2. Sampling configs in `config/sampling` (i.e. `greedy`, `qwen3_instruct`)
3. Model configs in `config/model` (default: `qwenvl`)
4. Prompt configs in `config/prompt` (default: `default`) with text-based output and Role Prompt

Other settings include:

5. Data Processing options, i.e. `data.size=224` or `data.split=cv`
6. Hardware settings, notably
    - `batch_size`: specifies how many videos are loaded into memory at once. Reduce of RAM-constrained
    - `num_workers`: Number of worker processed for data loading
7. Wandb logging config, notably
    - `wandb.mode` (online, offline or disabled)
    - `wandb.project` (also configured by `experiment`)


#### Debugging options
- `num_samples` (int): constrain the number of samples used for inference
- `vllm.use_mock` (bool): if True, skip vLLM engine and produce random predictions for debugging purposes that do not depend on vLLM
- `vllm=debug` for faster warm-up times

### Tech Stack
We use the [vLLM](https://docs.vllm.ai/en/latest/) inference engine, optimized for high-throughput and memory-efficient LLM inference with multimodal support.
[Hydra](https://hydra.cc/docs/intro/) is used for configuration management (see above)


## Create the environment
1. Install Conda
2. Run
```shell
conda env create -f environment.yml
conda activate cu129_vllm15
```
3. Install additional dependencies using uv (installed inside colab environment)
```shell
uv pip install vllm==0.15.0 --torch-backend=cu129
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -e .
```

At the time of writing, vLLM is compiled for cu129 by default. If you need a different version of
CUDA, you have to install vLLM from source.

## Environment variables

### Required

```shell
OMNIFALL_ROOT=path/to/omnifall
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### Recommended
These variables should be set **before** launching the vllm inference script.
```shell
CUDA_VISIBLE_DEVICES=0 # or e.g., 0,1
VLLM_CONFIGURE_LOGGING=0
```
