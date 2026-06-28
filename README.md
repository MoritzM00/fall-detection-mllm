# Video-Based Fall Detection with Multimodal Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)](https://docs.astral.sh/ruff/) [![vLLM](https://img.shields.io/badge/inference-vLLM-1e3a5f?logo=python&style=for-the-badge)](https://docs.vllm.ai) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&style=for-the-badge)](https://pre-commit.com)

This repository contains the code for my master's thesis on video-based fall detection with multimodal large language models (MLLMs).
It evaluates whether video-language models can identify fall events, fallen states, and general human activity classes such as `walking`, `standing`, and `sit_down`
on the OmniFall benchmark [paper](https://arxiv.org/abs/2505.19889).

The codebase supports four main workflows:

- Zero-shot inference with task instructions only.
- Few-shot inference with video exemplars selected by balanced/random sampling or embedding similarity.
- Chain-of-thought prompting for zero-shot reasoning traces.
- LoRA supervised fine-tuning of Qwen3-VL models with TRL `SFTTrainer`.

## Repository Layout

```text
src/falldet/              Python package for data loading, prompting, inference, metrics, and plots
scripts/vllm_inference.py Hydra entrypoint for vLLM inference and embedding generation
scripts/train_sft.py      Hydra entrypoint for LoRA supervised fine-tuning
scripts/build_tensor_cache.py
                          Utility for precomputing deterministic video tensors
scripts/ablations/        Experiment sweep runners
scripts/latex/            Table generation utilities for thesis/report outputs
scripts/plot/             Plotting utilities for saved runs and metrics
config/                   Hydra config groups for datasets, models, prompts, training, and vLLM
tests/                    Pytest suite
notebooks/                Exploratory analysis and development notebooks
```

## Setup

The project is designed around a Conda environment with `uv` inside it. The environment file installs Python, FFmpeg/PyAV, CUDA tooling, `uv`, and build helpers. `make install` then installs vLLM, flash-attn, runtime dependencies, development tools, and the package in editable mode.

```shell
make env
conda activate cu130_vllm20_py312
make install
```

At the time of writing, `make install` installs `vllm==0.20.1` with the `cu130` torch backend and `flash-attn==2.8.3`. If you need a different CUDA stack, adjust the install commands in the `Makefile` or install vLLM from source.

Useful development commands:

```shell
make test       # pytest
make lint       # ruff check
make format     # ruff format
make typecheck  # ty check
```

## Data and Environment Variables

Set dataset roots before running experiments. The default inference and training dataset is `omnifall/video/oops`, so `OMNIFALL_ROOT` is required for the default workflows. `WANFALL_ROOT` is required for WANFALL-only and mixed WANFALL/OMNIFALL configs.

```shell
export OMNIFALL_ROOT=/path/to/omnifall
export WANFALL_ROOT=/path/to/wanfall
```

Dataset labels and splits are referenced through Hugging Face `hf://` paths in the configs, and model checkpoints/processors are loaded from Hugging Face model repositories. Make sure the relevant repositories are accessible in your environment before running real inference or training.

Recommended runtime settings:

```shell
export CUDA_VISIBLE_DEVICES=0      # or 0,1,...
export VLLM_CONFIGURE_LOGGING=0
```

`scripts/vllm_inference.py` sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` when launched directly because vLLM's default fork behavior is not compatible with this workload.

The dataset config groups live under:

- `config/dataset/omnifall/video/`
- `config/dataset/wanfall/video/`
- `config/dataset/combined/video/`
- `config/dataset/*/features/` for feature-based configs used by embedding-classifier utilities

Combined video configs are supported by the dataset factory and training code. The current vLLM inference entrypoint evaluates a single individual dataset at a time; if a combined config contains multiple datasets, it logs a warning and uses the first one.

## Quick Smoke Test

Use the debug experiment first to verify the environment, dataset path, processor resolution, W&B settings, and evaluation wiring without running a full evaluation.

```shell
python scripts/vllm_inference.py experiment=debug vllm.use_mock=true
```

Remove `vllm.use_mock=true` when you want to test real model inference. In mock mode, the script skips the real vLLM engine and produces synthetic predictions. The debug preset processes 10 samples, uses the debug vLLM config, disables W&B, and writes local logs under `logs/`.

## Inference

The main inference entrypoint is:

```shell
python scripts/vllm_inference.py [Hydra overrides]
```

The root inference config is `config/inference_config.yaml`. It composes these defaults:

- `dataset=omnifall/video/oops`
- `vllm=default`
- `sampling=qwen3_instruct`
- `model=qwenvl`
- `prompt=default`
- `lora=none`
- `experiment=zeroshot`

Run the core experiment presets with:

```shell
# Zero-shot classification with InternVL3.5-8B
python scripts/vllm_inference.py experiment=zeroshot model=internvl model.params=8B

# Few-shot classification with balanced random exemplar selection
python scripts/vllm_inference.py experiment=fewshot model=qwenvl model.params=4B

# Few-shot classification with similarity-based exemplar retrieval
python scripts/vllm_inference.py experiment=fewshot_similarity model=qwenvl model.params=8B

# Zero-shot chain-of-thought prompting
python scripts/vllm_inference.py experiment=zeroshot_cot
```

Common inference overrides:

```shell
python scripts/vllm_inference.py num_samples=25
python scripts/vllm_inference.py dataset=omnifall/video/cmdfall
python scripts/vllm_inference.py data.split=cv
python scripts/vllm_inference.py data.size=224
python scripts/vllm_inference.py batch_size=8 num_workers=2
python scripts/vllm_inference.py experiment=fewshot prompt.shot_selection=random
python scripts/vllm_inference.py wandb.mode=offline
python scripts/vllm_inference.py vllm=debug
python scripts/vllm_inference.py vllm.use_mock=true
```

Predictions are written as JSONL files when `save_predictions=true`:

```text
outputs/predictions/<wandb-project>/<run_id>.jsonl
```

Evaluation results are written under:

```text
outputs/evaluation_results/<wandb-project>/
```

## Video Tensor Caching

Video preprocessing is deterministic if `data.seed` is set (not None), so decoded and transformed tensors can be cached across runs. There are two cache layers.

### Disk Cache

The disk cache stores preprocessed tensors as `.pt` files. Build it before inference if you want cache hits with the default `data.cache_read_only=true`. On a cache miss, inference still decodes the video; it just does not write new cache entries.

```shell
python scripts/build_tensor_cache.py experiment=zeroshot data.cache_dir=outputs/tensor_cache
python scripts/vllm_inference.py experiment=zeroshot data.cache_dir=outputs/tensor_cache
```

Each dataset, split, and mode combination gets an isolated namespace. Changes to preprocessing parameters such as `num_frames`, `model_fps`, or `data.size` use a different namespace and avoid stale cache entries.

### In-Memory Cache

The in-memory cache is useful for few-shot exemplar corpora because the same train videos are loaded repeatedly across batches.

```shell
python scripts/vllm_inference.py experiment=fewshot data.cache_in_memory=true
```

The memory cache is applied to the exemplar corpus, not to the test dataloader. Both cache layers can be combined. Exemplar loading checks memory first, then disk, then falls back to video decoding.

## Similarity-Based Few-Shot Retrieval

Similarity-based few-shot uses precomputed embeddings for both the exemplar corpus and the query split. Generate train embeddings first:

```shell
python scripts/vllm_inference.py experiment=embed data.mode=train
```

Then generate embeddings for the split you want to evaluate, usually `test`:

```shell
python scripts/vllm_inference.py experiment=embed data.mode=test
```

The `embed` preset switches to `task=embed`, uses `model=qwenvl_embed`, defaults to `data.mode=train`, and writes embeddings under:

```text
outputs/embeddings/
```

Then run retrieval-based few-shot inference:

```shell
python scripts/vllm_inference.py experiment=fewshot_similarity
```

The preset reads embeddings from `outputs/embeddings` and uses `embedding_model_name=Qwen3-VL-Embedding-2B` unless overridden. 
Embedding filenames are derived from the dataset name, mode, frame count, model FPS, embedding model name, and input size, for example:

```text
OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt
OOPS_cs_test_16@7_5_Qwen3-VL-Embedding-2B_448.pt
```

## Supervised Fine-Tuning

Fine-tuning is handled by `scripts/train_sft.py`. It uses Hydra, TRL `SFTTrainer`, PEFT LoRA, and the same dataset, model, and prompt config groups as inference.

```shell
python scripts/train_sft.py                # default: training=quick
python scripts/train_sft.py training=smoke # short wiring check
python scripts/train_sft.py training=full  # full preset
```

Common training overrides:

```shell
python scripts/train_sft.py model.params=4B
python scripts/train_sft.py wandb.mode=offline
python scripts/train_sft.py training.max_steps=20
python scripts/train_sft.py training.attn_implementation=null
```

Single-source dataset groups live in `config/dataset/omnifall/video/`. Mixed groups live in `config/dataset/combined/video/`.

```shell
python scripts/train_sft.py training=full dataset=omnifall/video/oops
python scripts/train_sft.py training=full dataset=omnifall/video/staged-cs
python scripts/train_sft.py training=full dataset=omnifall/video/staged-cv
python scripts/train_sft.py training=full dataset=omnifall/video/staged-oops
python scripts/train_sft.py training=full dataset=omnifall/video/all
python scripts/train_sft.py training=full dataset=combined/video/wanfall-rand-staged-cs-oops
```

By default, `dataset_val` mirrors `dataset`. Override it independently with Hydra's package syntax:

```shell
python scripts/train_sft.py training=full \
    dataset=omnifall/video/staged-cs \
    dataset@dataset_val=omnifall/video/cmdfall
```

For staged-only training, evaluating on `cmdfall` avoids leakage through the staged train split. Mixed-training runs can use the matching `*-cmdfall` evaluation groups, such as `omnifall/video/oops-cmdfall` or `combined/video/wanfall-oops-cmdfall`.

Training outputs are written to:

```text
outputs/training/<run_name>/
outputs/training/<run_name>/adapter
```

Load a trained adapter during inference with the `lora` config group:

```shell
python scripts/vllm_inference.py \
    model.params=8B \
    lora.path=outputs/training/<run_name>/adapter \
    lora.max_rank=8
```

## Multi-GPU Training

Single-node DDP with bf16:

```shell
accelerate launch --config_file config/accelerate/ddp_bf16.yaml \
    --num_processes 4 scripts/train_sft.py training=quick

torchrun --nproc_per_node=4 scripts/train_sft.py training=quick
```

DeepSpeed ZeRO-2 with optimizer and gradient sharding:

```shell
accelerate launch --config_file config/accelerate/deepspeed_zero2.yaml \
    --num_processes 4 scripts/train_sft.py training=full

torchrun --nproc_per_node=4 scripts/train_sft.py training=full \
    training.deepspeed=config/deepspeed/zero2.json
```

Relevant training configs:

- `config/training_config.yaml` composes `model`, `prompt`, `dataset`, `dataset_val`, `lora`, and `training`.
- `config/training/` contains `smoke.yaml`, `quick.yaml`, and `full.yaml`.
- `config/lora/train.yaml` defines PEFT LoRA hyperparameters.
- `config/accelerate/` contains single-node DDP and DeepSpeed launch configs.
- `config/deepspeed/zero2.json` is the DeepSpeed config used by torchrun or Accelerate.

## Ablation Runners

The LoRA-rank ablation runner sweeps LoRA rank, placement, and dataset choices.

```shell
# Print every accelerate launch command without running it
python scripts/ablations/run_sft_ablations.py --dry-run

# Default sweep: r in {4, 8, 16, 32}, placement=both, dataset=oops
python scripts/ablations/run_sft_ablations.py

# Data-mix ablation
python scripts/ablations/run_sft_ablations.py \
    --rank 16 --dataset staged staged-oops staged-oops-wanfall
```

Other experiment and analysis runners live under `scripts/ablations/`, `scripts/analysis/`, `scripts/latex/`, and `scripts/plot/`.

## Configuration Reference

Hydra config groups are the primary control surface.

- `config/experiment/` contains end-to-end presets such as `debug`, `zeroshot`, `fewshot`, `fewshot_similarity`, `zeroshot_cot`, and `embed`.
- `config/model/` contains model-family configs such as `qwenvl`, `qwenvl_embed`, `internvl`, `molmo`, and `keyevl`.
- `config/prompt/` controls prompt variants, output format, chain-of-thought behavior, and few-shot settings.
- `config/sampling/` controls generation settings such as greedy, low-temperature, nucleus, and Qwen-specific presets.
- `config/vllm/` controls vLLM engine settings. Use `vllm=debug` for faster startup during local checks.
- `config/dataset/` contains video and feature dataset definitions for OMNIFALL, WANFALL, and combined groups. Combined video configs are mainly useful for training unless the inference script is extended to evaluate multiple individual datasets in one run.

Useful high-impact overrides:

- `num_samples`: limit inference samples.
- `batch_size`: number of videos loaded per inference batch.
- `num_workers`: dataloader worker count.
- `data.size`: spatial preprocessing size.
- `data.split`: dataset split, such as `cs` or `cv` when supported by the dataset config.
- `data.cache_dir`: disk tensor cache root.
- `data.cache_in_memory`: enable exemplar in-memory caching for few-shot workflows.
- `wandb.mode`: `online`, `offline`, or `disabled`.
- `wandb.project`: W&B project name and local prediction subdirectory.

## Outputs and Logs

Default output roots:

```text
outputs/predictions/<wandb-project>/<run_id>.jsonl
outputs/evaluation_results/<wandb-project>/
outputs/embeddings/
outputs/training/<run_name>/
logs/local_logs.log
logs/training.log
logs/build_tensor_cache.log
```

W&B run names are generated from the model, frame count, frame rate, and run ID unless `wandb.name` is set.
