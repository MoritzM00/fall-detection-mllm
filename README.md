# Video-Based Fall Detection with Multimodal Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)](https://docs.astral.sh/ruff/) [![vLLM](https://img.shields.io/badge/inference-vLLM-1e3a5f?logo=python&style=for-the-badge)](https://docs.vllm.ai) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&style=for-the-badge)](https://pre-commit.com)

This repository contains the code for my master's thesis on video-based fall detection with multimodal large language models (MLLMs).
It evaluates whether video-language models can identify fall events, fallen states, and general human activity classes such as `walking`, `standing`, and `sit_down`
on the OmniFall benchmark [paper](https://arxiv.org/abs/2505.19889).

The full thesis PDF is available as a [GitHub release](https://github.com/MoritzM00/fall-detection-mllm/releases/tag/v1.0.0).

The codebase supports four main workflows:

- Zero-shot inference with task instructions only.
- Few-shot inference with video exemplars selected by balanced/random sampling or embedding similarity.
- Chain-of-thought prompting for zero-shot reasoning traces.
- LoRA supervised fine-tuning of Qwen3-VL models with TRL `SFTTrainer`.

## Repository Layout

```text
fall-detection-mllm/
├── config/                          # Hydra configuration files
│   ├── inference_config.yaml        # Main inference config
│   ├── training_config.yaml         # Main training config
│   ├── dataset/                     # Dataset + split definitions (omnifall, wanfall, combined)
│   ├── model/                       # Model configs (qwenvl, internvl, molmo, keyevl)
│   ├── prompt/                      # Prompt presets (default, baseline, cot, fewshot, embed)
│   ├── sampling/                    # Decoding configs (greedy, nucleus, low_temp, qwen3)
│   ├── lora/                        # LoRA configs (none, train)
│   ├── training/                    # Training presets (smoke, quick, full)
│   ├── vllm/                        # vLLM engine settings
│   ├── accelerate/                  # DDP / DeepSpeed launch configs
│   ├── deepspeed/                   # DeepSpeed ZeRO configs
│   └── experiment/                  # End-to-end presets (debug, zeroshot, fewshot, embed)
│
├── scripts/                         # Experiment runners and utilities
│   ├── vllm_inference.py            # vLLM inference + embedding generation (Hydra entrypoint)
│   ├── train_sft.py                 # LoRA supervised fine-tuning (Hydra entrypoint)
│   ├── build_tensor_cache.py        # Precompute deterministic video tensors
│   ├── run_oops_experiments.py      # OOPS zero-shot experiment runner
│   ├── ablations/                   # Sweep runners (component, prompt, fewshot, SFT, size)
│   ├── analysis/                    # Analysis scripts
│   ├── latex/                       # LaTeX table generators for thesis output
│   └── plot/                        # Plotting utilities for saved runs and metrics
│
├── src/falldet/                     # Main Python package
│   ├── data/                        # Dataset loading, video preprocessing, tensor caching
│   ├── inference/                   # vLLM engine wrapper, prompt building, few-shot sampling
│   │   └── prompts/                 # Prompt builder, components, parsers
│   ├── evaluation/                  # Evaluation orchestration + subgroup analysis
│   ├── metrics/                     # Classification and subgroup-stratified metric computation
│   ├── training/                    # SFT data pipeline: dataset, collator, eval sampling
│   ├── plot/                        # Confusion matrices, metric charts, video grids
│   ├── utils/                       # Shared helpers: formatting, LaTeX, logging, W&B
│   ├── schemas.py                   # Pydantic / dataclass schemas
│   ├── config.py                    # Hydra config dataclasses
│   └── embeddings.py                # Embedding utilities
│
├── tests/                           # pytest test suite
├── notebooks/                       # Exploratory analysis and development notebooks
├── README.md
├── Makefile
├── pyproject.toml                   # Package configuration
├── environment.yml                  # Conda environment
├── requirements.txt                 # Production dependencies
└── requirements-dev.txt             # Development dependencies
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

## Prompt Construction

Prompts are assembled from modular components by `PromptBuilder`
(`src/falldet/inference/prompts/builder.py`), driven entirely by the `prompt`
config group (`config/prompt/`). Every field maps to a `PromptConfig` attribute
(`src/falldet/schemas.py`). Changing a variant swaps one text block in or out;
the builder never edits the others.

### Zero-Shot Assembly Order

`build_prompt()` concatenates the following sections with blank lines between
them. Optional sections are skipped entirely when their selector is `null`/`false`:

1. **Role** — included only if `role_variant` is set. Expert-persona preamble.
2. **Task** — always included. The instruction telling the model to classify the
   primary action and assign exactly one label.
3. **Clip-overlap note** — included only if `clip_overlap_note=true`. Tells the
   model to classify the action in the _first_ part of a multi-action clip.
4. **Labels** — always included. The allowed-label list, formatted per
   `labels_variant`.
5. **Definitions & decision rules** — included only if `definitions_variant` is set.
6. **Chain-of-thought instruction** — included only if `cot=true`.
7. **Output format** — included unless `output_format=null` (embed mode). Tells
   the model whether to answer as `text` (`The best answer is: <label>`) or `json`.

The **system message** is resolved separately by `get_system_message()`:
an explicit `prompt.system_instruction` wins; otherwise InternVL with `cot=true`
auto-injects the R1 thinking preamble; otherwise there is no system message.

### Variant Selectors

| Field                 | Values                                                 | Effect on the prompt                                                                                                                                                     |
| --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `role_variant`        | `null`, `standard`, `specialized`, `video_specialized` | Omit the role block, or use the generic HAR persona / a fall-detection-specialized / a video-analyst-specialized persona.                                                |
| `task_variant`        | `standard`, `extended`                                 | Short two-line task instruction vs. a longer one that explicitly mentions posture, motion dynamics, and environmental cues.                                              |
| `clip_overlap_note`   | `true`, `false`                                        | Add/remove the "focus on the first action in a multi-action clip" note.                                                                                                  |
| `labels_variant`      | `bulleted`, `comma`, `grouped`, `numbered`             | Formatting of the allowed-label list: bulleted/numbered lists, a comma-separated line, or Core/Extended groups.                                                          |
| `definitions_variant` | `null`, `standard`, `extended`                         | Omit decision rules, include the core fall/lying/sitting disambiguation rules, or the extended rule set that also covers locomotion and low postures.                    |
| `cot`                 | `true`, `false`                                        | Add the "reason step-by-step" instruction and wrap the parser in a CoT parser that strips `cot_start_tag`/`cot_end_tag` (`<think>`/`</think>`) before reading the label. |
| `output_format`       | `text`, `json`, `null`                                 | Choose the answer format / parser, or drop the format block entirely for embedding runs.                                                                                 |

The variant text lives in `src/falldet/inference/prompts/components.py`; the
registries (`ROLE_VARIANTS`, `TASK_VARIANTS`, `DEFINITIONS_VARIANTS`,
`LABEL_FORMAT_VARIANTS`, `OUTPUT_FORMAT_VARIANTS`) map each selector value to its
block.

### Few-Shot Assembly

When `num_shots > 0`, `Conversation` (`src/falldet/inference/conversation.py`)
builds an in-context-learning prompt instead. `build_fewshot_preamble()` reuses
the same role/task/labels/definitions blocks, then appends a demonstrations
explanation (and, for InternVL, a "do not think" instruction). Exemplar turns and
the query turn are then laid out according to these fields:

| Field               | Values                                                     | Effect                                                                                                                                                                                                              |
| ------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_shots`         | int (`0` = zero-shot)                                      | Number of in-context exemplars. CoT and few-shot are mutually exclusive (validated in `PromptConfig`).                                                                                                              |
| `shot_selection`    | `random`, `balanced`, `similarity`, `per_class_similarity` | How exemplars are sampled: uniform random, class-balanced, nearest-neighbor by embedding, or nearest-neighbor per class. The two similarity modes require precomputed embeddings (`experiment=fewshot_similarity`). |
| `exemplar_seed`     | int                                                        | Seed for reproducible exemplar sampling.                                                                                                                                                                            |
| `exemplar_ordering` | `ascending`, `descending`, `random`                        | Orders exemplars by similarity score (ignored for `random` selection).                                                                                                                                              |
| `fewshot_preamble`  | `system`, `user`                                           | Whether the role/task/labels preamble goes in the system message or as the first user turn.                                                                                                                         |
| `fewshot_response`  | `inline`, `assistant`                                      | `inline` keeps each exemplar's answer inside the same user turn (`[REQUEST n] … [RESPONSE n] …`); `assistant` splits each exemplar into a user turn plus a real `assistant` answer turn (`[DEMONSTRATION n]`).      |

### Prompt Presets

The `config/prompt/*.yaml` files are ready-made selector bundles wired to the
experiment presets:

- `default.yaml` — zero-shot baseline used by `experiment=zeroshot`: standard role,
  standard task, clip-overlap note on, bulleted labels, no definitions, text output.
- `baseline.yaml` — the minimal reference point for component ablations: no role,
  standard task, no clip-overlap note, no definitions, text output.
- `cot.yaml` — zero-shot chain-of-thought (`cot=true`), used by `experiment=zeroshot_cot`.
- `fewshot.yaml` — five balanced exemplars, ascending order, user preamble,
  assistant-style responses; used by `experiment=fewshot` and `fewshot_similarity`.
- `embed.yaml` — embedding pass: `output_format=null`, no role, with an explicit
  `system_instruction`; used by `experiment=embed`.

Override any field on the command line, e.g.:

```shell
python scripts/vllm_inference.py experiment=zeroshot \
    prompt.role_variant=specialized \
    prompt.definitions_variant=extended \
    prompt.output_format=json
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

### Validation Metrics and Best-Model Selection

Validation is enabled whenever `training.eval_strategy != "no"` (it is `steps`
in the `quick`, `full`, and `smoke` presets, evaluating every
`training.eval_steps` and also once before training when
`training.eval_on_start=true`). The validation set is loaded with `mode="val"`
and a fixed `seed=0`, so it is deterministic regardless of `data.seed`. Each
validation dataset is capped independently to `training.max_eval_samples_per_ds`
samples via `stratified_sample_indices` (`src/falldet/training/eval_sampling.py`),
which stratifies by label and guarantees at least one sample per class so rare
classes such as `fall` and `lie_down` are never dropped. When `dataset_val`
resolves to more than one dataset, each is evaluated separately and the Trainer
logs per-dataset metrics under the `eval_{dataset}_{metric}` prefix.

**How the metrics are computed.** Validation metrics are *teacher-forced*, not
free-running generation. For each eval batch the model produces logits over the
ground-truth completion; `preprocess_logits_for_metrics`
(`src/falldet/training/metrics.py`) reduces the `(N, seq_len, vocab)` logits to
`(N, seq_len)` next-token argmax IDs (shifted by one so a prediction aligns with
its target) to keep CPU memory bounded. The `compute_metrics` callback then
decodes only the unmasked completion tokens (those not set to `-100`) for both
predictions and labels, parses each decoded string into a class label with
`KeywordOutputParser`, and delegates to the project-wide
`falldet.metrics.base.compute_metrics`. That produces the same metric suite used
at inference time: `accuracy`, `balanced_accuracy`, `macro_f1`, the binary
fall / fallen / fall∪fallen sensitivity-specificity-F1 triples, per-class
metrics, and class-distribution / sample-count diagnostics. All metrics are
logged to W&B (`report_to=wandb`).

Note that these validation metrics are **not directly comparable** to the real
(autoregressive) inference metrics. Because each position's argmax is
conditioned on the ground-truth prefix, the decoded tokens can form a label that
generation would never produce, so `KeywordOutputParser` falls back to the
`other` class and logs `No valid label found in text. Defaulting to 'other'.`.
These warnings are expected and far more frequent during teacher-forced
validation than at inference time; treat the validation numbers as a proxy for
tracking relative progress across steps, not as a substitute for the inference
evaluation.

**Best-model tracking — no automatic early stopping.** No
`EarlyStoppingCallback` is registered, so training always runs to `max_steps`
(or `num_train_epochs` when `max_steps <= 0`); validation does not halt a run.
Best-model selection is driven by `metric_for_best_model` (e.g. `eval_macro_f1`,
with `greater_is_better=true`). When `dataset_val` has multiple datasets, the
metric key is automatically rebased onto the first dataset
(`eval_{first_dataset}_{metric}`). The best value seen is recorded in
`trainer.state.best_metric` and attached to the logged adapter artifact. The
preset default is `load_best_model_at_end: false`, so the *final-step* adapter is
what gets saved unless you opt in by setting `training.load_best_model_at_end=true`
(which requires `save_strategy`/`eval_strategy` to agree on cadence). Checkpoints
are written per `save_strategy`/`save_steps` and pruned to `save_total_limit`.

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

### Multi-GPU Training

The main fine-tuning runs were launched on 2 GPUs with the following command through the lora `run_sft_ablations.py` script (see [ablation runners](#ablation-runners))

Single-node DDP with bf16:

```shell
accelerate launch --config_file config/accelerate/ddp_bf16.yaml \
    --num_processes 2 scripts/train_sft.py training=full
```

Supported but not well tested: DeepSpeed ZeRO-2 with optimizer and gradient sharding:

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

`scripts/ablations/` holds standalone sweep runners. Each one builds Hydra CLI
commands for `vllm_inference.py` or `train_sft.py`, tags every run with
descriptive W&B tags, and (except the size shell script) supports `--dry-run` to
print commands without executing and `--start-from N` to resume a partially
completed sweep. The Python runners default to `--model qwenvl --params 8B`.

### `run_component_ablations.py` — one-at-a-time prompt components

Holds every prompt component at the `baseline.yaml` reference point
(no role, standard task, bulleted labels, no definitions, text output) and sweeps
**one component at a time**, so each run differs from the baseline in a single
field. Sweeps `role_variant` (`null`, `standard`, `specialized`,
`video_specialized`), `task_variant` (`standard`, `extended`), `labels_variant`
(`bulleted`, `numbered`, `grouped`, `comma`), and `definitions_variant`
(`null`, `standard`, `extended`). Duplicate baselines are deduplicated, giving
10 unique runs. Base experiment `zeroshot`, tags `[ablation, component, …]`.

```shell
python scripts/ablations/run_component_ablations.py --dry-run
```

### `run_prompt_ablations.py` — full prompt grid

Runs the **full cross-product** of `role_variant` (`null`, `standard`),
`definitions_variant` (`null`, `standard`), and `output_format` (`json`, `text`)
= 8 combinations. Base experiment `zeroshot`, `wandb.project=prompt-ablations`,
tags `[ablation, prompt, …]`.

```shell
python scripts/ablations/run_prompt_ablations.py --dry-run
```

### `run_fewshot_ablations.py` — exemplar count / selection / ordering

Sweeps `num_shots` (default `1 2 3 5`), `shot_selection` (default `balanced`;
choices `random`, `balanced`, `similarity`, `per_class_similarity`), and
`exemplar_ordering` (default `ascending`; choices `ascending`, `descending`,
`random`). `random` selection collapses to a single ordering. Similarity-based
selections automatically switch to `experiment=fewshot_similarity` (which expects
precomputed embeddings); everything else uses `experiment=fewshot`. Accepts
multiple `--model` values (models vary slowest). Tags `[ablation, fewshot, shots-N, selection-S, ordering-O]`.

```shell
# Full default sweep across shot counts
python scripts/ablations/run_fewshot_ablations.py --dry-run

# Compare selection strategies at 5 shots
python scripts/ablations/run_fewshot_ablations.py \
    --shots 5 --selection balanced similarity per_class_similarity
```

### `run_fewshot_format_ablations.py` — exemplar layout

Sweeps the in-context layout: `fewshot_preamble` (`system`, `user`) ×
`fewshot_response` (`inline`, `assistant`) = 4 combinations per model. Base
experiment `fewshot`. Tags `[ablation, fewshot_format, preamble-P, response-R]`.

```shell
python scripts/ablations/run_fewshot_format_ablations.py --dry-run
```

### `run_sft_ablations.py` — LoRA fine-tuning sweep

Wraps `accelerate launch … scripts/train_sft.py training=full` and sweeps LoRA
`--rank` (default `4 8 16 32`, with `alpha = 2r` unless `--alpha r`), `--placement`
(`attn`, `mlp`, `both`; default `both`), and `--dataset` (default `oops`;
also `staged`, `staged-oops`, `staged-oops-wanfall`, `wanfall`, `all`). The
validation group is auto-selected per dataset (staged training datasets evaluate
on `cmdfall` to avoid leakage). `--num-processes` sets the accelerate process
count and `--per-device-batch-size` overrides the per-device batch.

```shell
# Default sweep: r in {4, 8, 16, 32}, placement=both, dataset=oops
python scripts/ablations/run_sft_ablations.py --dry-run

# Data-mix ablation at fixed rank
python scripts/ablations/run_sft_ablations.py \
    --rank 16 --dataset staged staged-oops staged-oops-wanfall
```

### `run_size_ablation.sh` — input resolution

A small bash loop over `data.size` (`224`, `448`, `768`) for the model(s) listed
in the script, using `experiment=zeroshot prompt=baseline`. Edit the `SIZES`/`MODELS`
arrays in the script to change the sweep. Tags `[ablation, input_size]`.

```shell
bash scripts/ablations/run_size_ablation.sh
```

Other experiment and analysis runners live under `scripts/analysis/`, `scripts/latex/`, and `scripts/plot/`.

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
