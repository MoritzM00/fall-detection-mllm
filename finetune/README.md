# Finetune

Supervised fine-tuning of Qwen3-VL with LoRA on the omnifall video splits, via
TRL `SFTTrainer` driven by Hydra.

## Layout

```
finetune/
├── sft_lora.py        # Hydra entry point (TRL SFTTrainer + PEFT LoRA)
├── sft_dataset.py     # SFTConversationDataset — wraps a base video dataset as {messages, video_metadata}
├── sft_collator.py    # PromptMaskedSFTCollator — masks the prompt, supervises only the assistant turn
├── environment.yml    # conda env spec
└── requirements-finetune.txt
```

## Setup

One-time, from the repo root:

```bash
conda env create -n falldet-finetune -f finetune/environment.yml
conda activate falldet-finetune
export UV_TORCH_BACKEND=cu129
uv pip install transformers peft trl accelerate datasets torch torchvision hydra-core pydantic wandb
uv pip install -e .
```

## Run

The env must be active. Run as a module so the repo root is on `sys.path`:

```bash
python -m finetune.sft_lora                     # default: training=quick
python -m finetune.sft_lora training=full       # full run preset
python -m finetune.sft_lora training=smoke      # 50-step wiring check
```

Common Hydra overrides:

```bash
# different model
python -m finetune.sft_lora model.params=4B

# disable W&B sync
python -m finetune.sft_lora wandb.mode=offline

# cap epoch length for a quick check
python -m finetune.sft_lora training.max_steps=20
```

## Full-run shortcuts (by data split)

Pair `training=full` with the dataset group you want. Splits live in `config/dataset/omnifall/video/`.

```bash
# OOPS only (one file, ~thousands of clips, default dataset)
python -m finetune.sft_lora training=full dataset=omnifall/video/oops

# Staged datasets only (cross-subject)
python -m finetune.sft_lora training=full dataset=omnifall/video/staged-cs

# Staged datasets only (cross-view)
python -m finetune.sft_lora training=full dataset=omnifall/video/staged-cv

# Staged + OOPS combined
python -m finetune.sft_lora training=full dataset=omnifall/video/staged-oops

# Everything (staged + OOPS + wanfall — needs WANFALL_ROOT env var)
python -m finetune.sft_lora training=full dataset=omnifall/video/all
```

To match the val set to the train set, override `dataset_val` too:

```bash
python -m finetune.sft_lora training=full \
    dataset=omnifall/video/staged-cs \
    dataset@dataset_val=omnifall/video/staged-cs
```

## Configs

- `config/training_config.yaml` — root config; composes `model`, `prompt`, `dataset`, `lora`, `training`.
- `config/training/` — `smoke.yaml` (wiring), `quick.yaml` (bounded), `full.yaml` (full run with cosine schedule, eval+save every 200 steps, `load_best_model_at_end`).
- `config/lora/train.yaml` — PEFT LoRA hyperparameters (`r=8`, `alpha=16`, attention + MLP projections).

## Outputs

```
outputs/training/<run_name>/
├── checkpoint-*/      # rolling checkpoints (save_total_limit applies)
└── adapter/           # final adapter saved at end of training
```

`run_name` is the W&B run name. Load the adapter at inference time with the
existing `lora` config group in `inference_config.yaml`.
