# DPO LoRA Implementation Plan

## First Milestone

Implement a small DPO training path alongside SFT:

```text
Existing SFT LoRA adapter
    -> DPO with one training dataset and one validation dataset
    -> export optimized policy adapter
    -> evaluate through existing vLLM inference
```

Supported scope:

- Qwen3-VL Instruct with an existing standard SFT LoRA adapter.
- Exactly one training dataset and one validation dataset, initially train/val splits of the same dataset.
- Zero-shot prompts and text answers using the existing conversation format.
- One deterministic random wrong label per video, fixed across epochs.
- Plain sigmoid DPO, with a frozen copy of the initial SFT adapter as reference.
- Single-GPU smoke and short runs, with Liger disabled.
- Selection by validation loss and a policy-only final export.

The milestone proves correct training and adapter interoperability. Improvement over SFT is an experimental outcome, not an implementation acceptance requirement. Reject unsupported configurations explicitly. Preserve existing SFT capabilities when changing shared code.

## Phase 1: Dependency and Dataset Compatibility Gate

Target `trl==1.12.0`. Resolve the training environment and pin the tested Liger version if its upgrade is required by the installed integration; DPO starts with `use_liger_kernel=False`. Keep dependency/SFT compatibility changes separate from DPO-specific changes.

Known blocker: TRL 1.12's SFTTrainer and DPOTrainer reject ordinary PyTorch datasets before calling dataset preparation. The current SFT wrapper is a PyTorch dataset. Overriding DPOTrainer's `_prepare_dataset()` alone is insufficient.

Validate the smallest shared lazy-dataset bridge first. The preferred approach is a Hugging Face map-style dataset of lightweight row indices, with a lazy transform resolving requested rows through the existing video/conversation dataset. Support scalar and batched access, retain required columns with `remove_unused_columns=False`, and decode only requested videos. Do not materialize decoded frames into Arrow or eagerly preprocess videos with `.map()`.

SFT retains its preparation-skip configuration. DPO uses a minimal preparation-bypass subclass after the dataset satisfies the constructor check. Validate this bridge with actual trainer construction and DataLoader/collator access, including workers. If it fails, resolve the bridge here before implementing the rest of DPO.

Acceptance checks:

1. Existing tests, including SFT collator and metric tests, pass.
2. SFTTrainer constructs with the repository's lazy data and custom collator under the resolved version.
3. A short real SFT smoke job completes when a GPU is available.
4. SFT prompt/loss behavior and adapter compatibility are preserved, without unrelated dependency downgrades.

## Phase 2: Random Preferences and Conversations

Create `src/falldet/training/preferences.py` with a small label-selection interface:

```python
from typing import Protocol


class NegativeSelector(Protocol):
    def select(self, query_index: int, positive_label: str) -> str:
        ...
```

Implement only `RandomNegativeSelector(labels, seed)`. Separate selection from decoding and formatting, without adding mining metadata, a strategy registry, or arbitrary response support now.

Requirements:

- Read observed training labels from metadata (`targets` or `video_segments`) without decoding videos.
- Canonicalize and deduplicate labels; require at least two training classes.
- Exclude the positive label and return exactly one wrong label.
- Use the training-label universe for validation negatives. In this initial same-dataset workflow, reject validation labels absent from that universe.
- Derive randomness from stable inputs such as `np.random.SeedSequence([seed, query_index])`, so selection is independent of access order and worker count and fixed across epochs.

Extend `src/falldet/training/dataset.py` with `DPOConversationDataset`. Its length equals the base dataset length: one video, one pair. Reuse `ConversationBuilder` and share the existing answer formatter where practical. Each item contains:

```python
{
    "prompt": conv_data.messages,
    "chosen": [{"role": "assistant", "content": [
        {"type": "text", "text": "The best answer is: fall"}
    ]}],
    "rejected": [{"role": "assistant", "content": [
        {"type": "text", "text": "The best answer is: lie_down"}
    ]}],
    "video_metadata": [...],
}
```

Validate that chosen/rejected labels and texts differ. Preserve the existing SFT/inference prompt vocabulary; the observed negative-label universe does not redefine the prompt.

Use the complete validation set or a fixed deterministic subset constructed before the conversation wrapper. Selector indices are local to that resulting dataset, including subsets. There is no embedding lookup or original-row mapping in this milestone. Record subset indices if a cap is used. Validation frames and preferences stay fixed across checkpoints.

Tests cover metadata-only label extraction, deterministic selection, wrong-label exclusion, formatting, and local indexing of a noncontiguous validation subset.

## Phase 3: Video Collator and Minimal Trainer

Add `VideoPreferenceCollator` to `src/falldet/training/collator.py`. For `B` pairs:

1. Render prompts with the processor chat template and `add_generation_prompt=True`.
2. Process the `B` videos once with Qwen3-VL metadata and the existing decoded-frame preprocessing settings.
3. Tokenize chosen/rejected answers without another assistant header. Include the supported model's assistant-ending/EOS token in each completion and its loss mask, consistent with SFT.
4. Duplicate processed prompts and visual tensors in all-chosen, then all-rejected order.
5. Join prompt and completion without internal padding gaps; right-pad final sequences.

Return:

```python
{
    "input_ids": ...,             # [2B, sequence_length]
    "attention_mask": ...,       # [2B, sequence_length]
    "completion_mask": ...,      # [2B, sequence_length]
    "pixel_values_videos": ...,
    "video_grid_thw": ...,
    "mm_token_type_ids": ...,    # full-sequence shape when returned
}
```

Preserve Qwen-required outputs and align flattened visual tensors and grid rows with text order. Extend multimodal token types with text values for completions. Pass video metadata to the processor; keep diagnostics/non-model metadata out of model inputs. Move every sequence-aligned mask together when flushing padding.

Prompt/padding positions have zero completion mask. Every row retains answer tokens and its terminator. Keep `max_length=None`; truncation is unsupported initially.

Create `src/falldet/training/dpo_trainer.py` with a minimal DPOTrainer subclass skipping eager preparation of the bridged dataset. Use the pinned TRL combined-batch loss. Do not copy the older external trainer's `concatenated_forward()` or Qwen forward patches.

Enforce `remove_unused_columns=False`, `precompute_ref_log_probs=False`, `max_length=None`, `use_liger_kernel=False`, and `padding_free=False`.

Essential checks:

- Actual trainer construction succeeds without eager video preprocessing.
- Fake-processor tests verify pair ordering, visual duplication, unequal lengths, termination, and mask alignment.
- Real-processor tokenization agrees with the supported chat template's completed assistant turn.
- Batched completion scores agree with separately scored examples using two distinct videos and unequal lengths, within numerical tolerance. Extra padding does not alter scores.
- The first completion token is scored at the correct causal position; policy/reference receive identical visual inputs.

## Phase 4: Train, Validate, and Export

Create `scripts/train_dpo.py`, reusing relevant Hydra, model/processor loading, dataset factory, logging, and W&B helpers. Avoid a broad training-lifecycle refactor.

Resolve one concrete training dataset and one concrete validation dataset. Reject multi-component configurations instead of silently choosing their first member. Pass validation as a single dataset, not a dictionary.

```python
base_model = AutoModelForImageTextToText.from_pretrained(...)
model = PeftModel.from_pretrained(
    base_model, config.dpo.sft_adapter_path, is_trainable=True,
)
trainer = VideoDPOTrainer(
    model=model, ref_model=None, peft_config=None, ...,
)
```

TRL 1.12 copies the supported standard LoRA `default` adapter into an internal `ref` adapter. Validate the adapter path and base-model compatibility. Reject unsupported adapter types, `target_parameters`, and non-LoRA trainable modules. Derive rank/target modules from the loaded adapter, including rank used for logging and the emitted vLLM command.

Verify on a fixed batch:

- Initial policy/reference log-probabilities match within tolerance.
- Plain sigmoid loss is approximately `log(2)` with zero initial reward margins.
- After an optimizer update, policy weights change while reference weights and reference scores remain unchanged.
- Only intended policy LoRA parameters are optimized.

Select checkpoints using the single validation set's `eval_loss`, with `greater_is_better=False`, `load_best_model_at_end=True`, and aligned save/evaluation intervals. The five-step smoke run evaluates and saves at its final step. Verify that best-model loading restores the intended policy, then export it.

Checkpoint resume is unsupported: reject non-null resume settings before training. Transformers 5.8's ordinary PEFT resume path can skip root `default` weights when a `ref/` subdirectory exists. Do not inherit SFT resume behavior. Intermediate checkpoints serve in-run best-model selection, not advertised DPO resumption.

Export only the policy:

```python
model.save_pretrained(adapter_dir, selected_adapters=["default"])
```

Verify reload through PEFT and vLLM. Record parent SFT identity, resolved configuration, and selected training step. Do not report interrupted runs as completed deliverables.

## Phase 5: Minimal Configuration and Logging

Add only:

```text
config/dpo_config.yaml
config/dpo/smoke.yaml
config/dpo/quick.yaml
config/preference/random.yaml
```

Root defaults reuse `model: qwenvl`, `prompt: default`, and one train/validation dataset, initially `omnifall/video/oops`. Reuse frame/FPS/resolution and loader settings, with deterministic sampling for the smoke run. The SFT adapter defines the adapter architecture; do not construct a new LoRA from training defaults.

Random preference config contains only `strategy: random` and `seed: 0`.

Initial smoke settings (other standard fields follow existing training schema conventions):

```yaml
sft_adapter_path: null  # required
beta: 0.1
loss_type: sigmoid
learning_rate: 1.0e-5
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
max_length: null
precompute_ref_log_probs: false
use_liger_kernel: false
padding_free: false
remove_unused_columns: false
eval_strategy: steps
save_strategy: steps
eval_steps: 5
save_steps: 5
max_steps: 5
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
resume_from_checkpoint: null  # non-null values rejected
```

Add required DPO schemas and `from_dictconfig_dpo()` in `src/falldet/schemas.py`, limited to supported options. Do not add unused strategy, multi-loss, multi-negative, or embedding configuration. Use bounded `max_steps` and straightforward warmup steps for smoke/quick runs.

Make only necessary DPO configuration/artifact changes to `src/falldet/utils/wandb.py`. Log parent SFT identity, actual rank, seeds, dataset configuration, pair counts, metadata-derived label distributions, train/validation loss, TRL reward accuracy/margins and log-probabilities, selected step, and final artifact location.

```bash
python scripts/train_dpo.py \
    dpo=smoke \
    dpo.sft_adapter_path=outputs/training/<sft-run>/adapter
```

## Rollout and Completion Criteria

1. Pass dependency/SFT compatibility and lazy-dataset construction checks.
2. Pass selector, conversation, and collator tests.
3. Pass numerical scoring and frozen-reference checks.
4. Complete a five-step real GPU DPO run with one training and one validation dataset.
5. Verify best-policy loading, policy-only export, and reload through PEFT and vLLM.
6. Complete a short random-DPO run and compare generated classifications with the parent SFT adapter using identical evaluation data, prompts, frames, and decoding.

Completion means real DPO updates, validation, export, and classification evaluation work without regressing SFT. Report classification metrics separately from preference accuracy. A measured improvement is not required to establish functional implementation.

## Deferred Follow-up Work

These require no placeholder implementations/configs and are not first-milestone acceptance criteria:

- **Multiple training/validation datasets:** composition, sampling weights, subset identity, and validation aggregation.
- **Checkpoint resume:** explicit policy/reference restoration and interrupted-versus-uninterrupted testing, including optimizer/scheduler state.
- **Similarity/confusion negatives:** extend label selection when implementing mining. Similarity requires nearest wrong-class training examples, chunked scoring, explicit query/corpus mappings, and versioned embedding manifests; current artifacts lack full sampling provenance. Combined embeddings belong with dataset composition.
- **Multiple or epoch-varying negatives:** pair expansion, reproducibility, and training-budget semantics.
- **Broader experiments:** continued-SFT controls, multiple seeds, common validation criteria, and matched training budgets before research claims.
- **Performance/model extensions:** Liger, distributed DPO, alternative losses, base initialization, other model/prompt variants, and truncation after correctness/memory measurements.
- **Arbitrary explicit preferences:** a richer response-level contract if needed; the initial selector returns only a wrong label.

## Implementation References

- [TRL 1.12 DPOTrainer](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/dpo_trainer.py): dataset check, combined batches, and reference initialization.
- [TRL 1.12 SFTTrainer](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/sft_trainer.py): dataset check and preparation bypass.
- [Transformers 5.8 Trainer](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/trainer.py): resume and best-model loading.
- Local `~/Qwen-VL-Series-Finetune` can inform preprocessing and tensor duplication, but its older trainer API is not a drop-in implementation. Preserve applicable attribution/license notices if substantive code is copied.
