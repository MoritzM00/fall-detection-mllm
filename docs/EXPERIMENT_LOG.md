# ML Experiment Logbook

Local, uncommitted record of completed experiments.

## 2026-09-21 — Random-negative DPO from pretrained Qwen3-VL

Logged: 2026-09-22

### Goal

Train Qwen3-VL-8B-Instruct with DPO from a fresh LoRA using random incorrect class labels, then compare intermediate checkpoints on OOPS-CS classification.

### Setup

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Initialization: pretrained model + fresh LoRA, rank 8; no SFT adapter
- Data: OOPS cross-subject split; 809 train, 409 validation, 2,804 test samples
- Input: 16 frames at 7.5 FPS, size 448
- DPO: sigmoid loss, beta 0.1; random negatives from all 16 labels; seed 0
- Training: 10 epochs / 1,020 steps; batch 1, gradient accumulation 8, LR 1e-5, BF16
- Kernels: FlashAttention 2 and Liger 0.8.2
- Checkpoints: saved and validated every epoch
- Runtime: 3h 38m 50s on GPU 1
- Git revision: `7f9d846b68d8fb1f6b1ed88368152ded50ad433e`

### Changes needed to run

- Wired `dpo.use_liger_kernel` into `DPOConfig` and upgraded Liger from 0.8.0 to 0.8.2 for TRL 1.12 compatibility.
- Changed random negative sampling to use the full label set. The training-fold-only label universe rejected validation samples whose positive class was absent from training.
- Kept batch size 1 because GPU utilization was already 91–100%.

### Results

The lowest DPO validation loss was at epoch 5, so the exported `adapter/` contains `checkpoint-510`.

| Epoch | DPO validation loss |
|---:|---:|
| 1 | 0.28395 |
| 3 | 0.20314 |
| 5 | **0.18704** |
| 6 | 0.18970 |
| 10 | 0.20119 |

Full OOPS-CS test results:

| Epoch | Accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|---:|
| 1 | 50.71% | 33.60% | 29.16% |
| 3 | 56.60% | **36.17%** | 34.25% |
| 6 | 57.63% | 35.77% | 34.27% |
| 10 | **58.10%** | 35.92% | **34.62%** |

### Finding

Epoch 10 performed best for accuracy and macro F1, while epoch 3 performed best for balanced accuracy. The checkpoint with the lowest DPO validation loss (epoch 5) was therefore not automatically the best checkpoint for downstream classification.

### Artifacts

- Training: `outputs/dpo/dpo-oops-random-scratch-liger-10epochs-b1-20260921_wey6g0yr/`
- W&B: `moritzm00/falldet-mllm-finetune/wey6g0yr`
- Metrics: `outputs/evaluation_results/fall-detection-dpo-eval/`
- Predictions: `outputs/predictions/fall-detection-dpo-eval/`

## 2026-09-26 — Confusion-guided DPO negatives

Logged: 2026-09-26

### Goal

Replace random DPO negatives with labels the zero-shot model actually confuses, and compare with the random-negative run above on OOPS-CS.

### Negative mining

- Source: Qwen3-VL-8B-Instruct zero-shot predictions on the **OOPS-CS train split** (W&B `fall-detection-zeroshot-v4/themoiiq`, `prompt.clip_overlap_note=false`; `outputs/predictions/fall-detection-zeroshot-v4/themoiiq.jsonl`). 809 rows, accuracy 45.9%, balanced accuracy 43.2%. The existing zero-shot run `p1r3exbe` was not used because it is on the test split.
- Confusion matrix `C[true, predicted]` over all 16 labels; per true class, `P = 0.9 · normalized off-diagonal errors + 0.1 · uniform over the 15 wrong labels` (uniform if the class has no errors).
- Per train clip, matched by (dataset, video path, start, end, label): if the model was wrong, the rejected label is its own prediction; otherwise it is sampled from `P[true]`, seeded by (seed, clip index) and fixed for all epochs.
- Validation has no per-clip predictions, so its negatives are sampled from the train-derived `P`.
- Result on train: 438 clips (54%) use the model's own error, 371 use a sampled label. Most frequent pairs: fall→jump 55, other→fall 51, jump→fall 43, other→jump 41, fallen→lying 39.

### Setup

- Same as the random-negative run except for the negatives: pretrained Qwen3-VL-8B-Instruct + fresh LoRA rank 8, 16 frames at 7.5 FPS, size 448, prompt with clip overlap note, sigmoid DPO beta 0.1, LR 1e-5 cosine with 10% warmup, 10 epochs / 1,020 steps, FlashAttention 2 + Liger 0.8.2, per-epoch eval and save.
- Hardware: 2× H100 (HoreKa 2 `gpu-h100`), per-device batch 1, gradient accumulation 4 → effective batch 8. ~4.3 s/step, ~1.5 h total (vs 3h 39m on one GPU).
- Command: `sbatch --gres=gpu:2 --cpus-per-task=32 --mem=256G slurm/dpo.sbatch dpo=oops dataset=omnifall/video/oops dataset@dataset_val=omnifall/video/oops preference=confusion 'preference.train_predictions_paths=[outputs/predictions/fall-detection-zeroshot-v4/themoiiq.jsonl]'`
- Git revision: code of `bea0114` (the job was submitted just before the commit).

### Changes needed to run

- Added `preference=confusion` (`src/falldet/training/confusion.py`, `preference_setup.py`), `slurm/dpo.sbatch` and the `dpo=oops` preset.
- The hk2 venv had TRL 1.3.0 (main's pin) and Liger 0.8.0; installed TRL 1.12.0 and Liger 0.8.2 and pinned Liger 0.8.2 in `requirements.txt`.
- Fixed multi-GPU output directories: non-main ranks got a disabled W&B run named `dummy-<id>` and wrote their RNG state to `outputs/dpo/dummy-*` (fixed in `c124e3c`; this run was still affected).

### Results

DPO validation loss: 0.693 (start), 0.516 (1), 0.354 (2), 0.377 (3), 0.342 (4), 0.311 (5), 0.321 (6), 0.304 (7), 0.305 (8), 0.301 (9), **0.300 (10)**. Training finished in 1h 36m (job 30888); lowest validation loss is the final epoch.

Full OOPS-CS test (epoch 3 = the random run's best balanced-accuracy epoch; later epochs = each run's lowest DPO validation loss):

| Model | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 |
|---|---:|---:|---:|---:|---:|
| Zero-shot (`p1r3exbe`) | 45.39% | **36.36%** | 26.31% | **67.78%** | 29.19% |
| Random negatives, epoch 3 | **56.60%** | 36.17% | **34.25%** | — | — |
| Confusion negatives, epoch 3 | 47.43% | 28.01% | 28.02% | 58.75% | 60.52% |
| Confusion negatives, epoch 10 | 55.74% | 27.76% | 29.67% | **76.25%** | 62.73% |
| Confusion + random sampling, epoch 3 | 51.39% | 30.38% | 30.69% | 66.73% | 62.91% |
| Confusion + random sampling, epoch 7 | 56.13% | 28.36% | 30.65% | 75.70% | **64.23%** |

Per-class recall, zero-shot → confusion epoch 3: fall 0.85 → 0.43, jump 0.31 → 0.16, lying 0.25 → 0.06; fallen 0.20 → 0.70, standing 0.37 → 0.73, other 0.37 → 0.54. "fall" predictions dropped from 934 to 290 and "other" rose from 399 to 906.

By epoch 10 the frequent classes recover (recall fall 0.66, other 0.65, fallen 0.64, standing 0.62, walk 0.55; 463 "fall" predictions) and fall F1 exceeds zero-shot, but the rare classes collapse (jump 0.18, kneeling 0.16, lying 0.00), so balanced accuracy stays at 27.8% and macro F1 below the random run (29.7% vs 34.6% at epoch 10).

### Finding

Confusion negatives make the model avoid the labels it was trained to reject most often, instead of adding per-video discrimination. Across the 809 training pairs, net (chosen − rejected) counts are other +130, fallen +66, stand_up +45 versus jump −89, lying −69, crawl −37; test predictions moved in the same direction. "fall" is roughly balanced overall (−9) but is rejected exactly on fall-like videos (true other/jump/walk), so the model learned to answer "other" there and fall recall halved. With random negatives each label is rejected about equally often, so this label-prior push largely cancels.

As in the random run, DPO validation loss kept improving while test classification got worse, so it is not a usable model-selection signal here.

### Follow-up: model errors + random sampling

- `preference.sample_from=random`: keep the model's own error for the 438 misclassified clips, use a uniform random wrong label for the 371 correct clips and all validation clips. The job was launched as `preference.fallback=random`; the option was renamed to `sample_from` (`confusion_matrix` | `random`) afterwards with identical behavior.
- Net chosen − rejected becomes fall +41 (was −9) and jump −41 (was −89); pure random is about fall +130, jump −7.
- Job 30940, W&B `falldet-mllm-finetune/8lctvfqg`, same settings otherwise; 1h 38m.
- DPO validation loss: 0.440 (1), 0.303 (2), 0.297 (3), 0.292 (4), 0.253 (5), 0.256 (6), **0.249 (7)**, 0.254 (8), 0.255 (9), 0.255 (10). Validation negatives are uniform random here, so these losses are not comparable with the confusion run.
- Epoch 3 test (table above): better than confusion sampling at epoch 3 (fall recall 0.53 vs 0.43, walk 0.58 vs 0.47), but still below zero-shot and random negatives on balanced accuracy. Most of the fall drop therefore comes from the model's own errors (other→fall 33, other→jump 29, jump→fall 26), not from the sampled negatives.
- Epoch 7 test (lowest validation loss): nearly identical to confusion sampling at epoch 10 (accuracy 56.1% vs 55.7%, balanced accuracy 28.4% vs 27.8%, fall F1 75.7% vs 76.3%). Recall: fall 0.66, other 0.66, fallen 0.69, standing 0.57, walk 0.53, stand_up 0.44; rare classes still collapse (jump 0.23, sitting 0.24, kneeling 0.11, lying 0.00). With longer training, how the correctly-classified clips are negated barely matters; the model's own errors dominate the outcome.

### Conclusion

After enough training, both confusion variants beat zero-shot on accuracy (+10 points) and fall/fallen F1 (fall ~76% vs 68%, fallen ~63% vs 29%), but lose about 8 points of balanced accuracy to rare-class collapse and stay below random negatives on balanced accuracy and macro F1. For 16-class OOPS-CS, random negatives remain the better DPO variant; confusion negatives are only preferable if fall/fallen detection is the target. Next: select checkpoints on teacher-forced validation classification metrics (`dpo.classification_metrics=true`, added in `291fbcb`; requires Liger off) rather than DPO loss.

### Artifacts

- Mining source predictions: `outputs/predictions/fall-detection-zeroshot-v4/themoiiq.jsonl`
- Training: `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_6i6nebez/` (rank-1 RNG state in `outputs/dpo/dummy-954wvj6t/`)
- Negative manifest: `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_6i6nebez/preference_mining.jsonl`
- Random-sampling run: `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_8lctvfqg/` (manifest in the same directory)
- W&B: `moritzm00/falldet-mllm-finetune/6i6nebez` (training), `moritzm00/fall-detection-dpo-eval/njvp8y93` (epoch 3 test), `fry4cw8i` (epoch 10 test); random-sampling run `8lctvfqg` (training), `zpdfkhpj` (epoch 3 test), `j89x04em` (epoch 7 test)
- Predictions: `outputs/predictions/fall-detection-dpo-eval/{njvp8y93,fry4cw8i,zpdfkhpj,j89x04em}.jsonl`
- Smoke test: `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_jtyocum5/` (5 steps, 1 GPU)
