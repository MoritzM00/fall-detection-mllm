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
