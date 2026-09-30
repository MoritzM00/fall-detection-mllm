# Fine-Tuning Experiments: LoRA SFT and DPO

Chronological record of completed fine-tuning experiments on Qwen3-VL (setup, results, findings, artifacts). Paths under `outputs/` and Slurm job IDs refer to the HoreKa 2 working copy.

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

## 2026-09-27 — LoRA SFT on the full data mix

Logged: 2026-09-27

### Goal

Fine-tune Qwen3-VL-8B-Instruct with LoRA SFT on all OmniFall + WanFall training data (`dataset=omnifall/video/all`) and track per-dataset validation metrics.

### Setup

- Model: `Qwen/Qwen3-VL-8B-Instruct` + fresh LoRA (`lora=train`); 16 frames at 7.5 FPS, size 448
- Data: 42,189 train samples (10 datasets, CS / random split); validation 6,637 samples across 9 datasets, capped at 1,000 per dataset (stratified)
- Training: `training=full`, 4,000 steps ≈ 3.03 epochs (1,318 steps/epoch); per-device batch 8, no gradient accumulation, 4 GPUs → effective batch 32; LR 1e-4 cosine, 400 warmup steps; FlashAttention 2 + Liger; DDP (no DeepSpeed)
- Eval and save every 1,000 steps plus eval at start; all checkpoints kept; no best-model selection
- Hardware: 4× H100 (HoreKa 2 `gpu-h100`, hkn0905), ~3.0 s/step, 3h 39m total (job 31987)
- Command: `sbatch --gres=gpu:4 --cpus-per-task=64 --mem=512G --time=12:00:00 slurm/train.sbatch training=full training.max_steps=4000 training.eval_steps=1000 training.save_steps=1000 training.save_total_limit=null training.load_best_model_at_end=false training.metric_for_best_model=null dataset=omnifall/video/all dataset@dataset_val=omnifall/video/all 'wandb.tags=[sft,all,4000steps]'`
- Git revision: `b041a41` plus the `ddp_find_unused_parameters` change (committed as `909656a`); the working tree also held another session's uncommitted edits to `src/falldet/training/collator.py` and `dataset.py`.

### Decisions

- **Batch size**: the OOPS SFT run (`74ilzwnw`, job 31978, per-device batch 8 on 2 GPUs) used 87–88 of 96 GB per GPU at 99–100% utilization, so per-device batch stays at 8; the larger effective batch (16 → 32) comes from 4 GPUs. Step time is unchanged (~3 s), so more GPUs add samples per step, not speed.
- **Steps**: first submitted with 10,000 steps (job 31984), cancelled after ~15 steps. That would have been ~7.6 epochs; the single-label target is memorized quickly (OOPS run: 97% token accuracy, loss 0.08 at ~3 epochs), and with cosine decay over 10k steps an intermediate checkpoint is not equivalent to a shorter run.
- **Multi-node**: not used. `train.sbatch` / `config/accelerate/ddp_bf16.yaml` are single-node only, and at a fixed step count extra nodes only raise the batch size, not throughput.
- **DeepSpeed**: not used (untested). ZeRO-2 saves little with LoRA; memory is dominated by activations.
- **Best-model selection**: disabled. With several validation datasets `metric_for_best_model` is rewritten to the first dataset's metric (cmdfall); a cross-dataset average is needed first.
- **DDP**: set `ddp_find_unused_parameters=false` for SFT presets (`909656a`); HF defaults to `True` for PEFT models, which added an extra autograd traversal per step.

### Results

Validation balanced accuracy (%) per dataset; step 0 is the zero-shot model:

| Dataset | Step 0 | 1k | 2k | 3k | 4k (final) | Macro F1 at 4k |
|---|---:|---:|---:|---:|---:|---:|
| cmdfall | 47.3 | 75.6 | 77.4 | 79.7 | 79.0 | 79.4 |
| up_fall | 48.2 | 84.4 | 85.2 | 86.6 | 87.0 | 86.7 |
| le2i | 43.2 | 71.1 | 73.3 | 71.3 | 70.8 | 69.3 |
| gmdcsa24 | 63.1 | 75.8 | 76.6 | 67.3 | 71.1 | 71.4 |
| edf | 37.3 | 32.1 | 45.2 | 44.4 | 45.6 | 42.7 |
| occu | 29.4 | 74.7 | 84.2 | 85.4 | 89.4 | 85.8 |
| caucafall | 59.4 | 71.9 | 78.1 | 90.6 | 90.6 | 89.0 |
| OOPS | 30.9 | 37.6 | 40.2 | 43.0 | 43.0 | 45.5 |
| wanfall | 58.0 | 54.2 | 59.0 | 73.0 | 66.1 | 63.6 |
| **Mean** | 46.3 | 64.2 | 68.8 | 71.3 | 71.4 | |

mcfd has no CS validation/test clips and is skipped. Final training loss 0.042.

OOPS-CS test (2,804 clips, vLLM greedy, final adapter), compared with the OOPS-only SFT run:

| Model | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 | Fall ∪ fallen F1 |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot (`p1r3exbe`) | 45.4% | 36.4% | 26.3% | 67.8% | 29.2% | — |
| SFT r8, OOPS only, 300 steps (`olspo2r0`) | **64.6%** | 38.0% | 39.3% | **84.9%** | **68.3%** | **83.3%** |
| SFT r8, all datasets, 4,000 steps (`4xo9l0um`) | 62.5% | **38.9%** | **42.1%** | 82.1% | 65.1% | 80.5% |

Rare-class F1, OOPS-only → all-data SFT: sit_down 0.18 → 0.38, lie_down 0.00 → 0.17, lying 0.00 → 0.16; other 0.62 → 0.57.

### Finding

Training on the full mix lifts every lab dataset by 25–60 points of validation balanced accuracy over zero-shot, with mean balanced accuracy still rising at 3k steps and flat from 3k to 4k (71.3 → 71.4), so ~3 epochs is about right; per-dataset swings on the small sets (gmdcsa24, caucafall, wanfall) are noisy. On OOPS test the extra data trades ~2 points of accuracy and fall/fallen F1 for +0.9 balanced accuracy and +2.8 macro F1: the rare OOPS classes benefit from lab-dataset examples, while the frequent fall/other classes are slightly worse than with OOPS-only training. Validation overstated the OOPS gain (43.0% vs 38.9% on test).

### Artifacts

- Job 31987, log `logs/slurm/falldet-sft-31987.out`
- OOPS test eval: job 32379, W&B `fall-detection-sft-eval/4xo9l0um`, `outputs/evaluation_results/fall-detection-sft-eval/test_results_sft-all-r8-4000steps-7emiqqt3-oops-test_4xo9l0um.json` (a full-mix test eval, job 32362 / `di2lx0ym`, was cancelled)
- W&B: `moritzm00/falldet-mllm-finetune/7emiqqt3`
- Training: `outputs/training/Qwen3-VL-8B-Instruct-F16at7.5_7emiqqt3/` (adapter in `adapter/`)
- Reference OOPS SFT run: W&B `74ilzwnw`, `outputs/training/Qwen3-VL-8B-Instruct-F16at7.5_74ilzwnw/` (300 steps, final validation accuracy 66.0%, balanced accuracy 37.7%, macro F1 39.3%)

## 2026-09-27 — Random-negative DPO with classification-metric selection

Logged: 2026-09-27

### Goal

Check that teacher-forced validation classification metrics (`dpo.classification_metrics=true`, `291fbcb`) are a usable checkpoint-selection signal for DPO, unlike DPO validation loss.

### Setup

- Same as the 2026-09-21 random-negative run except: 4 epochs (408 steps) instead of 10, so the cosine schedule decays over 4 epochs; Liger off (classification metrics need full logits); best checkpoint by `eval_balanced_accuracy`.
- Pretrained Qwen3-VL-8B-Instruct + fresh LoRA rank 8, 16 frames at 7.5 FPS, size 448, prompt with clip overlap note, sigmoid DPO beta 0.1, LR 1e-5 cosine with 10% warmup, random negatives over all 16 labels, seed 0.
- Validation: all 409 OOPS-CS validation clips, before training and after every epoch (~2 min per pass).
- Hardware: 2× H100 (`gpu-h100`), per-device batch 1, gradient accumulation 4 → effective batch 8; 6.6 s/step without Liger (vs ~4.3 s/step with it), 46 min total.
- Command: `sbatch --gres=gpu:2 --cpus-per-task=32 --mem=256G --time=04:00:00 slurm/dpo.sbatch dpo=oops dataset=omnifall/video/oops dataset@dataset_val=omnifall/video/oops preference=random dpo.num_train_epochs=4`
- Git revision: `e1f3896` (clean tree).

### Results

Teacher-forced validation metrics (a clip counts as correct only if every answer token is the argmax, which equals greedy decoding):

| Epoch | DPO loss | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.693 | 45.5% | 31.2% | 27.5% | 65.7% | 25.5% |
| 1 | 0.236 | 52.1% | 29.6% | 28.7% | 68.3% | 64.9% |
| 2 | 0.216 | 56.0% | 33.4% | 31.9% | 71.7% | 67.5% |
| 3 | 0.208 | **56.7%** | **33.4%** | **32.0%** | **73.0%** | 66.7% |
| 4 | 0.206 | 56.5% | 33.2% | 32.0% | **73.0%** | **67.5%** |

Best checkpoint by balanced accuracy: epoch 3 (`checkpoint-306`). Not yet evaluated on test.

### Finding

The classification metrics track what the earlier runs showed on test: at step 0 they match zero-shot greedy decoding (accuracy 45.5% on validation vs 45.4% on test; fall F1 65.7% vs 67.8%), balanced accuracy dips after epoch 1 and recovers from epoch 2. They are a usable selection signal; DPO loss decreases monotonically and is not. Epochs 2–4 are within 0.2 points of balanced accuracy, so a short schedule is enough.

### Artifacts

- Job 31977, log `logs/slurm/falldet-dpo-31977.out`
- W&B: `moritzm00/falldet-mllm-finetune/xhe9sy4e`
- Training: `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_xhe9sy4e/` (checkpoints 102/204/306/408)

## 2026-09-27 — Teacher-forced label scores for per-clip hard DPO negatives

Logged: 2026-09-27

### Goal

Prepare DPO on top of the OOPS SFT model (`74ilzwnw`). Confusion negatives from the zero-shot model do not fit it (another model's errors, and they pushed label priors), and the SFT model's own train predictions are likely memorized. Instead, score all 16 labels per clip with the SFT model and reject the highest-scoring wrong label, which gives a hard negative for every clip, including those the model gets right.

### Implementation

- `scripts/score_labels.py` (+ `slurm/score_labels.sbatch`): for each clip, one vLLM request per label whose prompt ends with that label's exact DPO completion (`The best answer is: <label><|im_end|>\n`; prompt and completion come from the same helpers as DPO). Scores are the summed prompt logprobs of the label-dependent completion tokens (the 5 shared leading tokens are skipped). Output is a prediction JSONL (`predicted_label` = top label) with `label_logprobs`, plus the usual metrics.
- `preference=scores` (`src/falldet/training/scored.py`): the rejected label is the highest-scoring wrong label; rows without scores (e.g. validation without `validation_scores_path`) get random negatives. Warns if the scores came from a different prompt config or adapter than `dpo.sft_adapter_path`.
- Prefix caching cannot be used: with `skip_reading_prefix_cache=False`, vLLM 0.20 returned out-of-range token ids in the prompt logprobs (`OverflowError`); by default vLLM already skips reading the cache for prompt-logprob requests. So each label is a full prefill: ~1.3 s per clip for all 16 labels on one H100.
- A failing run left the vLLM engine process alive and the Slurm job hanging after `sys.exit`; the script now exits with `os._exit(1)` on errors.

### Smoke test (zero-shot model, 32 OOPS train clips)

- 43.5 s for 32 clips (job 32046, W&B `fall-detection-label-scores-smoke/twjcrxna`).
- Top label agrees with greedy zero-shot predictions (`themoiiq`) on 29/32 clips. `themoiiq` ran with `prompt.clip_overlap_note=false`, the scorer with `true`; the 3 disagreements have clear margins (3–13 nats), consistent with the prompt difference. Not yet confirmed by rescoring without the note.

### SFT scoring (`74ilzwnw` adapter = `checkpoint-300`)

- Validation (409 clips, job 32066, W&B `fall-detection-label-scores/nclzeglc`, 544 s): accuracy 66.0%, balanced accuracy 39.1%, macro F1 36.6%. Accuracy matches the SFT run's own final validation accuracy (66.0%), a cross-check that the scores reproduce the model's decisions.
- Hardest-negative pairs on validation: other→standing 35, standing→other 32, walk→other 31, fall→jump 24, fall→other 21. Net chosen − rejected: fall +70, fallen +8 vs other −23, standing −21, lie_down −15, jump −14, lying −14. The label-prior imbalance seen with confusion negatives appears here too, now favouring fall.
- Train (809 clips, job 32065, W&B `fall-detection-label-scores/4tzsgxi1`, 1,069 s): accuracy 79.4%, balanced accuracy 75.9%, macro F1 77.3%. The SFT model has **not** memorized its 809 training clips after 300 steps: 167 are still wrong, the median positive-minus-hardest-negative margin is 2.25 nats, and 381 clips have a margin below 2 nats, so DPO gets real signal from these negatives.
- Hardest-negative pairs on train: other→standing 74, fall→jump 60, standing→other 56, fall→other 47, walk→other 45, stand_up→fallen 37, other→walk 35. Net chosen − rejected: fall +122, stand_up +22, sitting +11 vs standing −47, jump −43, lie_down −26, lying −24, other −23. As on validation, DPO will push the prior towards fall and away from standing/jump.

### DPO from the SFT adapter with score negatives

- Setup: `dpo=oops` from `74ilzwnw/adapter`, train and validation negatives from the scores above, 2 epochs (204 steps), validation and checkpoints every 50 steps, selection by validation balanced accuracy; otherwise as the random-negative run (beta 0.1, LR 1e-5 cosine, effective batch 8 on 2× H100, Liger off). 33 min.
- Command: `sbatch --gres=gpu:2 --cpus-per-task=32 --mem=256G --time=04:00:00 slurm/dpo.sbatch dpo=oops dataset=omnifall/video/oops dataset@dataset_val=omnifall/video/oops dpo.sft_adapter_path=outputs/training/Qwen3-VL-8B-Instruct-F16at7.5_74ilzwnw/adapter dpo.num_train_epochs=2 dpo.eval_strategy=steps dpo.eval_steps=50 dpo.save_strategy=steps dpo.save_steps=50 preference=scores preference.train_scores_path=outputs/predictions/fall-detection-label-scores/4tzsgxi1.jsonl preference.validation_scores_path=outputs/predictions/fall-detection-label-scores/nclzeglc.jsonl`
- Git revision: `909656a` plus the uncommitted label-scoring code.

| Step | DPO loss | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (SFT) | 0.693 | 66.3% | 38.3% | **38.6%** | 85.3% | **77.7%** |
| 50 | 0.604 | 65.8% | 37.9% | 35.8% | 81.1% | 74.4% |
| 100 | 0.592 | 66.8% | 37.7% | 35.0% | 84.9% | 72.3% |
| 150 | 0.665 | 67.5% | 38.6% | 36.1% | **85.9%** | 74.4% |
| 200 | 0.679 | 67.5% | 38.8% | 36.5% | **85.9%** | 73.8% |
| 204 | 0.680 | **67.7%** | **39.0%** | 36.5% | **85.9%** | 75.3% |

- Best checkpoint: `checkpoint-204` (exported `adapter/`).
- Full OOPS-CS test (2,804 clips; job 32359, W&B `fall-detection-dpo-eval/ym1mcl23`) vs the SFT adapter (W&B `fall-detection-sft-eval/olspo2r0`, same eval setup):

| Model | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 | Fall ∪ fallen F1 |
|---|---:|---:|---:|---:|---:|---:|
| SFT `74ilzwnw` | **64.6%** | 38.0% | 39.3% | **84.9%** | **68.3%** | **83.3%** |
| + DPO score negatives, step 204 | 64.2% | **39.4%** | **40.1%** | 83.7% | 68.1% | 83.2% |

- On test, 251/2,804 predictions changed. The balanced-accuracy and macro-F1 gains come from a few rare-class clips (lying 0 → 2/17, lie_down 0 → 2/11, squatting 2 → 4/7, squat_down 2 → 3/14, kneeling 12 → 14/45); recall fell for standing (0.53 → 0.49), sitting (0.53 → 0.46), stand_up (0.55 → 0.53) and jump (0.49 → 0.47), the labels the negatives rejected most. Frequent classes are essentially unchanged. Validation (macro F1 −2.1) and test (+0.8) disagree in sign, so the effect is within noise.
- Predicted-label shares moved with the negative imbalance: fall 0.218 → 0.288 (step 50) → 0.254 (step 100) vs true 0.230; standing 0.098 → 0.034 → 0.064 vs true 0.105.
- Finding: DPO with hardest-wrong-label negatives does not clearly improve the OOPS SFT model. Validation: +1.4 accuracy, +0.7 balanced accuracy, −2.1 macro F1; test: −0.4 accuracy, +1.4 balanced accuracy, +0.8 macro F1, driven by a handful of rare-class clips. Validation DPO loss rose from step 100 on while accuracy improved.
- Per-class sensitivities in these eval logs (and all earlier ones) were misaligned by a bug in `src/falldet/metrics/base.py` (per-class sklearn scores indexed by position among present labels, not by class index), fixed on 2026-09-27. Accuracy, balanced accuracy, macro F1 and OOPS fall/fallen metrics were not affected.

### Follow-up: label-balanced score negatives

- `preference.balance=true` (`05342d7`): rejected labels are assigned so every label is rejected exactly as often as it is chosen (net 0 for all labels), maximizing the summed score of the rejected labels (linear assignment). On the SFT train scores 70% of clips keep their hardest wrong label, 94% one of their top three; mean margin 2.59 vs 2.25. Most frequent pairs become two-way: other→fall 64, fall→other 61, standing→other 52, fallen→fall 49, other→standing 48, fall→fallen 41.
- Otherwise identical to the run above (job 32385, W&B `falldet-mllm-finetune/xje7ahi5`, 33 min, revision `05342d7`).

| Step | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 |
|---:|---:|---:|---:|---:|---:|
| 0 (SFT) | 66.3% | 38.3% | **38.6%** | **85.3%** | **77.7%** |
| 50 | **67.0%** | **39.3%** | 36.1% | 85.1% | 75.3% |
| 100 | 64.6% | 38.2% | 35.3% | 83.6% | 71.6% |
| 150 | 65.3% | 38.3% | 33.2% | 84.3% | 73.2% |
| 200 | 65.8% | 38.4% | 33.4% | 84.3% | 73.8% |
| 204 | 66.3% | 38.8% | 33.7% | 84.3% | 72.3% |

Best checkpoint: step 50. Full OOPS-CS test (job 32418, W&B `fall-detection-dpo-eval/8lc8j0w1`):

| Model | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 | Fall ∪ fallen F1 |
|---|---:|---:|---:|---:|---:|---:|
| SFT `74ilzwnw` | **64.6%** | 38.0% | 39.3% | **84.9%** | **68.3%** | **83.3%** |
| + DPO score negatives, step 204 | 64.2% | 39.4% | 40.1% | 83.7% | 68.1% | 83.2% |
| + DPO balanced score negatives, step 50 | 64.0% | **42.1%** | **42.4%** | 83.3% | 63.5% | 80.3% |

- The +4.1 balanced-accuracy gain is almost entirely rare classes, each weighted 1/16: kneel_down 0 → 1/3 clips (+2.1 points alone), lie_down 0 → 3/11 (+1.7), lying 0 → 2/17 (+0.7), squat_down 2 → 3/14 (+0.4). Macro F1 gains the same way.
- Frequent classes trade off: stand_up recall 0.55 → 0.66 and walk 0.64 → 0.70, but fallen 0.71 → 0.58 (61 of 318 fallen clips now predicted stand_up), standing 0.53 → 0.45, sitting 0.53 → 0.41. 388/2,804 test predictions changed.
- Finding: balancing removes the push towards "fall", but DPO on top of the SFT model still gives no robust improvement. Validation shows only a step-50 bump, and the test gains hang on a handful of rare-class clips while fallen detection (a primary target) drops 4.8 F1 points. The unbalanced run's net push was not the main reason DPO fails to help here.

### Artifacts

- DPO: job 32181, W&B `falldet-mllm-finetune/7lh9v3yp`, `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_7lh9v3yp/`; balanced: job 32385, `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_xje7ahi5/`
- Test predictions: `outputs/predictions/fall-detection-dpo-eval/{ym1mcl23,8lc8j0w1}.jsonl`
- Scores: `outputs/predictions/fall-detection-label-scores/4tzsgxi1.jsonl` (train), `nclzeglc.jsonl` (validation); smoke `outputs/predictions/fall-detection-label-scores-smoke/twjcrxna.jsonl`
- Logs: `logs/slurm/falldet-score-val-32066.out`, `logs/slurm/falldet-score-train-32065.out`, `logs/slurm/falldet-score-smoke-32046.out`

## 2026-09-27 — OOPS SFT (rank 8, 300 steps) and confusion DPO on top

Logged: 2026-09-27

### Goal

Train a short LoRA SFT baseline on OOPS-CS, then continue it with confusion-guided DPO built from the SFT model's own train-split errors, and compare both with zero-shot and the DPO-from-pretrained runs on the full test split.

### SFT setup

- Pretrained Qwen3-VL-8B-Instruct + fresh LoRA rank 8 (alpha 16, dropout 0.05, all attention/MLP projections), 16 frames at 7.5 FPS, size 448, default prompt (with clip overlap note)
- `training=full`: 300 steps; per-device batch 8, 2 GPUs → effective batch 16, 50 steps/epoch → 6 epochs; LR 1e-4 cosine, 30 warmup steps; FlashAttention 2 + Liger
- Eval and save every 50 steps (= every epoch) plus at start, on all 409 validation clips; best checkpoint by `eval_balanced_accuracy`
- Hardware: 2× H100 (`gpu-h100`), ~2.9 s/step, 19 min training, 22 min job
- Command: `sbatch --gres=gpu:2 --cpus-per-task=32 --mem=256G --time=06:00:00 slurm/train.sbatch training=full training.max_steps=300 lora.r=8 lora.lora_alpha=16 training.eval_steps=50 training.save_steps=50 training.save_total_limit=6 training.load_best_model_at_end=true training.metric_for_best_model=eval_balanced_accuracy 'wandb.tags=[sft,oops,r8,300steps]'`
- Git revision: `e1f3896` (clean tree). The validation metrics therefore used the old SFT decoding (every teacher-forced argmax token after a mismatch is decoded and keyword-parsed); accuracy and balanced accuracy are unaffected, macro F1 is not comparable with DPO runs. `b041a41` replaced it with the shared greedy-equivalent DPO decoding (on the untrained model: accuracy 46.0% vs 45.5%, balanced accuracy 31.3% vs 31.2%, macro F1 30.4% vs 27.5%).

### SFT results

Teacher-forced validation (old SFT decoding):

| Epoch | Loss | Accuracy | Balanced accuracy |
|---:|---:|---:|---:|
| 0 | 0.711 | 46.0% | 31.3% |
| 1 | 0.150 | 59.4% | 37.7% |
| 2 | 0.135 | 64.1% | 36.1% |
| 3 | 0.133 | 64.3% | 36.3% |
| 4 | 0.138 | 64.1% | 37.6% |
| 5 | 0.148 | 65.8% | 37.3% |
| 6 | 0.147 | **66.0%** | **37.7%** |

Best checkpoint: `checkpoint-300` (37.72% vs 37.69% at epoch 1), exported as `adapter/`. vLLM greedy (`experiment=zeroshot`, same prompt): train split (809 clips) accuracy 79.1%, balanced accuracy 75.9%, macro F1 77.3%; test split in the table below. Validation tracked test closely (balanced accuracy 37.7% vs 38.0%).

### DPO on top of SFT

- Initialization: `dpo.sft_adapter_path` = the SFT `adapter/`; the frozen reference is a copy of it (step-0 DPO loss 0.6931 = ln 2)
- Negatives: `preference=confusion` from the SFT model's own vLLM train predictions (`qisfpq56.jsonl`, 169/809 wrong). Mining: 169 train clips use the model's own wrong prediction; the other 640 train clips and all 409 validation clips are sampled from the SFT confusion matrix (0.9 errors + 0.1 uniform)
- `dpo=oops` with 2 epochs (204 steps): sigmoid, beta 0.1, LR 1e-5 cosine, 10% warmup; per-device batch 1, gradient accumulation 4, 2 GPUs → effective batch 8; Liger off; eval every epoch and at start, best by `eval_balanced_accuracy`
- Hardware: 2× H100, 7.2 s/step, 24.5 min training, 26 min job
- Command: `sbatch --gres=gpu:2 --cpus-per-task=32 --mem=256G --time=03:00:00 slurm/dpo.sbatch dpo=oops dpo.num_train_epochs=2 dpo.sft_adapter_path=outputs/training/Qwen3-VL-8B-Instruct-F16at7.5_74ilzwnw/adapter dataset=omnifall/video/oops dataset@dataset_val=omnifall/video/oops preference=confusion 'preference.train_predictions_paths=[outputs/predictions/fall-detection-sft-eval/qisfpq56.jsonl]' 'wandb.tags=[dpo,confusion,oops,on-sft,sft-74ilzwnw]'`
- Git revision: `909656a` plus another session's uncommitted refactor (`align_records_to_dataset`, `completion_text`, `preference_prompt_config`); read before the run, behavior-preserving for this path, confirmed by the 169 own-error rows in the mining manifest.

Teacher-forced validation (shared decoding):

| Epoch | DPO loss | Accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|---:|---:|
| 0 (= SFT) | 0.693 | 66.3% | 38.3% | 38.6% |
| 1 | 0.308 | **66.8%** | **39.0%** | 36.1% |
| 2 | 0.328 | 66.3% | 38.5% | 36.0% |

Best checkpoint: epoch 1 (`checkpoint-102`), exported as `adapter/`.

### Test results (OOPS-CS test, 2,804 clips, vLLM greedy)

| Model | Accuracy | Balanced accuracy | Macro F1 | Fall F1 | Fallen F1 | Fall ∪ fallen F1 |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot (`p1r3exbe`) | 45.4% | 36.4% | 26.3% | 67.8% | 29.2% | — |
| DPO random negatives from pretrained, epoch 3 | 56.6% | 36.2% | 34.3% | — | — | — |
| SFT r8, 300 steps | **64.6%** | 38.0% | **39.3%** | **84.9%** | **68.3%** | **83.3%** |
| SFT → confusion DPO, epoch 1 | 63.7% | **40.2%** | 39.0% | 83.6% | 67.8% | 82.6% |

Fall: sensitivity 83.8% → 82.2%, precision 86.0% → 84.9%. Fallen: sensitivity 71.4% → 69.2%, precision 65.4% → 66.5%. Fall ∪ fallen: sensitivity 83.9% → 82.3%, specificity 91.0% → 91.2%.

Per-class test recall, SFT → SFT+DPO: jump 0.49 → 0.60 (167 clips), sitting 0.53 → 0.59, squatting 0.29 → 0.57, lie_down 0.00 → 0.18, lying 0.00 → 0.12, kneeling 0.27 → 0.29; other 0.65 → 0.59, fall 0.84 → 0.82, fallen 0.71 → 0.69, crawl 0.33 → 0.00 (3 clips); walk, stand_up, standing, sit_down, squat_down, kneel_down unchanged. Prediction counts: other 719 → 589, jump 123 → 162, sitting 72 → 107, lie_down 4 → 18, lying 1 → 9.

### Finding

SFT is the strongest single step: +19 points accuracy, +13 macro F1 and +17 fall F1 over zero-shot in 22 minutes, but balanced accuracy only +1.6, because the rare classes (1–8 train clips each) are memorized (train balanced accuracy 75.9% vs test 38.0%). Confusion DPO on top of it, with the SFT model's own errors as negatives, un-collapses rare classes and moves predictions away from "other" (+2.2 balanced accuracy on test, the best so far), at the cost of −0.9 accuracy and slightly lower fall/fallen F1; macro F1 is flat because the extra rare-class predictions are often wrong. Validation predicted the direction (+0.7 balanced accuracy). Unlike confusion DPO from the zero-shot model, starting from SFT did not collapse rare classes. Many rare-class test counts are 3–17 clips, so their recall changes rest on a few clips; jump and other are the robust shifts.

### Artifacts

- SFT: job 31978, W&B `moritzm00/falldet-mllm-finetune/74ilzwnw`, `outputs/training/Qwen3-VL-8B-Instruct-F16at7.5_74ilzwnw/` (checkpoints 50–300, `adapter/` = 300)
- SFT vLLM evals: train job 31983, W&B `fall-detection-sft-eval/qisfpq56`, `outputs/predictions/fall-detection-sft-eval/qisfpq56.jsonl`; test job 31986, W&B `fall-detection-sft-eval/olspo2r0`, `outputs/evaluation_results/fall-detection-sft-eval/test_results_sft-r8-300steps-74ilzwnw-best-test_olspo2r0.json` (the train-split file is also prefixed `test_results_`)
- DPO: job 32042, W&B `moritzm00/falldet-mllm-finetune/fh51hvwg`, `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_fh51hvwg/` (checkpoints 102/204, `adapter/` = 102, `preference_mining.jsonl`)
- DPO test eval: job 32055, W&B `fall-detection-dpo-eval/s90jn7pi`, `outputs/evaluation_results/fall-detection-dpo-eval/test_results_dpo-confusion-on-sft-fh51hvwg-ep1_s90jn7pi.json`, `outputs/predictions/fall-detection-dpo-eval/s90jn7pi.jsonl`
