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

### Artifacts

- DPO: job 32181, W&B `falldet-mllm-finetune/7lh9v3yp`, `outputs/dpo/Qwen3-VL-8B-Instruct-F16at7.5_7lh9v3yp/`
- Scores: `outputs/predictions/fall-detection-label-scores/4tzsgxi1.jsonl` (train), `nclzeglc.jsonl` (validation); smoke `outputs/predictions/fall-detection-label-scores-smoke/twjcrxna.jsonl`
- Logs: `logs/slurm/falldet-score-val-32066.out`, `logs/slurm/falldet-score-train-32065.out`, `logs/slurm/falldet-score-smoke-32046.out`
