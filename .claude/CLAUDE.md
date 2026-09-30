# Repo Guidelines

See `README.md` for project overview, important commands and quickstart guide.

## Dont's

- Do not use typing.Any if possible
- Do not go over 500 lines per file (not a hard guardrail), refactor if this happens


## Project Structure

```
fall-detection-mllm/
├── config/                          # Hydra configuration files
│   ├── inference_config.yaml        # Main config (composes groups below)
│   ├── training_config.yaml         # SFT config; training/ and lora/ hold its presets
│   ├── dpo_config.yaml              # DPO config; dpo/ holds its presets
│   ├── preference/                  # DPO negative selection (random, similarity, confusion, scores)
│   ├── dataset/                     # Dataset + split definitions (omnifall, wanfall, combined)
│   ├── model/                       # Model configs (e.g., QwenVL, InternVL, Molmo)
│   ├── prompt/                      # Prompt templates/components (baseline, fewshot, CoT)
│   ├── sampling/                    # Decoding configs (greedy, nucleus, low_temp)
│   ├── vllm/                        # vLLM engine settings (TP, memory, etc.)
│   └── experiment/                  # Presets (debug, zeroshot, fewshot, zeroshot_cot)
│
├── notebooks/                       # Analysis / exploratory notebooks
├── scripts/                         # Experiment + plotting scripts
│   ├── vllm_inference.py            # Main inference script (Hydra entry point)
│   ├── train_sft.py                 # LoRA SFT (Hydra entry point)
│   ├── train_dpo.py                 # LoRA DPO (Hydra entry point)
│   ├── score_labels.py              # Teacher-forced per-label scores (DPO hard negatives)
│   ├── run_oops_experiments.py      # Run OOPS zero-shot experiments
│   ├── plot_cot_comparison.py       # Plot CoT comparisons
│   ├── plot_comparison_by_size.py   # Plot comparisons by model size
│   ├── ablations/                   # Ablation runners
│   └── latex/                       # LaTeX table generation
│
├── src/falldet/                   # Main Python package
│   ├── data/                        # Dataset handling + exemplar sampling
│   ├── inference/                   # Inference engine + prompt building
│   │   ├── base.py                  # Shared inference interfaces/utilities
│   │   ├── conversation.py          # Conversation/message formatting
│   │   ├── engine.py                # vLLM engine wrapper / runner
│   │   ├── mock_vllm.py             # Mock engine (tests/dev)
│   │   └── prompts/                 # Prompt builder, components, parsers
│   ├── training/                    # SFT/DPO datasets, collators, preferences, eval metrics
│   ├── evaluation/                  # Evaluation orchestration + visualizations
│   ├── metrics/                     # Metric computation (incl. subgroup metrics)
│   ├── utils/                       # Formatting, logging, LaTeX, wandb helpers
│   └── visualization.py             # High-level visualization utilities
│
├── docs/                            # Fine-tuning experiment log (SFT and DPO results)
├── slurm/                           # Slurm job scripts + cluster notes
├── tests/                           # pytest test suite
├── README.md                        # Usage + setup
├── LICENSE
├── pyproject.toml                   # Package configuration
├── environment.yml                  # Conda environment
├── requirements.txt                 # Production dependencies
└── requirements-dev.txt             # Development dependencies
```
