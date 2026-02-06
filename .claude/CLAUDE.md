# Infrequent Action Recognition

See `README.md` for project overview, important commands and quickstart guide.

## Project Structure

```
infrequent-action-recognition/
├── config/                          # Hydra configuration files
│   ├── inference_config.yaml        # Main config (composes groups below)
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
│   ├── run_oops_experiments.py      # Run OOPS zero-shot experiments
│   ├── plot_cot_comparison.py       # Plot CoT comparisons
│   ├── plot_comparison_by_size.py   # Plot comparisons by model size
│   ├── ablations/                   # Ablation runners
│   └── latex/                       # LaTeX table generation
│
├── src/infreqact/                   # Main Python package
│   ├── data/                        # Dataset handling + exemplar sampling
│   ├── inference/                   # Inference engine + prompt building
│   │   ├── base.py                  # Shared inference interfaces/utilities
│   │   ├── conversation.py          # Conversation/message formatting
│   │   ├── engine.py                # vLLM engine wrapper / runner
│   │   ├── mock_vllm.py             # Mock engine (tests/dev)
│   │   └── prompts/                 # Prompt builder, components, parsers
│   ├── evaluation/                  # Evaluation orchestration + visualizations
│   ├── metrics/                     # Metric computation (incl. subgroup metrics)
│   ├── utils/                       # Formatting, logging, LaTeX, wandb helpers
│   └── visualization.py             # High-level visualization utilities
│
├── tests/                           # pytest test suite
├── README.md                        # Usage + setup
├── LICENSE
├── pyproject.toml                   # Package configuration
├── environment.yml                  # Conda environment
├── requirements.txt                 # Production dependencies
└── requirements-dev.txt             # Development dependencies
```

## Development Guidelines

### 1. Simplicity First (Occam's Razor)

- Prefer simple, readable code over clever or complex solutions
- Choose the simplest approach that solves the problem correctly
- If two solutions work equally well, pick the one that is easier to understand and maintain

### 2. No Premature Optimization

- Write correct, working code first
- Add tests to verify correctness
- Optimize only after profiling identifies actual bottlenecks
- Iterate: correct → tested → optimized

### 3. Test-Driven Development

- Before implementing any feature, discuss with the user:
  - What exactly needs to be tested
  - How the tests will verify correct behavior
  - Be specific about what the tests are checking
- Get explicit confirmation that the test approach is correct before proceeding
- use best practices for writing tests: fixtures, parametrize etc.

### 4. Code Reuse Awareness

- Before implementing new functionality, search the codebase for:
  - Existing utilities that do similar things
  - Patterns already established in the project
  - Code that can be extended rather than duplicated
- Avoid reinventing functionality that already exists
