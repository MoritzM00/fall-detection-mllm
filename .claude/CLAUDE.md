# Infrequent Action Recognition

## Project Overview

**Purpose**: Fall detection and human activity recognition using Multimodal Large Language Models (MLLMs).

**Current Implementation**:
- Zero-shot inference with Qwen3-VL models (2B, 4B, 8B, 32B, MoE)
- Video-based action classification across 16 classes (walk, fall, fallen, sit, stand, lie, etc.)
- Evaluation on fall detection datasets (Omnifall, Wanfall)

**Future Directions**:
- Support for additional models, e.g. InternVL3.5, Molmo2
- Few-shot learning experiments
- Chain-of-Thought (CoT) reasoning (partially implemented)
- GRPO (Group Relative Policy Optimization) finetuning

---

## Tech Stack

Inference Engine: vLLM

## Project Structure

```
infrequent-action-recognition/
├── config/                          # Hydra configuration files
├── scripts/                         # Executable scripts
│   ├── vllm_inference.py            # Main inference script (Hydra entry point)
│
├── src/infreqact/                   # Main Python package
│   ├── data/                        # Dataset handling
│   │
│   ├── inference/                   # Inference logic
│   │   ├── base.py                  # LLM output parsing, input preparation
│   │   └── zeroshot.py              # Zero-shot prompts & collate functions
│   │
│   ├── evaluation/                  # Evaluation orchestration
│   │   ├── base.py                  # Main evaluation entry point
│   │   ├── subgroup.py              # Demographic subgroup evaluation
│   │   └── visual.py                # Terminal-based visualizations
│   │
│   ├── metrics/                     # Metric computation
│   └── utils/                       # Utilities
│
├── tests/                           # pytest test suite
├── pyproject.toml                   # Package configuration
├── environment.yml                  # Conda environment
├── requirements.txt                 # Production dependencies
└── requirements-dev.txt             # Development dependencies
```
---

## Configuration System

**Hydra-based Configuration**:
- Main config: `config/inference_config.yaml`
- Uses hierarchical composition with config groups
- Supports OmegaConf interpolation: `${model_fps}`, `${oc.env:OMNIFALL_ROOT}`

**Config Groups**:
- `dataset/`: Dataset configurations (omnifall, wanfall, combined)
- `model/`: Model configuration (currently only Qwen) 
- `sampling/`: Sampling strategies (greedy, nucleus, low_temp)
- `vllm/`: vLLM engine settings (tensor parallelism, memory)
- `experiment/`: Presets (debug for testing, zeroshot for full eval)

