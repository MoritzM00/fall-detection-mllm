# Makefile for fall-detection-mllm project
# Automates environment setup and common development tasks

# Configuration variables
ENV_NAME := cu130_vllm20_py312
MAX_JOBS := 8

# Phony targets (not actual files)
.PHONY: help env install setup env-hk install-hk flash-attn-hk test lint format clean

# Default target: show help
help:
	@echo "Available targets:"
	@echo "  make env       - Create conda environment from environment.yml"
	@echo "  make install   - Install all pip dependencies (requires active conda env)"
	@echo "  make setup     - Full setup: create env + install dependencies"
	@echo "  make env-hk    - HoreKa: create .venv from Lmod modules (slurm/env.sh)"
	@echo "  make install-hk    - HoreKa: install vLLM (cu129) + requirements into .venv"
	@echo "  make flash-attn-hk - HoreKa: build flash-attn (SFT only; run in a CPU job)"
	@echo "  make test      - Run pytest test suite"
	@echo "  make lint      - Run ruff linter"
	@echo "  make format    - Run ruff formatter"
	@echo "  make clean     - Remove build artifacts and caches"
	@echo ""
	@echo "Environment: $(ENV_NAME)"

# Create conda environment
env:
	@echo "Creating conda environment: $(ENV_NAME)"
	conda env create -f environment.yml -n $(ENV_NAME)
	@echo "Please run 'conda activate $(ENV_NAME)' to activate the environment before installing dependencies."

# Install all pip dependencies (must be run in active conda env)
install:
	@echo "Installing vLLM..."
	uv pip install vllm==0.20.1 --torch-backend=cu130
	@echo "Installing flash-attn (this may take a while)..."
	MAX_JOBS=$(MAX_JOBS) uv pip install flash-attn==2.8.3 --no-build-isolation
	@echo "Installing requirements..."
	uv pip install -r requirements.txt
	@echo "Installing dev requirements..."
	uv pip install -r requirements-dev.txt
	@echo "Installing package in editable mode..."
	uv pip install -e .
	@echo "Installation complete!"

# HoreKa 2: Lmod modules (CUDA 12.9) + uv venv instead of conda. See slurm/README.md.
HK_ENV := . slurm/env.sh
VLLM_CU129_WHEEL := https://github.com/vllm-project/vllm/releases/download/v0.20.1/vllm-0.20.1%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl

env-hk:
	$(HK_ENV) && uv venv --python "$$(command -v python3)" .venv

install-hk:
	$(HK_ENV) && uv pip install "vllm @ $(VLLM_CU129_WHEEL)" --torch-backend=cu129
	$(HK_ENV) && uv pip install -r requirements.txt -r requirements-dev.txt
	$(HK_ENV) && uv pip install av psutil  # from conda in environment.yml
	$(HK_ENV) && uv pip install -e .

# Only needed for SFT (attn_implementation=flash_attention_2); vLLM ships its own kernels.
# No prebuilt wheel for torch 2.11, so this compiles with nvcc: run it in a job with enough CPUs.
flash-attn-hk:
	$(HK_ENV) && uv pip install ninja psutil packaging
	$(HK_ENV) && MAX_JOBS=$${SLURM_CPUS_PER_TASK:-$(MAX_JOBS)} uv pip install flash-attn==2.8.3 --no-build-isolation

# Run tests
test:
	@echo "Running pytest..."
	pytest

# Run linter
lint:
	@echo "Running ruff linter..."
	ruff check

# Run formatter
format:
	@echo "Running ruff formatter..."
	ruff format

typecheck:
	@echo "Running ty type checker..."
	ty check

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"
