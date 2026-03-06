# Makefile for fall-detection-mllm project
# Automates environment setup and common development tasks

# Configuration variables
ENV_NAME := cu129_vllm15
MAX_JOBS := 8

# Phony targets (not actual files)
.PHONY: help env install setup test lint format clean

# Default target: show help
help:
	@echo "Available targets:"
	@echo "  make env       - Create conda environment from environment.yml"
	@echo "  make install   - Install all pip dependencies (requires active conda env)"
	@echo "  make setup     - Full setup: create env + install dependencies"
	@echo "  make test      - Run pytest test suite"
	@echo "  make lint      - Run ruff linter"
	@echo "  make format    - Run ruff formatter"
	@echo "  make clean     - Remove build artifacts and caches"
	@echo ""
	@echo "Environment: $(ENV_NAME)"
	@echo "Flash-attn max jobs: $(MAX_JOBS)"

# Create conda environment
env:
	@echo "Creating conda environment: $(ENV_NAME)"
	conda env create -f environment.yml -n $(ENV_NAME)
	@echo "Please run 'conda activate $(ENV_NAME)' to activate the environment before installing dependencies."

# Install all pip dependencies (must be run in active conda env)
install:
	@echo "Installing vLLM..."
	uv pip install vllm==0.15.1 --torch-backend=cu129
	@echo "Installing flash-attn (this may take a while)..."
	MAX_JOBS=$(MAX_JOBS) uv pip install flash-attn==2.8.3 --no-build-isolation
	@echo "Installing requirements..."
	uv pip install -r requirements.txt
	@echo "Installing dev requirements..."
	uv pip install -r requirements-dev.txt
	@echo "Installing package in editable mode..."
	uv pip install -e .
	@echo "Installation complete!"

install_new:
	uv pip install vllm --torch-backend=cu129
	MAX_JOBS=$(MAX_JOBS) uv pip install flash-attn --no-build-isolation
	uv pip install accelerate json-repair jupyter matplotlib seaborn pandas scikit-learn rich plotext hydra-core "wandb[media]" pydantic torchcodec==0.9.0 decord torch-c-dlpack-ext numpy
	uv pip install -r requirements-dev.txt
	uv pip install -e .

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
