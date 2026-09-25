#!/bin/bash
# GPU smoke test: driver, torch CUDA, vLLM generate with a small cached model,
# and internet access from the compute node.
#   sbatch slurm/smoke_gpu.sbatch
#SBATCH --job-name=falldet-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh

echo "== node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi | grep "CUDA Version"

echo "== internet"
for url in https://huggingface.co https://pypi.org https://api.wandb.ai; do
    if curl -sS -o /dev/null --max-time 10 "$url"; then echo "ok   $url"; else echo "FAIL $url"; fi
done

echo "== torch"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== vllm"
HF_HUB_OFFLINE=1 python - <<'EOF'
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", max_model_len=2048, gpu_memory_utilization=0.5)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
print("vllm output:", out[0].outputs[0].text)
EOF
echo "== done"
