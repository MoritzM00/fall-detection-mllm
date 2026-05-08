```bash
  conda env create -n falldet-finetune -f finetune/environment.yml
  conda activate falldet-finetune
  export UV_TORCH_BACKEND=cu129
  uv pip install transformers peft trl accelerate datasets torch torchvision hydra-core pydantic wandb
  uv pip install -e .
```
