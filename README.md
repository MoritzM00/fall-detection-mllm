## Create the environment
1. Install Conda
2. Run
```shell
conda env create -f environment.yml
conda activate colab
```
3. Install additional dependencies using uv (installed inside colab environment)
```shell
uv pip install torch torchvision --torch-backend=cu129
uv pip install psutil ninja
uv pip install flash-attn --no-build-isolation
uv pip install transformers accelerate pandas pillow jupyter json-repair qwen_vl_utils[decord]
```
4. Optional Dependencies
```shell
uv pip install pre-commit
```
