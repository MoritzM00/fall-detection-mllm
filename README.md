## Create the environment
1. Install Conda
2. Run
```shell
conda env create -f environment.yml
conda activate cuda-129
```
3. Install additional dependencies using uv (installed inside colab environment)
```shell
uv pip install vllm==0.12.0 --torch-backend=cu129
uv pip install psutil ninja
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install transformers accelerate pandas pillow jupyter json-repair "qwen_vl_utils[decord]" matplotlib torch-c-dlpack-ext scikit-learn
uv pip install -e .
```
4. Optional Dependencies
```shell
uv pip install pre-commit ruff pytest
```


Alternative with fixed versions: (with step 2 before)
```shell
uv pip install vllm==0.12.0 --torch-backend=cu129
uv pip install psutil==7.1.3 ninja==1.13.0
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -e .
```
