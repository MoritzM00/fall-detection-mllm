## Create the environment
1. Install Conda
2. Run
```shell
conda env create -f environment.yml
conda activate cu129_vllm14
```
3. Install additional dependencies using uv (installed inside colab environment)
```shell
uv pip install vllm==0.14.1 --torch-backend=cu129
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -e .
```

At the time of writing, vLLM is compiled for cu129 by default. If you need a different version of
CUDA, you have to install vLLM from source.

## Environment variables

### Required

```shell
OMNIFALL_ROOT=path/to/omnifall
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### Recommended
These variables should be set **before** launching the vllm inference script.
```shell
CUDA_VISIBLE_DEVICES=0 # or e.g., 0,1
VLLM_CONFIGURE_LOGGING=0
```
