# Running on HoreKa 2 (hk2) — Findings & Migration Plan

Status: **investigation only, nothing set up yet** (2026-09-24).
Docs: <https://docs.nhr.kit.edu/> (work in progress), legacy: <https://www.nhr.kit.edu/userdocs/horeka/>.

## Cluster facts

- Account / project group: `hk-project-p0029156` (default account).
- Target partition: **`gpu-h100`** (HoreKa Teal) — 4× H100, AMD EPYC 9354 (64 cores), 768 GiB RAM,
  2× 3.84 TB local NVMe, max walltime 2 days, 21 nodes. B200 (`gpu-b200`, ARM Grace) is not online yet.
- `dev-gpu-h100`: max 1 h, 1 concurrent job — use for smoke tests (`salloc`).
- Other GPU partitions: `gpu-h200` (4× H200), `gpu-h200-8` (8× H200, 1 node).
- `agent` partition: single-core, max 3 days — where Claude Code / VS Code Remote sessions run
  (2 CPUs, no GPU). Interactive AI agent usage on login nodes is prohibited.
- **Nodes are shared**: always specify `--cpus-per-task` and `--mem` (or `--mem-per-cpu`);
  `--partition` and `--time` are mandatory. GPUs via `--gres=gpu:<n>`.
- Architecture: agent node and H100 nodes are **x86_64** → an environment built here runs on H100.
  B200 nodes are ARM and would need a separate environment.
- No conda installed. Software via Lmod/EasyBuild modules (`module avail`, `module spider`).
- Containers: Apptainer (recommended) and enroot/pyxis are available. No Docker/Podman.

## Software modules (verified to load together)

```shell
module load Python/3.12.3-GCCcore-13.3.0 FFmpeg/7.0.2-GCCcore-13.3.0 CUDA/13.0.2
```

This covers everything `environment.yml` gets from conda (python 3.12, ffmpeg, cuda-toolkit 13.0)
except `uv`, `av`, `psutil`, `ninja`, which are pip-installable. Also available: `git`,
`CMake`, `GCC` 13–16, `NCCL`/`cuDNN` (only for CUDA 12.x).

## Storage

| Location | Path | Properties | Suggested use |
| --- | --- | --- | --- |
| `$HOME` | `/home/ka_anthropomatik/ka_dt0662` | 50 GB, 2M inodes | dotfiles, CLI tools |
| `$PROJECT` | `/hfs2/data/project/hk-project-p0029156/ka_dt0662` | permanent, backed up, 10 TB shared quota; one private subdir per member of the project group | repo + uv venv, `.cache/uv`, `.cache/huggingface` (HF_HOME) |
| Workspace | `ws_allocate <name> <days>` | no backup, 60 days, 3× extendable | datasets, HF cache, tensor cache |
| `$TMPDIR` (job) | node-local NVMe | fast, wiped after job | stage dataset / tensor cache per job |

- Repo, venv and caches live together in `$PROJECT`: `$HOME` and `$PROJECT` are the same GPFS
  filesystem but different filesets, so hardlinks (uv cache → `.venv`) fail across them and uv
  would copy every file. `UV_CACHE_DIR` / `HF_HOME` are set in `~/.bashrc` and `~/.zshrc`.
- No workspaces exist yet.
- Shared `/hfs2/data/dataset/datasets` only holds ERA5 → OmniFall / WanFall must be copied over.

## Open questions / to verify

1. **Data location** (workspace vs `$PROJECT` vs split) — decide with supervisor.
2. **NVIDIA driver version** on H100 nodes — `cu130` wheels (vLLM 0.20.1) need a CUDA-13-capable driver.
   Check with `salloc -p dev-gpu-h100 --gres=gpu:1 -t 00:10:00` → `nvidia-smi`.
3. **Internet access from compute nodes** (HF Hub, W&B, PyPI). Could not test: `curl` is denied in
   the repo's `.claude/settings.json`. If offline: pre-download models (`HF_HUB_OFFLINE=1`),
   `wandb.mode=offline` + `wandb sync` from the agent node.

## Required changes (plan)

1. **Environment**: replace `make env` (conda) with `module load …` + uv-managed venv; keep the
   `make install` steps (vLLM, flash-attn, requirements). Build flash-attn inside a CPU/dev job
   (agent node has only 2 CPUs; `MAX_JOBS=8` would be slow/OOM) or use a prebuilt wheel.
2. **Slurm scripts** (new `slurm/` directory):
   - `env.sh`: modules, venv activation, `HF_HOME`, `OMNIFALL_ROOT`, `WANFALL_ROOT`, `VLLM_CONFIGURE_LOGGING=0`.
   - inference job: `-p gpu-h100 --gres=gpu:1 --cpus-per-task=16 --mem=…`.
   - SFT job: `--gres=gpu:2` + `accelerate launch --num_processes 2` (matches `config/accelerate/ddp_bf16.yaml`).
   - tensor cache build on `cpu` partition.
   - Do **not** export `CUDA_VISIBLE_DEVICES` (Slurm sets it). `tensor_parallel_size=null` uses
     `torch.cuda.device_count()`, which respects the allocation — no code change needed.
3. **Ablation runners** (`scripts/ablations/*.py`) run sweeps sequentially via `subprocess`;
   convert to one job per run, e.g. a Slurm job array fed from the `--dry-run` command list.
4. **Paths**: `outputs/` and `logs/` are repo-relative; with the repo in `$PROJECT` they no longer
   hit the `$HOME` quota. `HF_HOME` → `$PROJECT/.cache/huggingface`; point dataset roots at the chosen data location.
   Transfer datasets via `rsync`.
5. **Minor**: `curl` deny in `.claude/settings.json` blocks installers/connectivity checks;
   LaTeX scripts write to `~/thesis-overleaf/...` (irrelevant on cluster).

## Useful commands

```shell
sacctmgr show user -n -P format=DefaultAccount   # default account
squeue --me                                       # own jobs
ws_allocate falldet 60 && ws_find falldet         # create workspace
salloc -p dev-gpu-h100 --gres=gpu:1 -c 16 --mem=64G -t 01:00:00   # interactive GPU
```
