# Running on HoreKa 2 (hk2) — Findings & Migration Plan

Status (2026-09-26): environment (`slurm/env.sh`, `make env-hk install-hk flash-attn-hk`) and
`slurm/inference.sbatch` and `slurm/train.sbatch` (SFT smoke test on 1 and 2 GPUs) work end to end on `gpu-h100`.
Tensor cache and ablation jobs are not ported yet.
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
module load Python/3.12.3-GCCcore-13.3.0 FFmpeg/7.0.2-GCCcore-13.3.0 CUDA/12.9.1
```

CUDA 12.9 (not 13.0) matches the `vllm+cu129` wheel / torch 2.11+cu129. This covers everything
`environment.yml` gets from conda (python 3.12, ffmpeg, cuda-toolkit)
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
- **Datasets stay on the LSDF** (`/lsdf` → `/gfse/data/LSDF/lsdf01/lsdfos`). It is mounted by the
  Slurm prolog **only for jobs submitted with `--constraint=LSDF`** (`-C LSDF`); all hk2 CPU, agent
  and GPU nodes have the feature. The cvhci project dir is `/lsdf/kit/anthropomatik/projects/cvhci`,
  exported as `$LSDF` in `~/.bashrc` / `~/.zshrc` (on the cvhci cluster set `LSDF=/lsdf`, so paths
  written as `$LSDF/...` work on both systems). `env.sh` sets
  `OMNIFALL_ROOT=$LSDF/data/activity/fall_detection/cvhci_fall` and `WANFALL_ROOT=$LSDF/data/activity/WanFall`.
- The LSDF (`/gfse/data/LSDF/lsdf01`) is mounted in the agent session as well.

## Open questions / to verify

1. ~~**Data location**~~ — resolved 2026-09-25: read directly from the LSDF with `-C LSDF` (see Storage).
2. ~~**NVIDIA driver version**~~ — resolved 2026-09-25 on `dev-gpu-h100` (hkn0901): driver 595.71.05,
   CUDA 13.2 → `cu130` wheels would also work; current env (torch 2.11.0+cu129) runs fine.
3. ~~**Internet access from compute nodes**~~ — resolved 2026-09-25: huggingface.co, pypi.org and
   api.wandb.ai are reachable from `dev-gpu-h100`. A GPU smoke test passed (torch CUDA + vLLM
   generate with Qwen2.5-0.5B).
4. **`dev-gpu-h100` nodes report `CLUSTER=hk`**, so the system profile sets a stale HoreKa 1
   `MODULEPATH`; `env.sh` now overrides it whenever `/software/easybuild/` is missing.

## Required changes (plan)

1. **Environment**: replace `make env` (conda) with `module load …` + uv-managed venv; keep the
   `make install` steps (vLLM, flash-attn, requirements). Build flash-attn inside a CPU/dev job
   (agent node has only 2 CPUs; `MAX_JOBS=8` would be slow/OOM) or use a prebuilt wheel.
2. **Slurm scripts** (new `slurm/` directory):
   - `env.sh`: modules, venv activation, `HF_HOME`, `OMNIFALL_ROOT`, `WANFALL_ROOT`, `VLLM_CONFIGURE_LOGGING=0`.
   - inference job: `-p gpu-h100 --gres=gpu:1 --cpus-per-task=16 --mem=…`.
   - SFT job (`train.sbatch`): `accelerate launch` with one DDP process per GPU; `sbatch --gres=gpu:2 --cpus-per-task=32 --mem=256G`.
   - tensor cache build on `cpu` partition.
   - Do **not** export `CUDA_VISIBLE_DEVICES` (Slurm sets it). `tensor_parallel_size=null` uses
     `torch.cuda.device_count()`, which respects the allocation — no code change needed.
3. **Ablation runners** (`scripts/ablations/*.py`) run sweeps sequentially via `subprocess`;
   convert to one job per run, e.g. a Slurm job array fed from the `--dry-run` command list.
4. **Paths**: `outputs/` and `logs/` are repo-relative; with the repo in `$PROJECT` they no longer
   hit the `$HOME` quota. `HF_HOME` → `$PROJECT/.cache/huggingface`; point dataset roots at the chosen data location.
   Transfer datasets via `rsync`.
5. **Minor**: LaTeX scripts write to `~/thesis-overleaf/...` (irrelevant on cluster).

## Useful commands

```shell
sacctmgr show user -n -P format=DefaultAccount   # default account
squeue --me                                       # own jobs
ws_allocate falldet 60 && ws_find falldet         # create workspace
salloc -p dev-gpu-h100 -C LSDF --gres=gpu:1 -c 16 --mem=64G -t 01:00:00   # interactive GPU + LSDF
sbatch -C LSDF ...                                # any job that reads datasets
```
