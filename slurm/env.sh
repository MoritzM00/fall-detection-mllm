# HoreKa 2 environment: Lmod modules + uv venv.
# Source it (bash or zsh) from an interactive shell or a Slurm job:
#   source slurm/env.sh
# The venv itself is created by `make env-hk` / `make install-hk`.

# Lmod is not always initialised in non-interactive shells (sbatch, make)
if ! type module >/dev/null 2>&1 && [ -n "${LMOD_PKG:-}" ]; then
    if [ -n "${ZSH_VERSION:-}" ]; then
        . "$LMOD_PKG/init/zsh"
    else
        . "$LMOD_PKG/init/bash"
    fi
fi

# Empty when a job runs with --export=NONE; some nodes (e.g. dev-gpu-h100)
# report CLUSTER=hk and get a stale HoreKa 1 MODULEPATH from the system profile
if [[ "${MODULEPATH:-}" != *"/software/easybuild/"* ]]; then
    _arch="$(uname -m)"
    export MODULEPATH="/software/easybuild/$_arch/modules/all/:/software/commercial/$_arch/modules/:/software/community/$_arch/modules/"
    unset _arch
fi

module purge
module load Python/3.12.3-GCCcore-13.3.0 FFmpeg/7.0.2-GCCcore-13.3.0 CUDA/12.9.1

if [ -n "${BASH_VERSION:-}" ]; then
    _falldet_env="${BASH_SOURCE[0]}"
else
    _falldet_env="${(%):-%x}"
fi
export FALLDET_REPO="$(cd "$(dirname "$_falldet_env")/.." && pwd)"
unset _falldet_env

export PROJECT="${PROJECT:-/hfs2/data/project/hk-project-p0029156/ka_dt0662}"
# Cache and venv must share a fileset so uv can hardlink
export UV_CACHE_DIR="$PROJECT/.cache/uv"
export UV_PYTHON_DOWNLOADS=never
export HF_HOME="$PROJECT/.cache/huggingface"
export VLLM_CONFIGURE_LOGGING=0

# Datasets live on the LSDF, which is only mounted in jobs with --constraint=LSDF
export LSDF="${LSDF:-/lsdf/kit/anthropomatik/projects/cvhci}"
export OMNIFALL_ROOT="$LSDF/data/activity/fall_detection/cvhci_fall"
export WANFALL_ROOT="$LSDF/data/activity/WanFall"

if [ -f "$FALLDET_REPO/.venv/bin/activate" ]; then
    . "$FALLDET_REPO/.venv/bin/activate"
fi
