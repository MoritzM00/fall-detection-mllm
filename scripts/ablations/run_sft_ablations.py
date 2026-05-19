#!/usr/bin/env python
"""Run SFT fine-tuning LoRA rank ablations on OOPS.

Wraps `scripts/train_sft.py` with Hydra overrides. Default sweep:

    --placement  both                  # all attention + MLP modules
    --rank       {4, 8, 16, 32}        # LoRA rank r (alpha = 2r)
    --dataset    oops                  # OOPS training split

Validation set is always forced to match the training set; the existing
`training.max_eval_samples_per_ds` downsampling in `config/training/full.yaml` is
inherited. Every other hyperparameter (LR, steps, frames, ...) stays at
the values configured by `training=full` + `lora/train.yaml`.

Examples:

    # Default rank sweep: placement=both, r in {4,8,16,32}, dataset=oops
    python scripts/ablations/run_sft_ablations.py --dry-run

    # Single rank
    python scripts/ablations/run_sft_ablations.py --rank 8
"""

from __future__ import annotations

import argparse
import subprocess

PLACEMENT_MODULES: dict[str, list[str]] = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "both": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

DATASET_GROUP: dict[str, str] = {
    "oops": "omnifall/video/oops",
    "staged-oops": "omnifall/video/staged-oops",
    "all": "omnifall/video/all",
    # data-mix ablations (effective_batch=16, cs split for lab, rand for wanfall)
    "staged": "omnifall/video/staged-cs",
    "staged-oops-wanfall": "combined/video/wanfall-rand-staged-cs-oops",
    "wanfall": "wanfall/video/rand",
}

# Eval group for each training dataset: staged splits are replaced by cmdfall
# as the representative staged benchmark. Mixed datasets mirror the training
# composition but swap the staged component for cmdfall.
DATASET_VAL_GROUP: dict[str, str] = {
    "oops": "omnifall/video/oops",
    "staged": "omnifall/video/cmdfall",
    "staged-oops": "omnifall/video/oops-cmdfall",
    "staged-oops-wanfall": "combined/video/wanfall-oops-cmdfall",
    "wanfall": "wanfall/video/rand",
    "all": "combined/video/wanfall-oops-cmdfall",
}


def build_command(
    placement: str,
    rank: int,
    dataset: str,
    params: str,
    model: str,
    num_processes: int,
    alpha_mode: str,
    per_device_batch_size: int | None = None,
) -> list[str]:
    modules = PLACEMENT_MODULES[placement]
    ds_group = DATASET_GROUP[dataset]
    val_group = DATASET_VAL_GROUP.get(dataset, ds_group)
    alpha = rank if alpha_mode == "r" else 2 * rank
    tags = [
        "ablation",
        "sft",
        model,
        f"placement-{placement}",
        f"rank-{rank}",
        f"alpha-{alpha_mode}",
        f"data-{dataset}",
    ]

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "config/accelerate/ddp_bf16.yaml",
        "--num_processes",
        str(num_processes),
        "scripts/train_sft.py",
        "training=full",
        f"model={model}",
        f"model.params={params}",
        f"dataset={ds_group}",
        f"dataset@dataset_val={val_group}",
        f"lora.r={rank}",
        f"lora.lora_alpha={alpha}",
        f"lora.target_modules=[{','.join(modules)}]",
        f"wandb.tags=[{','.join(tags)}]",
    ]
    if per_device_batch_size is not None:
        cmd.append(f"training.per_device_train_batch_size={per_device_batch_size}")
    return cmd


def format_command(cmd: list[str]) -> str:
    parts = []
    for part in cmd:
        if "=" in part and (
            part.startswith("wandb.tags=") or part.startswith("lora.target_modules=")
        ):
            key, value = part.split("=", 1)
            parts.append(f"'{key}={value}'")
        else:
            parts.append(part)
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--placement",
        nargs="+",
        choices=sorted(PLACEMENT_MODULES.keys()),
        default=["both"],
        help="LoRA target placement(s) to sweep (default: both).",
    )
    parser.add_argument(
        "--rank",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32],
        help="LoRA rank(s) to sweep; alpha is set to 2*r (default: 4 8 16 32).",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=sorted(DATASET_GROUP.keys()),
        default=["oops"],
        help="Training dataset(s); val matches train (default: oops).",
    )
    parser.add_argument(
        "--alpha",
        choices=["r", "2r"],
        default="2r",
        help="LoRA alpha convention: r (scale-invariant) or 2r (default).",
    )
    parser.add_argument(
        "--model",
        default="qwenvl",
        choices=["qwenvl", "internvl"],
        help="Hydra model config group to use (default: qwenvl).",
    )
    parser.add_argument("--params", default="8B", help="model.params override (default: 8B).")
    parser.add_argument(
        "--num-processes", type=int, default=2, help="accelerate --num_processes (default: 2)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help=(
            "Override training.per_device_train_batch_size. "
            "Effective batch = per_device * grad_accum * num_processes. "
            "E.g. --per-device-batch-size 8 with --num-processes 2 gives effective batch 16."
        ),
    )
    parser.add_argument(
        "--start-from", type=int, default=0, help="Skip first N experiments (0-indexed)."
    )
    args = parser.parse_args()

    configs = [
        (placement, rank, dataset)
        for placement in args.placement
        for rank in args.rank
        for dataset in args.dataset
    ]

    print(f"Total experiments: {len(configs)}")
    print("Plan:")
    for i, (placement, rank, dataset) in enumerate(configs):
        marker = " (skip)" if i < args.start_from else ""
        print(f"  {i + 1:>3}. placement={placement:<5} rank={rank:<3} dataset={dataset}{marker}")
    print()

    for i, (placement, rank, dataset) in enumerate(
        configs[args.start_from :], start=args.start_from
    ):
        cmd = build_command(
            placement,
            rank,
            dataset,
            args.params,
            args.model,
            args.num_processes,
            args.alpha,
            args.per_device_batch_size,
        )
        print(f"\n{'=' * 60}")
        print(
            f"Experiment {i + 1}/{len(configs)}: placement={placement} rank={rank} dataset={dataset}"
        )
        print(f"{'=' * 60}")
        print(format_command(cmd))
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: experiment failed with exit code {e.returncode}")
            raise

    if not args.dry_run:
        print(f"\n{'=' * 60}\nAll experiments completed!\n{'=' * 60}")


if __name__ == "__main__":
    main()
