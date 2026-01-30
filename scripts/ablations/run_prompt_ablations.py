#!/usr/bin/env python
"""Run prompt ablation experiments with full 2^3 grid."""

import argparse
import subprocess
from itertools import product

ABLATION_VARS = {
    "prompt.include_role": [True, False],
    "prompt.include_definitions": [True, False],
    "prompt.output_format": ["json", "text"],
}


def generate_experiment_configs():
    """Generate all 8 ablation combinations."""
    keys = list(ABLATION_VARS.keys())
    for values in product(*ABLATION_VARS.values()):
        yield dict(zip(keys, values))


def build_tags(config: dict) -> list[str]:
    """Build descriptive W&B tags from config."""
    tags = ["ablation", "prompt"]

    # Add component tags
    if config["prompt.include_role"]:
        tags.append("role")
    if config["prompt.include_definitions"]:
        tags.append("definitions")

    # Add format tag
    tags.append(f"format-{config['prompt.output_format']}")

    return tags


def build_command(config: dict, model: str = "qwen/instruct", params: str = "8B") -> list[str]:
    """Build hydra command for a config."""
    cmd = [
        "python",
        "scripts/vllm_inference.py",
        "experiment=zeroshot",
        "data.mode=test",
        f"model={model}",
        f"model.params={params}",
        "sampling=qwen3_instruct",
        "vllm.tensor_parallel_size=1",
        "log_videos=0",
    ]

    # Add ablation overrides
    for key, value in config.items():
        val_str = str(value).lower() if isinstance(value, bool) else value
        cmd.append(f"{key}={val_str}")

    # Add descriptive W&B tags - format as ['tag1', 'tag2', ...]
    tags = build_tags(config)
    cmd.append(f"wandb.tags={tags}")

    return cmd


def format_command_for_display(cmd: list[str]) -> str:
    """Format command for readable display."""
    # Find the wandb.tags argument and format it specially
    formatted_parts = []
    for part in cmd:
        if part.startswith("wandb.tags="):
            # Format as "wandb.tags=['tag1', 'tag2']"
            formatted_parts.append(f'"{part}"')
        else:
            formatted_parts.append(part)
    return " ".join(formatted_parts)


def run_experiment(
    config: dict, dry_run: bool = False, model: str = "qwen/instruct", params: str = "8B"
):
    """Run single experiment."""
    cmd = build_command(config, model=model, params=params)

    if dry_run:
        print(f"[DRY-RUN] {format_command_for_display(cmd)}\n")
        return

    print(f"Running: {format_command_for_display(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with exit code {e.returncode}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt ablation experiments for Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip first N experiments (for resuming, 0-indexed)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwenvl",
        help="Model name for experiments (default: qwenvl)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="8B",
        help="Model parameters for experiments (default: 8B)",
    )
    args = parser.parse_args()

    configs = list(generate_experiment_configs())
    print(f"Total experiments: {len(configs)}")

    if args.start_from > 0:
        print(f"Resuming from experiment {args.start_from + 1}")

    for i, config in enumerate(configs[args.start_from :], start=args.start_from):
        print(f"\n{'=' * 60}")
        print(f"Experiment {i + 1}/{len(configs)}")
        print(f"Config: {config}")
        print(f"{'=' * 60}")
        run_experiment(config, args.dry_run, model=args.model, params=args.params)

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("All experiments completed!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
