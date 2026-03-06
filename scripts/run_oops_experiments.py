#!/usr/bin/env python3
"""
Run experiments on video datasets for InternVL and Qwen VL models.

This script runs the zeroshot experiment on a video dataset across different
model sizes sequentially, with a cooldown between runs for vLLM cleanup.

Examples:
    # Run all InternVL models on oops (default dataset)
    python scripts/run_oops_experiments.py --model internvl

    # Run all Qwen models on cmdfall
    python scripts/run_oops_experiments.py --model qwenvl --dataset cmdfall

    # Run on multiple datasets
    python scripts/run_oops_experiments.py --model internvl --dataset cmdfall up-fall oops

    # Run both model families on up-fall
    python scripts/run_oops_experiments.py --model internvl qwenvl --dataset up-fall

    # Dry run to preview commands
    python scripts/run_oops_experiments.py --model internvl --dry-run

    # Run specific sizes only
    python scripts/run_oops_experiments.py --model internvl --sizes 2B 4B

    # Run with chain of thought
    python scripts/run_oops_experiments.py --model qwenvl --cot
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

# =============================================================================
# Model Parameter Arrays
# =============================================================================

# Standard models (non-MoE)
INTERNVL_STANDARD = ["2B", "4B", "8B", "14B", "38B"]
QWEN_STANDARD = ["2B", "4B", "8B", "32B"]
MOLMO_STANDARD = ["2B", "4B"]
KEYEVL_STANDARD = ["8B"]  # remove do_resize from vllm config before running keyevl experiments

# MoE models: {total_params: active_params}
INTERNVL_MOE: dict[str, str] = {
    "30B": "A3B",
    # "241B": "A28B",
}
QWEN_MOE: dict[str, str] = {
    "30B": "A3B",
    # "235B": "A22B",
}

VIDEO_DATASETS = [
    "up-fall",
    "le2i",
    "cmdfall",
    "oops",
    "gmdcsa24",
    "edf",
    "occu",
    "caucafall",
    "mcfd",
]
DEFAULT_DATASET = "oops"


# =============================================================================
# Command Generation
# =============================================================================


def is_moe_model(model: str, params: str) -> bool:
    """Check if model is a Mixture of Experts model."""
    if model == "internvl":
        return params in INTERNVL_MOE
    elif model == "qwenvl":
        return params in QWEN_MOE
    return False


def get_active_params(model: str, params: str) -> str | None:
    """Get active params for MoE model, or None for standard models."""
    if model == "internvl":
        return INTERNVL_MOE.get(params)
    elif model == "qwenvl":
        return QWEN_MOE.get(params)
    return None


def generate_command(
    model: str, params: str, cot: bool = False, dataset: str = DEFAULT_DATASET
) -> str:
    """
    Generate the vllm_inference.py command.

    Args:
        model: Model family ('internvl' or 'qwenvl')
        params: Model parameter size (e.g., '8B', '30B')
        cot: Whether to use chain of thought
        dataset: Dataset name to run on

    Returns:
        Complete command string
    """
    experiment = "zeroshot_cot" if cot else "zeroshot"

    cmd_parts = [
        "python scripts/vllm_inference.py",
        f"model={model}",
        f"model.params={params}",
        f"experiment={experiment}",
        f"dataset/omnifall/video@dataset={dataset}",
    ]

    # Add CoT-specific parameters for qwen models
    if cot and model == "qwenvl":
        cmd_parts.append("model.variant=Thinking")

    # Add MoE-specific parameters
    active_params = get_active_params(model, params)
    if active_params:
        cmd_parts.append(f"model.active_params={active_params}")
        cmd_parts.append("vllm.enable_expert_parallel=true")

    return " ".join(cmd_parts)


def get_params_for_model(model: str) -> list[str]:
    """Get all parameter sizes for the given model family (standard + MoE)."""
    if model == "internvl":
        return INTERNVL_STANDARD + list(INTERNVL_MOE.keys())
    elif model == "qwenvl":
        return QWEN_STANDARD + list(QWEN_MOE.keys())
    elif model == "molmo":
        return MOLMO_STANDARD
    elif model == "keyevl":
        return KEYEVL_STANDARD
    else:
        raise ValueError(f"Unknown model: {model}")


# =============================================================================
# Execution
# =============================================================================


def execute_runs(
    models: list[str],
    sizes: list[str] | None = None,
    dry_run: bool = False,
    cooldown: int = 10,
    cot: bool = False,
    datasets: list[str] | None = None,
) -> None:
    """
    Execute all runs sequentially.

    Args:
        models: List of model families ('internvl', 'qwenvl')
        sizes: Optional list of specific sizes to run
        dry_run: If True, only print commands without executing
        cooldown: Seconds to wait between runs
        cot: If True, run chain of thought experiments
        datasets: Dataset names to run on
    """
    if datasets is None:
        datasets = [DEFAULT_DATASET]

    # Build list of (model, params, dataset) tuples
    runs: list[tuple[str, str, str]] = []
    for model in models:
        params_list = get_params_for_model(model)
        if sizes:
            params_list = [p for p in params_list if p in sizes]
        for params in params_list:
            for dataset in datasets:
                runs.append((model, params, dataset))

    if not runs:
        logging.error("No models to run! Check your --sizes filter.")
        logging.info(f"  InternVL standard: {INTERNVL_STANDARD}")
        logging.info(f"  InternVL MoE: {list(INTERNVL_MOE.keys())}")
        logging.info(f"  Qwen standard: {QWEN_STANDARD}")
        logging.info(f"  Qwen MoE: {list(QWEN_MOE.keys())}")
        sys.exit(1)

    total_runs = len(runs)
    successful = 0
    failed = 0

    prefix = "[DRY RUN] " if dry_run else ""
    experiment_type = "chain of thought" if cot else "zeroshot"
    datasets_label = ", ".join(datasets)
    logging.info(
        f"\n{prefix}Running {total_runs} {experiment_type} experiments on [{datasets_label}] datasets"
    )
    logging.info("=" * 60)

    for idx, (model, params, dataset) in enumerate(runs, start=1):
        command = generate_command(model, params, cot=cot, dataset=dataset)

        action = "Would run" if dry_run else "Running"
        moe_label = " (MoE)" if is_moe_model(model, params) else ""
        logging.info(
            f"\n[{idx}/{total_runs}] {action} {model.upper()} {params}{moe_label} on {dataset}"
        )
        logging.info(f"  Command: {command}")

        if dry_run:
            continue

        # Execute command
        try:
            subprocess.run(command, shell=True, check=True)
            logging.info("  Completed successfully")
            successful += 1
        except subprocess.CalledProcessError as e:
            logging.error(f"  Failed with exit code {e.returncode}")
            failed += 1

        # Cooldown between runs (except after last run)
        if idx < total_runs:
            logging.info(f"\n[Cooldown: {cooldown} seconds]")
            time.sleep(cooldown)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)
    logging.info(f"Total runs: {total_runs}")

    if not dry_run:
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {failed}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments on a video dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available model sizes:
  InternVL:
    Standard: {", ".join(INTERNVL_STANDARD)}
    MoE:      {", ".join(f"{k} ({v})" for k, v in INTERNVL_MOE.items())}

  Qwen VL:
    Standard: {", ".join(QWEN_STANDARD)}
    MoE:      {", ".join(f"{k} ({v})" for k, v in QWEN_MOE.items())}

Examples:
  python scripts/run_oops_experiments.py --model internvl
  python scripts/run_oops_experiments.py --model qwenvl
  python scripts/run_oops_experiments.py --model internvl qwenvl
  python scripts/run_oops_experiments.py --model internvl --dry-run
  python scripts/run_oops_experiments.py --model internvl --sizes 2B 4B 30B
        """,
    )

    parser.add_argument(
        "--model",
        nargs="+",
        choices=["internvl", "qwenvl", "molmo", "keyevl"],
        required=True,
        help="Model families to run",
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=VIDEO_DATASETS,
        default=[DEFAULT_DATASET],
        help=f"Video dataset(s) to run on (default: {DEFAULT_DATASET})",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        help="Specific model sizes to run (e.g., 2B 4B 8B)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing",
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Seconds to wait between runs for vLLM cleanup (default: 10)",
    )

    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use chain of thought (sets experiment=zeroshot_cot, model.variant=Thinking for qwen)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    execute_runs(
        models=args.model,
        sizes=args.sizes,
        dry_run=args.dry_run,
        cooldown=args.cooldown,
        cot=args.cot,
        datasets=args.dataset,
    )


if __name__ == "__main__":
    main()
