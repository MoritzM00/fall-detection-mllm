#!/usr/bin/env python3
"""
Launch multiple zeroshot experiments for InternVL and Qwen VL models.

This script runs the zeroshot experiment across different model sizes sequentially,
with automatic MoE detection and model validation via HuggingFace.

Examples:
    # Run all InternVL models
    python scripts/launch_zeroshot_sweep.py --models internvl

    # Run specific sizes with 4 GPUs
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/launch_zeroshot_sweep.py --models internvl --sizes 38B 241B

    # Run both model families, excluding large models
    python scripts/launch_zeroshot_sweep.py --models internvl qwenvl --exclude-sizes 235B 241B

    # Dry run to preview commands
    python scripts/launch_zeroshot_sweep.py --dry-run --models internvl qwenvl

    # Resume from run #5
    python scripts/launch_zeroshot_sweep.py --start-from 5 --models internvl

    # Override tensor parallel for all runs
    python scripts/launch_zeroshot_sweep.py --models qwenvl --tensor-parallel 4

    # Pass additional Hydra config overrides
    python scripts/launch_zeroshot_sweep.py --models internvl \\
        --extra "data.size=448" "batch_size=16"
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from infreqact.config import resolve_model_path_from_config

if TYPE_CHECKING:
    from typing import Any

# =============================================================================
# Model Definitions
# =============================================================================

# InternVL 3.5 models
INTERNVL_STANDARD = ["1B", "2B", "4B", "8B", "14B", "38B"]
INTERNVL_MOE: dict[str, str] = {
    "20B": "A4B",
    "30B": "A3B",
    "241B": "A28B",
}

# Qwen3-VL Instruct models
QWEN_STANDARD = ["2B", "4B", "8B", "32B"]
QWEN_MOE: dict[str, str] = {
    "30B": "A3B",
    "235B": "A22B",
}

# Valid sizes for validation
VALID_INTERNVL_SIZES = set(INTERNVL_STANDARD) | set(INTERNVL_MOE.keys())
VALID_QWEN_SIZES = set(QWEN_STANDARD) | set(QWEN_MOE.keys())


# =============================================================================
# Helper Functions
# =============================================================================


def build_model_config(family: str, size: str) -> dict[str, Any]:
    """
    Build a model config dict compatible with resolve_model_path_from_config.

    Args:
        family: Model family ('internvl' or 'qwenvl')
        size: Model size (e.g., '8B', '30B')

    Returns:
        Config dict with family, version, variant, params, and active_params
    """
    if family == "internvl":
        config = {
            "family": "InternVL",
            "version": "3_5",
            "variant": None,
            "params": size,
            "active_params": INTERNVL_MOE.get(size),  # None for standard models
        }
    elif family == "qwenvl":
        config = {
            "family": "Qwen",
            "version": "3",
            "variant": "Instruct",
            "params": size,
            "active_params": QWEN_MOE.get(size),  # None for standard models
        }
    else:
        raise ValueError(f"Unknown model family: {family}")

    return config


def get_model_path(family: str, size: str) -> str:
    """
    Get HuggingFace model path using existing config resolution.

    Args:
        family: Model family ('internvl' or 'qwenvl')
        size: Model size (e.g., '8B', '30B')

    Returns:
        Full HuggingFace model path (e.g., 'OpenGVLab/InternVL3_5-8B-HF')
    """
    config_dict = build_model_config(family, size)
    config = OmegaConf.create(config_dict)
    return resolve_model_path_from_config(config)


def is_moe_model(family: str, size: str) -> bool:
    """
    Check if model is a Mixture of Experts model.

    Args:
        family: Model family ('internvl' or 'qwenvl')
        size: Model size (e.g., '8B', '30B')

    Returns:
        True if model is MoE, False otherwise
    """
    if family == "internvl":
        return size in INTERNVL_MOE
    elif family == "qwenvl":
        return size in QWEN_MOE
    return False


def validate_size(family: str, size: str) -> bool:
    """
    Validate that the size is valid for the given model family.

    Args:
        family: Model family ('internvl' or 'qwenvl')
        size: Model size to validate

    Returns:
        True if valid, False otherwise
    """
    if family == "internvl":
        return size in VALID_INTERNVL_SIZES
    elif family == "qwenvl":
        return size in VALID_QWEN_SIZES
    return False


def validate_model_exists(model_path: str) -> bool:
    """
    Validate that model exists on HuggingFace.

    Args:
        model_path: Full HuggingFace model path

    Returns:
        True if model exists, False otherwise
    """
    try:
        from huggingface_hub import model_info

        model_info(model_path)
        return True
    except Exception as e:
        logging.debug(f"Model validation failed for {model_path}: {e}")
        return False


def validate_all_models(runs: list[tuple[str, str]]) -> bool:
    """
    Validate all models before starting runs.

    Args:
        runs: List of (family, size) tuples

    Returns:
        True if all models valid, False otherwise
    """
    logging.info("Validating models on HuggingFace...")

    all_valid = True
    for family, size in runs:
        model_path = get_model_path(family, size)
        if validate_model_exists(model_path):
            logging.info(f"  ✓ {model_path}")
        else:
            logging.error(f"  ✗ {model_path} not found")
            all_valid = False

    return all_valid


# =============================================================================
# Command Generation
# =============================================================================


def generate_run_command(
    family: str,
    size: str,
    tensor_parallel: int | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """
    Generate the vllm_inference.py command.

    Args:
        family: Model family ('internvl' or 'qwenvl')
        size: Model size (e.g., '8B', '30B')
        tensor_parallel: Optional tensor parallel size override
        extra_args: Additional Hydra config overrides

    Returns:
        Complete command string
    """
    config_dict = build_model_config(family, size)

    # Base command
    cmd_parts = [
        "python scripts/vllm_inference.py",
        "experiment=zeroshot",
        f"model={family}",
        f"model.params={size}",
    ]

    # Add active_params for MoE models
    if config_dict["active_params"]:
        cmd_parts.append(f"model.active_params={config_dict['active_params']}")
        cmd_parts.append("vllm.enable_expert_parallel=true")

    # Add tensor parallel override if specified
    if tensor_parallel is not None:
        cmd_parts.append(f"vllm.tensor_parallel_size={tensor_parallel}")

    # Add extra arguments
    if extra_args:
        cmd_parts.extend(extra_args)

    return " ".join(cmd_parts)


def build_run_list(
    models: list[str],
    sizes: list[str] | None = None,
    exclude_sizes: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Build list of (family, size) tuples to run.

    Args:
        models: List of model families ('internvl', 'qwenvl')
        sizes: Optional list of specific sizes to include
        exclude_sizes: Optional list of sizes to exclude

    Returns:
        List of (family, size) tuples
    """
    runs = []

    for model_family in models:
        if model_family == "internvl":
            all_sizes = INTERNVL_STANDARD + list(INTERNVL_MOE.keys())
        elif model_family == "qwenvl":
            all_sizes = QWEN_STANDARD + list(QWEN_MOE.keys())
        else:
            logging.warning(f"Unknown model family: {model_family}, skipping")
            continue

        # Filter by --sizes if specified
        if sizes:
            # Validate provided sizes
            for s in sizes:
                if not validate_size(model_family, s):
                    logging.warning(f"Size '{s}' is not valid for {model_family}, skipping")
            all_sizes = [s for s in all_sizes if s in sizes]

        # Filter by --exclude-sizes if specified
        if exclude_sizes:
            all_sizes = [s for s in all_sizes if s not in exclude_sizes]

        # Add to runs
        for size in all_sizes:
            runs.append((model_family, size))

    return runs


# =============================================================================
# Execution
# =============================================================================


def execute_runs(
    runs: list[tuple[str, str]],
    start_from: int = 1,
    dry_run: bool = False,
    tensor_parallel: int | None = None,
    extra_args: list[str] | None = None,
    cooldown: int = 10,
) -> None:
    """
    Execute all runs sequentially.

    Args:
        runs: List of (family, size) tuples
        start_from: Run number to start from (1-indexed)
        dry_run: If True, only print commands without executing
        tensor_parallel: Optional tensor parallel size override
        extra_args: Additional Hydra config overrides
        cooldown: Seconds to wait between runs
    """
    total_runs = len(runs)
    successful = 0
    failed = 0
    failed_runs: list[tuple[str, str, int]] = []

    # Setup logging to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"zeroshot_sweep_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    prefix = "[DRY RUN] " if dry_run else ""
    logging.info(f"\n{prefix}Starting Zeroshot Sweep - {total_runs} models")
    logging.info("=" * 60)

    for idx, (family, size) in enumerate(runs, start=1):
        # Skip if resuming from later run
        if idx < start_from:
            logging.info(f"[{idx}/{total_runs}] Skipping {family.upper()} {size}")
            continue

        model_path = get_model_path(family, size)
        is_moe = is_moe_model(family, size)

        # Generate command
        command = generate_run_command(family, size, tensor_parallel, extra_args)

        # Build labels
        moe_label = " (MoE with expert parallel)" if is_moe else ""
        tp_label = f", TP={tensor_parallel}" if tensor_parallel else ""

        # Display progress
        action = "Would run" if dry_run else "Running"
        logging.info(
            f"\n[{idx}/{total_runs}] {action} {family.upper()} {size}{tp_label}{moe_label}"
        )
        logging.info(f"  Model: {model_path}")
        logging.info(f"  Command: {command}")

        if dry_run:
            continue

        # Execute command
        try:
            subprocess.run(
                command,
                shell=True,
                check=True,
            )
            logging.info("  ✓ Completed successfully")
            successful += 1
        except subprocess.CalledProcessError as e:
            logging.error(f"  ✗ Failed with exit code {e.returncode}")
            failed += 1
            failed_runs.append((family, size, e.returncode))
            # Continue to next run even if this one fails

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

        if failed_runs:
            logging.info("\nFailed runs:")
            for family, size, exit_code in failed_runs:
                logging.info(f"  - {family.upper()} {size} (exit code {exit_code})")

        logging.info(f"\nLog file: {log_file}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multiple zeroshot experiments for InternVL and Qwen VL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available model sizes:
  InternVL 3.5:
    Standard: 1B, 2B, 4B, 8B, 14B, 38B
    MoE:      20B (A4B), 30B (A3B), 241B (A28B)

  Qwen3-VL Instruct:
    Standard: 2B, 4B, 8B, 32B
    MoE:      30B (A3B), 235B (A22B)

Examples:
  # Run all InternVL models
  python scripts/launch_zeroshot_sweep.py --models internvl

  # Run specific sizes with multiple GPUs
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python scripts/launch_zeroshot_sweep.py --models internvl --sizes 38B 241B

  # Exclude large models
  python scripts/launch_zeroshot_sweep.py --models qwenvl --exclude-sizes 235B

  # Dry run to preview commands
  python scripts/launch_zeroshot_sweep.py --dry-run --models internvl qwenvl

  # Resume from run #5
  python scripts/launch_zeroshot_sweep.py --start-from 5 --models internvl
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["internvl", "qwenvl"],
        required=True,
        help="Model families to run (internvl, qwenvl, or both)",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        help="Specific model sizes to run (e.g., 2B 4B 8B). "
        "Must be valid for the selected model family.",
    )

    parser.add_argument(
        "--exclude-sizes",
        nargs="+",
        help="Model sizes to exclude (e.g., 38B 241B)",
    )

    parser.add_argument(
        "--tensor-parallel",
        type=int,
        help="Override tensor parallel size for all runs. "
        "By default, uses all visible GPUs (set CUDA_VISIBLE_DEVICES beforehand).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without execution (skips validation)",
    )

    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Resume from specific run number (1-indexed, default: 1)",
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Seconds to wait between runs for vLLM cleanup (default: 10)",
    )

    parser.add_argument(
        "--extra",
        nargs="+",
        help="Additional Hydra config overrides (e.g., 'data.size=448' 'batch_size=16')",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Build run list
    runs = build_run_list(args.models, args.sizes, args.exclude_sizes)

    if not runs:
        logging.error("No models to run! Check your --sizes and --exclude-sizes filters.")
        logging.info("\nValid sizes:")
        logging.info(f"  InternVL: {sorted(VALID_INTERNVL_SIZES)}")
        logging.info(f"  Qwen VL:  {sorted(VALID_QWEN_SIZES)}")
        sys.exit(1)

    logging.info(f"Planned runs: {len(runs)} models")

    # List all planned runs
    for idx, (family, size) in enumerate(runs, start=1):
        moe_label = " (MoE)" if is_moe_model(family, size) else ""
        logging.info(f"  {idx}. {family.upper()} {size}{moe_label}")

    # Validate models (skip in dry-run mode)
    if not args.dry_run:
        logging.info("")
        if not validate_all_models(runs):
            logging.error("\nValidation failed! Some models do not exist on HuggingFace.")
            logging.error("Use --dry-run to preview commands without validation.")
            sys.exit(1)
        logging.info("\nAll models validated successfully!")

    # Execute runs
    execute_runs(
        runs=runs,
        start_from=args.start_from,
        dry_run=args.dry_run,
        tensor_parallel=args.tensor_parallel,
        extra_args=args.extra,
        cooldown=args.cooldown,
    )


if __name__ == "__main__":
    main()
