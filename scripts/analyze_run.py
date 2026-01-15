#!/usr/bin/env python3
"""Analyze predictions from a run in detail.

Load predictions from local JSONL or W&B, then compute and display full metrics.

Usage:
    # From local file
    python scripts/analyze_run.py --file outputs/predictions/run.jsonl

    # From W&B (using env vars for entity/project)
    python scripts/analyze_run.py --run-id abc123

    # From W&B (explicit entity/project)
    python scripts/analyze_run.py --entity moritzm00 --project fall-detection-zeroshot-v2 --run-id abc123
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from infreqact.evaluation.visual import visualize_evaluation_results
from infreqact.metrics.base import compute_metrics
from infreqact.utils.predictions import extract_labels_for_metrics, load_predictions_jsonl
from infreqact.utils.wandb import load_run_from_wandb

# Load environment variables from .env file (if exists)
load_dotenv()


def print_config(config: dict, indent: int = 0) -> None:
    """Pretty print a nested config dict."""
    prefix = "  " * indent
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze predictions from a run in detail.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file",
        type=Path,
        help="Path to local JSONL predictions file",
    )
    input_group.add_argument(
        "--run-id",
        type=str,
        help="W&B run ID to download predictions from",
    )

    # W&B options
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (defaults to WANDB_ENTITY env var)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project (defaults to WANDB_PROJECT env var)",
    )

    # Analysis options
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Recompute and display metrics from predictions",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Load predictions
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        print(f"Loading predictions from: {args.file}\n")
        metadata, predictions = load_predictions_jsonl(args.file)
        config = metadata.get("config", {})
    else:
        print(f"Loading predictions from W&B run: {args.run_id}\n")
        config, predictions = load_run_from_wandb(
            run_id=args.run_id,
            project=args.project,
            entity=args.entity,
        )

    # Extract labels
    ground_truths, predicted_labels = extract_labels_for_metrics(predictions)

    # Compute metrics (only if requested)
    if args.recompute_metrics:
        print("=" * 70)
        print("Computing Metrics")
        print("=" * 70)
        metrics = compute_metrics(y_pred=predicted_labels, y_true=ground_truths)

        # Get dataset name from config or predictions
        if config:
            dataset_name = config.get("dataset", {}).get("name", "Unknown")
        else:
            dataset_name = "Unknown"

        # Visualize results
        visualize_evaluation_results(metrics, title=f"Results: {dataset_name}")

    # Error analysis
    print()
    print("=" * 70)
    print("Error Analysis")
    print("=" * 70)
    errors = [p for p in predictions if p["label_str"] != p["predicted_label"]]
    print(f"Total predictions: {len(predictions)}")
    print(f"Errors: {len(errors)} ({len(errors) / len(predictions):.1%})")

    # Label distribution
    gt_counts = Counter(ground_truths)
    pred_counts = Counter(predicted_labels)
    print(f"\nGround Truth Distribution: {dict(gt_counts)}")
    print(f"Predicted Distribution:    {dict(pred_counts)}")

    # Show first few errors
    if errors:
        print("\nFirst 5 errors:")
        for pred in errors[:5]:
            print(
                f"  {pred['video_path']}: GT={pred['label_str']} -> Pred={pred['predicted_label']}"
            )
    print()


if __name__ == "__main__":
    args = parse_args()
    main(args)
