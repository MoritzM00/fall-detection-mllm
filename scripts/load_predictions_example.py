#!/usr/bin/env python3
"""Example script showing how to load and use JSONL predictions files.

This script demonstrates various use cases for working with predictions saved
by the inference pipeline without needing to re-run inference.

Usage:
    python scripts/load_predictions_example.py <path_to_predictions.jsonl>
"""

import sys
from pathlib import Path

from infreqact.utils.predictions import (
    extract_labels_for_metrics,
    load_predictions_jsonl,
)


def main(predictions_file: str):
    """Load and analyze predictions file."""
    print(f"Loading predictions from: {predictions_file}\n")

    # Load the predictions file
    metadata, predictions = load_predictions_jsonl(predictions_file)

    # =========================================================================
    # Use Case 1: Inspect run metadata
    # =========================================================================
    print("=" * 70)
    print("Run Metadata")
    print("=" * 70)
    print(f"Model:        {metadata['model']}")
    print(f"Dataset:      {metadata['dataset']}")
    print(f"Timestamp:    {metadata['timestamp']}")
    print(f"W&B Run ID:   {metadata['wandb_run_id']}")
    print(f"Total Predictions: {len(predictions)}")
    print()

    # =========================================================================
    # Use Case 2: Debug the prompt that was used
    # =========================================================================
    print("=" * 70)
    print("Prompt Configuration")
    print("=" * 70)
    prompt_config = metadata["prompt_config"]
    print(f"Output Format:     {prompt_config['output_format']}")
    print(f"Include Role:      {prompt_config['include_role']}")
    print(f"Include Definitions: {prompt_config['include_definitions']}")
    print(f"CoT Enabled:       {prompt_config['cot']}")
    print(f"Labels Used:       {prompt_config.get('labels', 'default')}")
    print()
    print("Full Prompt:")
    print("-" * 70)
    print(metadata["prompt"])
    print()

    # =========================================================================
    # Use Case 3: Recompute metrics
    # =========================================================================
    print("=" * 70)
    print("Recompute Metrics (without re-running inference)")
    print("=" * 70)
    ground_truths, predicted_labels = extract_labels_for_metrics(predictions)

    # Simple accuracy
    correct = sum(g == p for g, p in zip(ground_truths, predicted_labels))
    accuracy = correct / len(ground_truths)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(ground_truths)})")

    # Label distribution
    from collections import Counter

    gt_counts = Counter(ground_truths)
    pred_counts = Counter(predicted_labels)
    print(f"\nGround Truth Distribution: {dict(gt_counts)}")
    print(f"Predicted Distribution:    {dict(pred_counts)}")
    print()

    # =========================================================================
    # Use Case 4: Analyze specific predictions
    # =========================================================================
    print("=" * 70)
    print("Sample Predictions")
    print("=" * 70)
    for i, pred in enumerate(predictions[:3]):
        match = "✓" if pred["label_str"] == pred["predicted_label"] else "✗"
        print(f"{match} Prediction {pred['idx']}:")
        print(f"  Video:     {pred['video_path']}")
        print(f"  Time:      {pred['start_time']:.2f}s - {pred['end_time']:.2f}s")
        print(f"  GT:        {pred['label_str']}")
        print(f"  Predicted: {pred['predicted_label']}")
        if pred.get("reasoning"):
            print(f"  Reasoning: {pred['reasoning'][:100]}...")
        print()

    # =========================================================================
    # Use Case 5: Find errors
    # =========================================================================
    print("=" * 70)
    print("Error Analysis")
    print("=" * 70)
    errors = [p for p in predictions if p["label_str"] != p["predicted_label"]]
    print(f"Found {len(errors)} errors ({len(errors) / len(predictions):.1%})")

    if errors:
        print("\nFirst few errors:")
        for pred in errors[:5]:
            print(
                f"  {pred['video_path']}: GT={pred['label_str']} → Pred={pred['predicted_label']}"
            )
    print()

    # =========================================================================
    # Use Case 6: Filter predictions by criteria
    # =========================================================================
    print("=" * 70)
    print("Filter by Criteria")
    print("=" * 70)

    # Example: Find all fall predictions
    fall_predictions = [p for p in predictions if p["predicted_label"] in ["fall", "fallen"]]
    print(f"Fall/Fallen predictions: {len(fall_predictions)}")

    # Example: Find predictions from specific video
    if predictions:
        example_video = predictions[0]["video_path"]
        same_video = [p for p in predictions if p["video_path"] == example_video]
        print(f"Predictions from '{example_video}': {len(same_video)}")

    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print("\nExample:")
        print("  python scripts/load_predictions_example.py outputs/predictions/<run_name>.jsonl")
        sys.exit(1)

    predictions_file = sys.argv[1]
    if not Path(predictions_file).exists():
        print(f"Error: File not found: {predictions_file}")
        sys.exit(1)

    main(predictions_file)
