"""Utilities for loading and working with prediction files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_predictions_jsonl(file_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load predictions from JSONL file.

    The JSONL format contains:
    - Line 1: Metadata dict with type="metadata", model, dataset, config, timestamp, wandb_run_id
    - Lines 2+: Prediction dicts with type="prediction", idx, and all prediction fields

    Args:
        file_path: Path to JSONL predictions file

    Returns:
        Tuple of (metadata, predictions)
        - metadata: Dict with model, dataset, config, timestamp, wandb_run_id
        - predictions: List of prediction dicts (each contains idx, video_path, label_str, predicted_label, etc.)

    Example:
        >>> metadata, predictions = load_predictions_jsonl("predictions.jsonl")
        >>> ground_truths = [p['label_str'] for p in predictions]
        >>> predicted_labels = [p['predicted_label'] for p in predictions]
    """
    file_path = Path(file_path)

    with open(file_path) as f:
        # First line is metadata
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Empty file: {file_path}")

        metadata = json.loads(first_line)
        if metadata.get("type") != "metadata":
            raise ValueError(f"First line must be metadata, got type={metadata.get('type')}")

        # Remaining lines are predictions
        predictions = []
        for line_num, line in enumerate(f, start=2):
            if not line.strip():
                continue

            try:
                pred = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

            if pred.get("type") != "prediction":
                raise ValueError(
                    f"Line {line_num} expected type=prediction, got {pred.get('type')}"
                )

            predictions.append(pred)

    return metadata, predictions


def extract_labels_for_metrics(
    predictions: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Extract ground truth and predicted labels from predictions.

    Args:
        predictions: List of prediction dicts from load_predictions_jsonl

    Returns:
        Tuple of (ground_truth_labels, predicted_labels)
    """
    ground_truths = [p["label_str"] for p in predictions]
    predicted_labels = [p["predicted_label"] for p in predictions]
    return ground_truths, predicted_labels


def save_predictions_jsonl(
    output_path: Path,
    config: dict,
    predictions: list[dict],
    wandb_run_id: str | None = None,
) -> None:
    """Save predictions in JSONL format with metadata.

    Args:
        output_path: Path to output JSONL file
        model_name: Model name
        dataset_name: Dataset name
        config: Configuration dictionary
        predictions: List of prediction dicts
        wandb_run_id: Optional W&B run ID for linking
    """
    with open(output_path, "w") as f:
        # Write metadata line first
        metadata = {
            "type": "metadata",
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "wandb_run_id": wandb_run_id,
        }
        f.write(json.dumps(metadata) + "\n")

        # Write each prediction
        for idx, pred in enumerate(predictions):
            pred_copy = pred.copy()
            pred_copy["type"] = "prediction"
            pred_copy["idx"] = idx
            f.write(json.dumps(pred_copy) + "\n")
