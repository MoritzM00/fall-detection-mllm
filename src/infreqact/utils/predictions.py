"""Utilities for loading and working with prediction files."""

import json
from pathlib import Path
from typing import Any


def load_predictions_jsonl(file_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load predictions from JSONL file.

    The JSONL format contains:
    - Line 1: Metadata dict with type="metadata", model, dataset, prompt, prompt_config, timestamp, wandb_run_id
    - Lines 2+: Prediction dicts with type="prediction", idx, and all prediction fields

    Args:
        file_path: Path to JSONL predictions file

    Returns:
        Tuple of (metadata, predictions)
        - metadata: Dict with model, dataset, prompt, prompt_config, timestamp, wandb_run_id
        - predictions: List of prediction dicts (each contains idx, video_path, label_str, predicted_label, etc.)

    Example:
        >>> metadata, predictions = load_predictions_jsonl("predictions.jsonl")
        >>> print(f"Model: {metadata['model']}")
        >>> print(f"Prompt: {metadata['prompt']}")
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
