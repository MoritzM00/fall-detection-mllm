"""Evaluation utilities for video model evaluation (vLLM compatible).

This module provides evaluation functions that work with vLLM inference outputs,
without requiring HuggingFace Trainer or Accelerator dependencies.
"""

import json
import logging
import os
import time
from typing import Any

import numpy as np

from infreqact.metrics.base import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_predictions(
    predictions: list[str] | list[int] | np.ndarray,
    references: list[str] | list[int] | np.ndarray,
    dataset_name: str = "test",
    output_dir: str | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Evaluate predictions against ground truth labels.

    Args:
        predictions: Predicted labels (strings or indices)
        references: Ground truth labels (strings or indices)
        dataset_name: Name of the dataset for logging
        output_dir: Directory to save results (default: "outputs")
        save_results: Whether to save results to JSON file

    Returns:
        Dictionary of computed metrics
    """
    logger.info(f"Evaluating {len(predictions)} predictions on {dataset_name}")

    # Compute comprehensive metrics
    metrics = compute_metrics(y_pred=predictions, y_true=references)

    # Save results if requested
    if save_results:
        if output_dir is None:
            output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        metrics_file = os.path.join(
            output_dir, f"{dataset_name}_metrics_{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")

    return metrics


def save_predictions(
    predictions: list[Any],
    output_file: str,
    references: list[Any] | None = None,
    additional_data: dict[str, Any] | None = None,
) -> None:
    """
    Save predictions to a JSON file.

    Args:
        predictions: List of predictions
        output_file: Path to output JSON file
        references: Optional list of ground truth labels
        additional_data: Optional additional data to save
    """
    data = {
        "predictions": predictions,
    }

    if references is not None:
        data["references"] = references

    if additional_data is not None:
        data.update(additional_data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Saved predictions to {output_file}")
