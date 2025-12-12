import json
import os
import time
from typing import Any

import numpy as np

from .base import evaluate_predictions, logger


def evaluate_with_subgroups(
    predictions: list[str] | list[int] | np.ndarray,
    references: list[str] | list[int] | np.ndarray,
    metadata: dict[str, list],
    dataset_name: str = "test",
    subgroup_keys: list[str] | None = None,
    output_dir: str | None = None,
    save_results: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluate predictions with subgroup analysis.

    Args:
        predictions: Predicted labels (strings or indices)
        references: Ground truth labels (strings or indices)
        metadata: Dictionary of metadata for subgroup analysis
                  e.g., {"age_group": [...], "gender": [...], ...}
        dataset_name: Name of the dataset for logging
        subgroup_keys: List of metadata keys to analyze (default: all available)
        output_dir: Directory to save results (default: "outputs")
        save_results: Whether to save results to JSON files

    Returns:
        Tuple of (overall_metrics, subgroup_metrics)
    """
    # First, compute overall metrics
    overall_metrics = evaluate_predictions(
        predictions=predictions,
        references=references,
        dataset_name=dataset_name,
        output_dir=output_dir,
        save_results=save_results,
    )

    # Determine which subgroups to evaluate
    if subgroup_keys is None:
        subgroup_keys = list(metadata.keys())

    # Import subgroup evaluation utilities if available
    try:
        from fall_da.utils.subgroup_evaluation import (
            compute_subgroup_metrics,
            print_subgroup_summary,
        )

        logger.info(f"Computing subgroup metrics for: {subgroup_keys}")

        # Convert predictions and references to numpy arrays
        pred_array = (
            np.array(predictions) if not isinstance(predictions, np.ndarray) else predictions
        )
        ref_array = np.array(references) if not isinstance(references, np.ndarray) else references

        # Compute subgroup metrics
        subgroup_metrics = compute_subgroup_metrics(
            predictions=pred_array,
            references=ref_array,
            metadata=metadata,
            subgroup_keys=subgroup_keys,
        )

        # Print subgroup summary
        print_subgroup_summary(subgroup_metrics)

        # Save subgroup results
        if save_results and subgroup_metrics:
            if output_dir is None:
                output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)

            subgroup_file = os.path.join(
                output_dir, f"{dataset_name}_subgroup_metrics_{time.strftime('%Y%m%d-%H%M%S')}.json"
            )
            with open(subgroup_file, "w") as f:
                json.dump(subgroup_metrics, f, indent=4)
            logger.info(f"Saved subgroup metrics to {subgroup_file}")

        return overall_metrics, subgroup_metrics

    except ImportError:
        logger.warning("Subgroup evaluation utilities not available. Skipping subgroup analysis.")
        return overall_metrics, {}


def extract_metadata_from_dataset(
    dataset: Any,
    metadata_keys: list[str] | None = None,
) -> dict[str, list] | None:
    """
    Extract metadata from a dataset for subgroup evaluation.

    Args:
        dataset: Dataset object to extract metadata from
        metadata_keys: List of metadata keys to extract
                      (default: ["age_group", "gender", "ethnicity", "bmi_band"])

    Returns:
        Dictionary mapping metadata keys to lists of values, or None if unavailable
    """
    if metadata_keys is None:
        metadata_keys = ["age_group", "gender", "ethnicity", "bmi_band"]

    metadata = {key: [] for key in metadata_keys}

    try:
        # Check if dataset has video_segments (e.g., WanfallVideoDataset)
        if hasattr(dataset, "video_segments"):
            logger.info("Extracting metadata via video_segments")
            for segment in dataset.video_segments:
                for key in metadata_keys:
                    value = segment.get(key, None)
                    metadata[key].append(value)

        # Check if dataset has direct metadata access via dataframe
        elif hasattr(dataset, "data") and hasattr(dataset.data, "to_dict"):
            logger.info("Extracting metadata via dataframe")
            data_dict = dataset.data.to_dict("list")
            for key in metadata_keys:
                if key in data_dict:
                    metadata[key] = data_dict[key]
                else:
                    metadata[key] = [None] * len(dataset)

        # Fallback: iterate through dataset (slow)
        else:
            logger.warning("No fast metadata access available, falling back to dataset iteration")
            for i in range(len(dataset)):
                if i % 100 == 0:
                    logger.info(f"  Processed {i}/{len(dataset)} samples...")
                sample = dataset[i]
                for key in metadata_keys:
                    value = sample.get(key, None)
                    metadata[key].append(value)

        # Check if metadata is actually available
        if all(all(v is None for v in values) for values in metadata.values()):
            logger.warning("No metadata available in dataset")
            return None

        logger.info(
            f"Successfully extracted metadata for {len(metadata[metadata_keys[0]])} samples"
        )
        return metadata

    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None
