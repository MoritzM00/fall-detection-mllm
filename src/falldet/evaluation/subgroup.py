import logging
from typing import Any

import numpy as np

import wandb
from falldet.data.dataset import GenericVideoDataset
from falldet.metrics.subgroup import compute_subgroup_metrics, print_subgroup_summary

from .visual import visualize_subgroup_results

logger = logging.getLogger(__name__)


def perform_subgroup_evaluation(
    dataset: Any,
    predictions: np.ndarray,
    references: np.ndarray,
    dataset_name: str,
    run: wandb.Run | None = None,
) -> dict[str, Any] | None:
    """
    Perform subgroup evaluation for datasets with demographic metadata.

    Args:
        dataset: The dataset to extract metadata from
        predictions: Array of predicted labels
        references: Array of ground truth labels
        dataset_name: Name of the dataset (for logging)
        run: WandB run for logging

    Returns:
        Dictionary of subgroup metrics, or None if no metadata available
    """
    logger.info(f"Performing subgroup evaluation for {dataset_name}")

    try:
        # Extract metadata directly from the dataset
        # The dataset should be in the same order as predictions/references
        logger.info(f"Extracting metadata from {len(dataset)} samples...")

        metadata = {
            "age_group": [],
            "gender": [],
            "ethnicity": [],
            "bmi_band": [],
        }

        # Check if dataset has video_segments (WanfallVideoDataset)
        if hasattr(dataset, "video_segments"):
            # Direct access from video_segments list (very fast, no video loading)
            logger.info("Using fast metadata extraction via video_segments")
            for segment in dataset.video_segments:
                for key in metadata:
                    value = segment.get(key, None)
                    metadata[key].append(value)
        # Check if dataset has direct metadata access via dataframe
        elif hasattr(dataset, "data") and hasattr(dataset.data, "to_dict"):
            # Direct access from underlying dataframe (also fast)
            logger.info("Using fast metadata extraction via dataframe")
            data_dict = dataset.data.to_dict("list")
            for key in metadata:
                if key in data_dict:
                    metadata[key] = data_dict[key]
                else:
                    metadata[key] = [None] * len(dataset)
        else:
            # Fallback: iterate through dataset (very slow, loads videos!)
            logger.warning("No fast metadata access available, falling back to dataset iteration")
            logger.warning(f"This will load {len(dataset)} videos and may take a VERY long time...")

            for i in range(len(dataset)):
                if i % 100 == 0:
                    logger.info(f"  Processed {i}/{len(dataset)} samples...")
                sample = dataset[i]
                for key in metadata:
                    # Get metadata value from sample, default to None if not present
                    value = sample.get(key, None)
                    metadata[key].append(value)

        logger.info(f"Metadata extraction complete: extracted {len(metadata['age_group'])} samples")

        # Verify we have the right number of samples
        if len(metadata["age_group"]) != len(predictions):
            logger.warning(
                f"Metadata length ({len(metadata['age_group'])}) does not match predictions length ({len(predictions)}). Skipping subgroup evaluation."
            )
            return None

        # Check if metadata is actually available
        if all(all(v is None for v in values) for values in metadata.values()):
            logger.warning(
                f"No metadata available in dataset {dataset_name}. Skipping subgroup evaluation."
            )
            return None

        # Compute subgroup metrics
        subgroup_metrics = compute_subgroup_metrics(
            predictions=predictions,
            references=references,
            metadata=metadata,
            subgroup_keys=["age_group", "ethnicity", "gender", "bmi_band"],
        )

        # Visualize subgroup results with horizontal bar charts
        visualize_subgroup_results(subgroup_metrics, dataset_name=dataset_name)

        # Print summary to console (table format)
        print_subgroup_summary(subgroup_metrics)

        # Log to WandB
        if run:
            for subgroup_key, categories in subgroup_metrics.items():
                for category, metrics in categories.items():
                    for metric_name, metric_value in metrics.items():
                        run.log(
                            {
                                f"{dataset_name}/subgroup/{subgroup_key}/{category}/{metric_name}": metric_value
                            }
                        )

        return subgroup_metrics

    except Exception as e:
        logger.error(f"Error in subgroup evaluation for {dataset_name}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def extract_metadata_from_dataset(
    dataset: GenericVideoDataset,
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
