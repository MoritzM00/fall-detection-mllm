import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import wandb
from falldet.config import resolve_model_name_from_config
from falldet.data.dataset import GenericVideoDataset
from falldet.data.video_dataset import label2idx
from falldet.schemas import InferenceConfig
from falldet.utils.predictions import (
    load_predictions_jsonl,
    prediction_jsonl_path,
    prediction_jsonl_relpath,
)

logger = logging.getLogger(__name__)


def initialize_run_from_config(config: InferenceConfig):
    wandb_mode = config.wandb.mode
    logger.info(f"Initializing W&B in {wandb_mode} mode")
    run_id = wandb.util.generate_id()
    base_name, tags = create_name_and_tags_from_config(config)
    run_name = f"{base_name}_{run_id}"
    run = wandb.init(
        project=config.wandb.project,
        id=run_id,
        name=run_name,
        tags=tags,
        config=config.model_dump(),
        mode=wandb_mode,
    )
    logger.info(f"W&B run initialized with name: {run.name}, id: {run.id}")
    logger.info(f"Run tags: {list(run.tags)}")
    return run


def create_name_and_tags_from_config(config: InferenceConfig) -> tuple[str, list[str]]:
    """Create a W&B base run name and tags based on the configuration.

    Args:
        config: Validated inference configuration.

    Returns:
        tuple: (base_name, tags) where base_name is the descriptive name prefix and tags is a
        list of strings.
    """
    if config.wandb.name:
        base_name = config.wandb.name
    else:
        model_info = resolve_model_name_from_config(config.model)

        frame_count = config.num_frames
        model_fps = config.model_fps

        dataset_info = f"F{frame_count}@{model_fps}"
        base_name = f"{model_info}-{dataset_info}"

    tags = list(config.wandb.tags) if config.wandb.tags else []

    for dataset_item in config.dataset.video_datasets:
        tags.append(dataset_item.name)

    model_family = config.model.family
    tags.append(model_family)

    if config.prompt.cot:
        tags.append("cot")

    tags = [tag.lower() for tag in tags]
    tags = list(set(tags))  # Ensure uniqueness
    return base_name, tags


def get_prediction_output_path(output_root: str | Path, project: str, run_id: str) -> Path:
    """Return the canonical local path for predictions belonging to a W&B run."""
    return prediction_jsonl_path(output_root, project, run_id)


def _download_predictions_file(
    run: Any,
    *,
    destination_path: Path,
    project: str,
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Download a run's canonical predictions JSONL and persist it locally."""
    expected_filename = destination_path.name
    expected_relpath = prediction_jsonl_relpath(project, run_id).as_posix()
    jsonl_files = [file for file in run.files() if file.name.endswith(".jsonl")]

    prioritized_files = [
        file
        for file in jsonl_files
        if file.name.endswith(expected_relpath) or file.name.endswith(expected_filename)
    ]
    fallback_files = [file for file in jsonl_files if file not in prioritized_files]

    for file in prioritized_files + fallback_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            file.download(root=temp_dir, replace=True)
            downloaded_path = Path(temp_dir) / file.name
            metadata, predictions = load_predictions_jsonl(downloaded_path)
            metadata_run_id = metadata.get("wandb_run_id")
            if metadata_run_id and metadata_run_id != run_id:
                continue

            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(downloaded_path.read_bytes())
            logger.info(f"Saved predictions to {destination_path}")
            return metadata, predictions

    raise FileNotFoundError(f"No prediction JSONL for run {run_id} found in W&B files")


def log_videos_with_predictions(
    dataset: GenericVideoDataset | torch.utils.data.Subset,
    predictions: list[str] | list[int] | np.ndarray,
    references: list[str] | list[int] | np.ndarray,
    dataset_name: str,
    n_videos: int = 5,
) -> None:
    # Get target_fps from dataset or underlying dataset if it's a Subset
    target_fps = (
        dataset.target_fps if hasattr(dataset, "target_fps") else dataset.dataset.target_fps
    )

    for idx in range(min(n_videos, len(predictions))):
        sample = dataset[idx]
        video = sample["video"].numpy()

        caption = f"Predicted: {predictions[idx]}, True: {references[idx]}"
        wandb.log(
            {
                f"{dataset_name}_{sample['video_path']}": wandb.Video(
                    video,
                    caption=caption,
                    format="mp4",
                    fps=target_fps,
                )
            }
        )


def log_confusion_matrix(
    predictions: list[str],
    references: list[str],
    dataset_name: str,
) -> None:
    """Log confusion matrix to wandb.

    Args:
        predictions: Predicted class labels (as strings)
        references: Ground truth class labels (as strings)
        dataset_name: Name of the dataset for logging

    Note:
        This function creates a contiguous index mapping for present classes only.
        wandb.plot.confusion_matrix expects contiguous integer indices (0, 1, 2, ...)
        even if the dataset only contains a subset of all possible classes.
    """
    # Collect unique labels from both predictions and references
    unique_labels = sorted(
        set(predictions) | set(references), key=lambda x: label2idx.get(x, float("inf"))
    )

    # Create contiguous mapping for present classes
    local_label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Convert string labels to contiguous integer indices
    predictions_idx = [local_label_to_idx[p] for p in predictions]
    references_idx = [local_label_to_idx[r] for r in references]

    # Class names in the same order as indices
    class_names = unique_labels

    wandb.log(
        {
            f"{dataset_name}_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=references_idx,
                preds=predictions_idx,
                class_names=class_names,
            )
        }
    )


def load_run_from_wandb(
    run_id: str,
    project: str | None = None,
    entity: str | None = None,
    output_root: Path | str = "outputs",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load run config and predictions from W&B via the canonical local prediction path.

    Predictions are stored locally under
    ``<output_root>/predictions/<project>/<run_id>.jsonl``. If the file already
    exists, it is loaded directly. Otherwise it is downloaded from W&B and saved
    to that canonical location before loading.

    Args:
        run_id: W&B run ID.
        project: W&B project (defaults to ``WANDB_PROJECT`` env var).
        entity: W&B entity (defaults to ``WANDB_ENTITY`` env var).
        output_root: Root output directory containing the ``predictions/`` subtree.

    Returns:
        Tuple of (config_dict, predictions_list).

    Raises:
        ValueError: If entity or project is not provided and env var is not set.
        FileNotFoundError: If no JSONL predictions file is found in the run.
    """
    project = project or os.getenv("WANDB_PROJECT")
    if not project:
        raise ValueError("Project not provided and WANDB_PROJECT environment variable not set")

    local_path = get_prediction_output_path(output_root, project, run_id)
    if local_path.exists():
        logger.info(f"Loading local predictions from {local_path}")
        metadata, predictions = load_predictions_jsonl(local_path)
        metadata_run_id = metadata.get("wandb_run_id")
        if metadata_run_id and metadata_run_id != run_id:
            raise ValueError(
                f"Local predictions belong to run {metadata_run_id}, expected {run_id}"
            )
        return metadata.get("config", {}), predictions

    entity = entity or os.getenv("WANDB_ENTITY")
    if not entity:
        raise ValueError("Entity not provided and WANDB_ENTITY environment variable not set")

    logger.info(f"Loading W&B run {entity}/{project}/{run_id}")

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    metadata, predictions = _download_predictions_file(
        run,
        destination_path=local_path,
        project=project,
        run_id=run_id,
    )
    return metadata.get("config", {}), predictions
