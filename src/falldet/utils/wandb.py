import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from falldet.config import resolve_model_name_from_config
from falldet.data.dataset import GenericVideoDataset
from falldet.data.video_dataset import label2idx

logger = logging.getLogger(__name__)


def initialize_run_from_config(cfg: DictConfig):
    wandb_mode = cfg.get("wandb", {}).get("mode", "online")
    logger.info(f"Initializing W&B in {wandb_mode} mode")
    name, tags = create_name_and_tags_from_config(cfg)
    run = wandb.init(
        project=cfg.wandb.project,
        name=name,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=wandb_mode,
    )
    logger.info(f"W&B run initialized with name: {run.name}, id: {run.id}")
    logger.info(f"Run tags: {list(run.tags)}")
    return run


def create_name_and_tags_from_config(cfg: DictConfig) -> tuple[str, list[str]]:
    """Create a W&B run name and tags based on the configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        tuple: (name, tags) where name is a string or None, and tags is a list of strings.
    """  # Create a descriptive run name
    if cfg.wandb.get("name", None):
        # Use name from config if available
        base_name = cfg.wandb.name
    else:
        # Create name from config parameters
        model_info = resolve_model_name_from_config(cfg.model)

        frame_count = cfg.get("num_frames", None)
        model_fps = cfg.get("model_fps", None)

        dataset_info = f"F{frame_count}@{model_fps}"
        base_name = f"{model_info}-{dataset_info}"

    if wandb.run is not None:
        id = wandb.run.id
    else:
        id = wandb.util.generate_id()

    run_name = f"{base_name}_{id}"

    # tags contain info about the experiment, dataset and model (highlevel)
    tags = cfg.wandb.get("tags", [])
    if tags is None:
        tags = []

    # name is inside cfg.dataset.video_dataset[i].name
    for dataset_cfg in cfg.dataset.get("video_datasets", []):
        dataset_name = dataset_cfg.get("name", None)
        if dataset_name is not None:
            tags.append(dataset_name)

    # Use model family from config
    model_family = cfg.model.get("family", resolve_model_name_from_config(cfg.model).split("-")[0])
    tags.append(model_family)

    if cfg.get("cot", False):
        tags.append("cot")

    tags = [tag.lower() for tag in tags]
    tags = list(set(tags))  # Ensure uniqueness
    return run_name, tags


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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load run config and predictions from W&B.

    Args:
        run_id: W&B run ID
        project: W&B project (defaults to WANDB_PROJECT env var)
        entity: W&B entity (defaults to WANDB_ENTITY env var)

    Returns:
        Tuple of (config_dict, predictions_list)

    Raises:
        ValueError: If entity or project is not provided and env var is not set
        FileNotFoundError: If no JSONL predictions file is found in the run
    """
    # Import here to avoid circular import
    from falldet.utils.predictions import load_predictions_jsonl

    entity = entity or os.getenv("WANDB_ENTITY")
    project = project or os.getenv("WANDB_PROJECT")

    if not entity:
        raise ValueError("Entity not provided and WANDB_ENTITY environment variable not set")
    if not project:
        raise ValueError("Project not provided and WANDB_PROJECT environment variable not set")

    logger.info(f"Loading W&B run {entity}/{project}/{run_id}")

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Get full config
    config = dict(run.config)

    # Find and download JSONL file
    for file in run.files():
        if file.name.endswith(".jsonl"):
            with tempfile.TemporaryDirectory() as temp_dir:
                file.download(root=temp_dir, replace=True)
                _, predictions = load_predictions_jsonl(Path(temp_dir) / file.name)
                return config, predictions

    raise FileNotFoundError(f"No JSONL predictions file found in run {run_id}")
