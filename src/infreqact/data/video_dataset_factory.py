"""Video dataset factory for creating and configuring video datasets.

This module provides factory functions for creating video datasets from Hydra
configurations, handling multiple dataset types, splits, and evaluation groups.
"""

import logging
from typing import Any

from omegaconf import DictConfig

from .multi_video_dataset import MultiVideoDataset
from .video_dataset import OmnifallVideoDataset
from .wanfall_video_dataset import WanfallVideoDataset

logger = logging.getLogger(__name__)


def get_video_datasets(
    cfg: DictConfig,
    mode: str,
    run: Any | None = None,
    return_individual: bool = False,
    split="cs",
    size: tuple[int, int] | None = None,
) -> MultiVideoDataset | dict[str, Any]:
    """
    Create and return video datasets based on configuration.

    This is a TRUE FACTORY - creates different dataset types (Omnifall/WanFall) based on config.

    CRITICAL PRESERVED LOGIC:
    - Split suffix handling (CRITICAL #6): Always include split suffix in dataset key
    - Dataset config selection (CRITICAL #8): Different configs for train/val/test
    - Evaluation group handling (CRITICAL #9): Only create groups if explicitly specified

    Args:
        cfg: Hydra configuration
        mode: Dataset mode ('train', 'val', 'test')
        run: WandB run for logging
        return_individual: Whether to return individual datasets as dict (for evaluation)
        split: Split type ('cs' for cross-subject, 'cv' for cross-view)
        size: Optional tuple specifying the (height, width) to resize frames to

    Returns:
        If return_individual is False:
            MultiVideoDataset instance
        If return_individual is True:
            Dict with:
                '{group}_combined': MultiVideoDataset instances (one per evaluation group)
                'all_combined': MultiVideoDataset with all datasets
                'individual': Dict of dataset_name -> OmnifallVideoDataset instances
    """
    datasets = {}
    dataset_groups = {}  # Track which datasets belong to which evaluation group
    logging.info(f"Creating video datasets for mode: {mode}, split: {split}")

    # Select the appropriate dataset configuration based on mode
    # CRITICAL #8: Different dataset configs for train/val/test
    if mode == "train" and hasattr(cfg, "dataset_train"):
        dataset_config = cfg.dataset_train
        logger.info(f"Using specific {mode} dataset configuration")
    elif mode == "val" and hasattr(cfg, "dataset_val"):
        dataset_config = cfg.dataset_val
        logger.info(f"Using specific {mode} dataset configuration")
    elif mode == "test" and hasattr(cfg, "dataset_test"):
        dataset_config = cfg.dataset_test
        logger.info(f"Using specific {mode} dataset configuration")
    else:
        dataset_config = cfg.dataset
        logger.info(f"Using default dataset configuration for {mode}")

    for ds_config in dataset_config.video_datasets:
        # Get per-dataset split with multiple fallbacks:
        # 1. Per-dataset split field (if specified)
        # 2. Extract from split_root (if it contains config=)
        # 3. Global split parameter from dataset config
        # 4. Function parameter split
        dataset_split = ds_config.get("split", None)
        split_root_config = None

        # Extract split from split_root if present (this is the source of truth)
        if hasattr(ds_config, "split_root") and "config=" in ds_config.split_root:
            split_root_config = ds_config.split_root.split("config=")[-1].split("/")[0]

        # If no explicit split field, use split_root or fallback to parameter
        if dataset_split is None:
            if split_root_config is not None:
                dataset_split = split_root_config
                logger.info(f"Using split '{dataset_split}' from split_root for {ds_config.name}")
            else:
                dataset_split = split
                logger.info(
                    f"Using split '{dataset_split}' from function parameter for {ds_config.name}"
                )
        else:
            logger.info(
                f"Using explicit split '{dataset_split}' from dataset config for {ds_config.name}"
            )

        # CRITICAL VALIDATION: Ensure split matches split_root (fail fast if mismatch)
        if split_root_config is not None and dataset_split != split_root_config:
            error_msg = (
                f"\n{'=' * 80}\n"
                f"CRITICAL CONFIGURATION ERROR: Split mismatch for dataset '{ds_config.name}'\n"
                f"{'=' * 80}\n"
                f"  Dataset split config: '{dataset_split}'\n"
                f"  Split from split_root: '{split_root_config}'\n"
                f"  Split root path: {ds_config.split_root}\n"
                f"\n"
                f"These MUST match! The split_root HuggingFace path contains 'config={split_root_config}'\n"
                f"but you specified split='{dataset_split}' in your config.\n"
                f"\n"
                f"Fix this by:\n"
                f"  1. Updating split: '{split_root_config}' in your dataset config, OR\n"
                f"  2. Updating split_root to use 'config={dataset_split}'\n"
                f"{'=' * 80}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Select appropriate dataset class based on dataset name
        # FACTORY LOGIC: Choose WanFall or Omnifall dataset class
        if ds_config.name.lower() == "wanfall":
            DatasetClass = WanfallVideoDataset
        else:
            DatasetClass = OmnifallVideoDataset

        dataset = DatasetClass(
            video_root=ds_config.video_root,
            annotations_file=ds_config.annotations_file,
            target_fps=dataset_config.target_fps,
            vid_frame_count=dataset_config.vid_frame_count,
            split_root=ds_config.split_root,
            dataset_name=ds_config.name,
            mode=mode,
            split=dataset_split
            if DatasetClass == OmnifallVideoDataset
            else None,  # WanFall doesn't use split types
            data_fps=ds_config.get("dataset_fps", None),
            path_format=dataset_config.path_format,
            max_retries=10,
            size=size,
        )

        if len(dataset) > 0:
            # CRITICAL #6: ALWAYS include split suffix in dataset key for consistency
            # Use explicit split parameter from config (now required for all datasets)
            split_suffix = dataset_split if dataset_split else "cs"
            dataset_key = f"{ds_config.name}_{split_suffix}"

            datasets[dataset_key] = dataset
            logger.info(
                f"Added dataset '{dataset_key}' with {len(dataset)} segments "
                f"(mode={mode}, split_type={split_suffix})"
            )

            # Track which evaluation group this dataset belongs to
            # CRITICAL #9: Only add to group if evaluation_group is explicitly specified
            group = ds_config.get("evaluation_group", None)
            if group is not None:
                if group not in dataset_groups:
                    dataset_groups[group] = []
                dataset_groups[group].append(dataset)

            if run is not None and run.id:
                run.log({f"{ds_config.name}_{mode}_size": len(dataset)})
        else:
            logger.warning(f"Dataset {ds_config.name} is empty for {mode} split. Skipping.")
            if run is not None and run.id:
                run.log({f"{ds_config.name}_{mode}_size": 0})

    if not datasets:
        raise ValueError(f"No datasets could be loaded for {mode} split")

    # Convert dict_values to list to make it picklable
    dataset_list = list(datasets.values())
    multi_dataset = MultiVideoDataset(dataset_list)
    logger.info(
        f"Created combined dataset with {len(multi_dataset)} total segments for {mode} split"
    )

    if return_individual:
        # Create combined datasets for each evaluation group
        result = {}
        for group_name, group_datasets in dataset_groups.items():
            combined_dataset = MultiVideoDataset(group_datasets)
            result[f"{group_name}_combined"] = combined_dataset
            logger.info(
                f"Created {group_name}_combined dataset with {len(combined_dataset)} segments"
            )

        # Only create all_combined if explicitly requested via config
        # CRITICAL #9: This prevents random groups from appearing without explicit configuration
        if dataset_config.get("create_all_combined", False):
            result["all_combined"] = multi_dataset
            logger.info(f"Created all_combined dataset with {len(multi_dataset)} segments")

        # Add individual datasets
        result["individual"] = {k: v for k, v in datasets.items()}

        return result
    else:
        return multi_dataset
