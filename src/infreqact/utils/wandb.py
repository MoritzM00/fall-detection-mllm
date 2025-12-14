import logging

from omegaconf import DictConfig, OmegaConf

import wandb

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
        model_info = f"{cfg.model.name}"

        frame_count = cfg.get("num_frames", None)
        model_fps = cfg.get("model_fps", None)

        dataset_info = f"F{frame_count}@{model_fps}"
        base_name = f"{model_info}-{dataset_info}"

    # Add unique ID to prevent collisions
    run_name = f"{base_name} {wandb.util.generate_id()}"

    # tags contain info about the experiment, dataset and model (highlevel)
    tags = cfg.wandb.get("tags", [])
    if tags is None:
        tags = []
    tags.append(cfg.dataset.get("name", "unknown_dataset"))
    # Use model family from config
    model_family = cfg.model.get("family", cfg.model.name.split("-")[0])
    tags.append(model_family)

    if cfg.get("cot", False):
        tags.append("cot")

    tags = [tag.lower() for tag in tags]
    tags = list(set(tags))  # Ensure uniqueness
    return run_name, tags
