#!/usr/bin/env python
"""Populate the disk tensor cache by iterating a dataset.

Runs preprocessing (decode → transform) for every item and writes the result
to the disk cache.  Subsequent inference runs that specify the same cache_dir
will skip PyAV entirely for cached items.

Usage:
    python scripts/build_tensor_cache.py \\
        dataset=omnifall/video/oops \\
        data.cache_dir=outputs/tensor_cache \\
        data.mode=test
"""

import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from falldet.data.video_dataset_factory import get_video_datasets
from falldet.schemas import from_dictconfig
from falldet.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="inference_config")
def main(cfg: DictConfig) -> None:
    setup_logging(log_file="build_tensor_cache.log", console_level=logging.INFO)
    config = from_dictconfig(cfg)

    if config.data.cache_dir is None:
        raise ValueError(
            "data.cache_dir must be set. "
            "Pass e.g. data.cache_dir=outputs/tensor_cache on the command line."
        )

    mode = config.data.mode
    logger.info(f"Building tensor cache: mode={mode}, cache_dir={config.data.cache_dir}")

    dataset = get_video_datasets(
        config=config,
        mode=mode,
        size=config.data.size,
        max_size=config.data.max_size,
        seed=config.data.seed,
    )

    logger.info(f"Dataset: {len(dataset)} items")

    errors = 0
    for i in tqdm(range(len(dataset)), desc="Caching"):
        try:
            dataset[i]
        except Exception as e:
            logger.warning(f"Skipping item {i}: {e}")
            errors += 1

    logger.info(f"Cache build complete. {len(dataset) - errors}/{len(dataset)} items cached.")
    if errors:
        logger.warning(f"{errors} items failed and were skipped.")


if __name__ == "__main__":
    main()
