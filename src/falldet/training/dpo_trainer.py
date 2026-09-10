"""Minimal TRL trainer adaptation for lazy video preference datasets."""

from typing import Any

from datasets import Dataset, IterableDataset
from trl import DPOTrainer


class VideoDPOTrainer(DPOTrainer):
    """Keep the lazy dataset intact; the custom collator performs processing."""

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: Any,
        args: Any,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        del processing_class, args, dataset_name
        return dataset
