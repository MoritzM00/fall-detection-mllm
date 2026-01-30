import bisect
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiVideoDataset(Dataset):
    """
    Wrapper for multiple OmnifallVideoDataset instances.
    Enables training on multiple video datasets simultaneously with proper indexing.
    """

    def __init__(self, datasets: list[Dataset]):
        """
        Initialize multi-dataset wrapper.

        Args:
            datasets: List of OmnifallVideoDataset instances
        """
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = np.cumsum(self.dataset_sizes).tolist()

        # Log dataset information
        total_segments = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
        logging.info(
            f"MultiVideoDataset initialized with {len(datasets)} datasets, {total_segments} total segments"
        )
        for i, dataset in enumerate(datasets):
            logging.info(f"  Dataset {i} ({dataset.dataset_name}): {len(dataset)} segments")

    @property
    def targets(self):
        """Return all class labels across all datasets."""
        return torch.cat([dataset.targets for dataset in self.datasets])

    @property
    def domain_ids(self):
        """Return domain ID for each segment (dataset index)."""
        domain_ids = []
        for i, dataset in enumerate(self.datasets):
            domain_ids.extend([i] * len(dataset))
        return torch.tensor(domain_ids)

    @property
    def dataset_names(self):
        """Return list of dataset names."""
        return [dataset.dataset_name for dataset in self.datasets]

    def __len__(self):
        """Total number of segments across all datasets."""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get item from the appropriate dataset.

        Args:
            idx: Global index across all datasets

        Returns:
            Dictionary with video data and metadata
        """
        # Ensure idx is a plain Python int to avoid numpy type issues
        idx = int(idx)

        # Find which dataset this index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        # Calculate the local index within that dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Get the sample from the appropriate dataset
        sample = self.datasets[dataset_idx][sample_idx]

        # Add domain information
        sample["domain_id"] = dataset_idx
        sample["domain_name"] = self.datasets[dataset_idx].dataset_name

        return sample

    def get_dataset_by_name(self, name: str) -> Dataset | None:
        """
        Get a dataset by its name.

        Args:
            name: Dataset name to search for

        Returns:
            Dataset instance if found, None otherwise
        """
        for dataset in self.datasets:
            if hasattr(dataset, "dataset_name") and dataset.dataset_name == name:
                return dataset
        return None

    def get_dataset_statistics(self) -> dict[str, dict[str, int]]:
        """
        Get statistics for each dataset.

        Returns:
            Dictionary mapping dataset names to their statistics
        """
        stats = {}
        for dataset in self.datasets:
            if hasattr(dataset, "dataset_name"):
                name = dataset.dataset_name
                stats[name] = {
                    "total_segments": len(dataset),
                    "total_videos": len(dataset.samples) if hasattr(dataset, "samples") else 0,
                }

                # Add class distribution if available
                if hasattr(dataset, "targets"):
                    targets = dataset.targets
                    unique_classes, counts = torch.unique(targets, return_counts=True)
                    stats[name]["class_distribution"] = {
                        int(cls): int(count) for cls, count in zip(unique_classes, counts)
                    }

        return stats

    def __repr__(self):
        """String representation of the multi-dataset."""
        dataset_info = ", ".join(
            [
                f"{dataset.dataset_name}({len(dataset)} segments)"
                for dataset in self.datasets
                if hasattr(dataset, "dataset_name")
            ]
        )
        total = len(self)
        return f"MultiVideoDataset({len(self.datasets)} datasets: {dataset_info}, total={total} segments)"
