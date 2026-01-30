"""Few-shot exemplar sampling utilities."""

import logging
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ExemplarSampler:
    """Sample and cache few-shot exemplars from a dataset."""

    def __init__(
        self,
        dataset: Dataset,
        num_shots: int,
        strategy: Literal["random", "balanced"] = "balanced",
        seed: int = 42,
    ):
        """Initialize the sampler.

        Args:
            dataset: Source dataset (typically training set)
            num_shots: Number of exemplars to sample
            strategy: Sampling strategy - "random" or "balanced"
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_shots = num_shots
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        self._cache: list[dict] | None = None
        self._class_to_indices: dict[str, list[int]] | None = None

    def sample(self) -> list[dict]:
        """Sample exemplars (cached after first call).

        Returns:
            List of exemplar dicts with 'video', 'label_str', etc.
        """
        if self._cache is not None:
            return self._cache

        if self.num_shots == 0:
            self._cache = []
            return []

        indices = self._sample_indices()
        logger.info(f"Loading {len(indices)} exemplars...")
        self._cache = [self.dataset[idx] for idx in indices]
        logger.info(f"Cached exemplars: {[ex['label_str'] for ex in self._cache]}")
        return self._cache

    def _build_class_index(self) -> dict[str, list[int]]:
        """Build mapping from class labels to dataset indices."""
        if self._class_to_indices is not None:
            return self._class_to_indices

        class_to_indices: dict[str, list[int]] = {}
        for idx in range(len(self.dataset)):
            segment = self.dataset.video_segments[idx]
            label = segment["label_str"]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        self._class_to_indices = class_to_indices
        logger.info(f"Built class index: {len(class_to_indices)} classes")
        return class_to_indices

    def _sample_indices(self) -> list[int]:
        """Get indices based on strategy."""
        if self.strategy == "balanced":
            return self._sample_balanced_indices()
        elif self.strategy == "random":
            return self._sample_random_indices()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _sample_balanced_indices(self) -> list[int]:
        """Sample roughly equal exemplars per class."""
        class_to_indices = self._build_class_index()
        classes = sorted(class_to_indices.keys())
        per_class = max(1, self.num_shots // len(classes))

        indices: list[int] = []
        for cls in classes:
            available = np.array(class_to_indices[cls])
            n = min(per_class, len(available))
            sampled = self.rng.choice(available, n, replace=False).tolist()
            indices.extend(sampled)

        indices_arr = np.array(indices)
        self.rng.shuffle(indices_arr)
        return indices_arr[: self.num_shots].tolist()

    def _sample_random_indices(self) -> list[int]:
        """Sample randomly from entire dataset."""
        all_indices = np.arange(len(self.dataset))
        return self.rng.choice(
            all_indices, min(self.num_shots, len(all_indices)), replace=False
        ).tolist()
