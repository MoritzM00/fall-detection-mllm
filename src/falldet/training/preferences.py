"""Deterministic negative-label selection for preference training."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from falldet.data.video_dataset import idx2label, label2idx


class NegativeSelector(Protocol):
    def select(self, query_index: int, positive_label: str) -> str: ...


def canonicalize_label(value: Any) -> str:
    """Convert repository label metadata to a validated canonical label string."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected one label value, got tensor shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)) and float(value).is_integer():
        label = idx2label.get(int(value))
        if label is None:
            raise ValueError(f"Unknown numeric label: {value}")
        return label
    if isinstance(value, str):
        label = value.strip()
        if label in label2idx:
            return label
    raise ValueError(f"Unknown label: {value!r}")


def label_for_index(dataset: Dataset, index: int) -> str:
    """Read one label from dataset metadata without invoking video decoding."""

    if isinstance(dataset, Subset):
        return label_for_index(dataset.dataset, int(dataset.indices[index]))

    child_datasets = getattr(dataset, "datasets", None)
    cumulative_sizes = getattr(dataset, "cumulative_sizes", None)
    if child_datasets is not None and cumulative_sizes is not None:
        previous = 0
        for child, end in zip(child_datasets, cumulative_sizes, strict=True):
            if index < end:
                return label_for_index(child, index - previous)
            previous = end
        raise IndexError(index)

    segments = getattr(dataset, "video_segments", None)
    if segments is not None:
        segment = segments[index]
        value = segment.get("label_str", segment.get("label"))
        return canonicalize_label(value)

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return canonicalize_label(targets[index])

    raise TypeError(f"Dataset {type(dataset).__name__} exposes no supported label metadata")


def observed_labels(dataset: Dataset) -> tuple[str, ...]:
    """Return the stable, canonical label universe represented by a dataset."""

    length = len(dataset)  # ty: ignore[invalid-argument-type]
    labels = {label_for_index(dataset, index) for index in range(length)}
    return tuple(sorted(labels, key=label2idx.__getitem__))


class RandomNegativeSelector:
    """Select one wrong label deterministically from a fixed label universe."""

    def __init__(self, labels: tuple[str, ...] | list[str], seed: int):
        if seed < 0:
            raise ValueError("Preference seed must be non-negative")
        canonical = {canonicalize_label(label) for label in labels}
        self.labels = tuple(sorted(canonical, key=label2idx.__getitem__))
        if len(self.labels) < 2:
            raise ValueError("DPO requires at least two observed training classes")
        self.seed = seed

    def select(self, query_index: int, positive_label: str) -> str:
        positive = canonicalize_label(positive_label)
        if positive not in self.labels:
            raise ValueError(
                f"Positive label {positive!r} is absent from the training label universe"
            )
        if query_index < 0:
            raise ValueError("Query index must be non-negative")

        candidates = tuple(label for label in self.labels if label != positive)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, query_index]))
        return candidates[int(rng.integers(len(candidates)))]
