"""Negative-label selection for preference training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
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


def _segment_for_index(dataset: Dataset, index: int) -> tuple[dict, str | None]:
    """Return segment metadata and its dataset name without decoding video."""

    if isinstance(dataset, Subset):
        return _segment_for_index(dataset.dataset, int(dataset.indices[index]))

    child_datasets = getattr(dataset, "datasets", None)
    cumulative_sizes = getattr(dataset, "cumulative_sizes", None)
    if child_datasets is not None and cumulative_sizes is not None:
        previous = 0
        for child, end in zip(child_datasets, cumulative_sizes, strict=True):
            if index < end:
                return _segment_for_index(child, index - previous)
            previous = end
        raise IndexError(index)

    segments = getattr(dataset, "video_segments", None)
    if segments is None:
        raise TypeError(
            f"Dataset {type(dataset).__name__} exposes no segment metadata for embedding alignment"
        )
    return segments[index], getattr(dataset, "dataset_name", None)


def _sample_identity(sample: dict, dataset_name: str | None = None) -> tuple:
    """Build the stable identity shared by video rows and embedding manifests."""

    try:
        video_path = str(sample["video_path"])
        start_value = sample["start_time"] if "start_time" in sample else sample["start"]
        end_value = sample["end_time"] if "end_time" in sample else sample["end"]
        start = float(start_value)
        end = float(end_value)
        label = canonicalize_label(sample.get("label_str", sample.get("label")))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Incomplete embedding sample identity: {sample!r}") from error
    source = sample.get("dataset", dataset_name)
    if source is None:
        raise ValueError(f"Embedding sample has no dataset identity: {sample!r}")
    return (str(source).casefold(), video_path, round(start, 6), round(end, 6), label)


def align_embeddings_to_dataset(
    dataset: Dataset,
    embeddings: torch.Tensor,
    samples: Sequence[dict],
) -> torch.Tensor:
    """Reorder an embedding artifact to match a dataset or Subset exactly."""

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must have shape [rows, dim], got {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(samples):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match manifest rows ({len(samples)})"
        )

    artifact_indices: dict[tuple, int] = {}
    for index, sample in enumerate(samples):
        identity = _sample_identity(sample)
        if identity in artifact_indices:
            raise ValueError(f"Duplicate embedding sample identity: {identity}")
        artifact_indices[identity] = index

    row_indices = []
    length = len(dataset)  # ty: ignore[invalid-argument-type]
    for index in range(length):
        segment, dataset_name = _segment_for_index(dataset, index)
        identity = _sample_identity(segment, dataset_name)
        try:
            row_indices.append(artifact_indices[identity])
        except KeyError as error:
            raise ValueError(
                f"Dataset row {index} is absent from the embedding manifest: {identity}"
            ) from error

    return embeddings.index_select(0, torch.tensor(row_indices, dtype=torch.long)).contiguous()


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


class EmbeddingSimilarityNegativeSelector:
    """Choose the label of the most similar wrong-class corpus example."""

    def __init__(
        self,
        query_embeddings: torch.Tensor,
        corpus_embeddings: torch.Tensor,
        query_labels: Sequence[str],
        corpus_labels: Sequence[str],
        chunk_size: int = 256,
    ):
        if query_embeddings.ndim != 2 or corpus_embeddings.ndim != 2:
            raise ValueError("Query and corpus embeddings must both have shape [rows, dim]")
        if query_embeddings.shape[1] != corpus_embeddings.shape[1]:
            raise ValueError("Query and corpus embedding dimensions must match")
        if query_embeddings.shape[0] != len(query_labels):
            raise ValueError("Query embedding rows must match query labels")
        if corpus_embeddings.shape[0] != len(corpus_labels):
            raise ValueError("Corpus embedding rows must match corpus labels")
        if chunk_size <= 0:
            raise ValueError("Similarity chunk_size must be positive")
        if corpus_embeddings.shape[0] == 0:
            raise ValueError("Similarity corpus cannot be empty")

        self.query_labels = tuple(canonicalize_label(label) for label in query_labels)
        self.corpus_labels = tuple(canonicalize_label(label) for label in corpus_labels)
        if len(set(self.corpus_labels)) < 2:
            raise ValueError("Similarity mining requires at least two corpus classes")

        queries = query_embeddings.detach().cpu().float()
        corpus = corpus_embeddings.detach().cpu().float()
        if not torch.isfinite(queries).all() or not torch.isfinite(corpus).all():
            raise ValueError("Embeddings must contain only finite values")
        if (queries.norm(dim=1) == 0).any() or (corpus.norm(dim=1) == 0).any():
            raise ValueError("Cosine similarity is undefined for zero-norm embeddings")

        queries = F.normalize(queries, dim=1)
        corpus = F.normalize(corpus, dim=1)
        corpus_label_ids = torch.tensor([label2idx[label] for label in self.corpus_labels])
        selected_indices: list[int] = []
        selected_scores: list[float] = []

        with torch.no_grad():
            for start in range(0, len(queries), chunk_size):
                stop = min(start + chunk_size, len(queries))
                scores = queries[start:stop] @ corpus.T
                query_label_ids = torch.tensor(
                    [label2idx[label] for label in self.query_labels[start:stop]]
                )
                scores.masked_fill_(
                    query_label_ids[:, None] == corpus_label_ids[None, :], -torch.inf
                )
                best_scores, best_indices = scores.max(dim=1)
                if not torch.isfinite(best_scores).all():
                    raise ValueError("At least one query has no wrong-class corpus example")
                selected_indices.extend(int(index) for index in best_indices)
                selected_scores.extend(float(score) for score in best_scores)

        self.corpus_indices = tuple(selected_indices)
        self.scores = tuple(selected_scores)
        self.negative_labels = tuple(self.corpus_labels[index] for index in self.corpus_indices)

    def select(self, query_index: int, positive_label: str) -> str:
        if query_index < 0 or query_index >= len(self.query_labels):
            raise IndexError(query_index)
        positive = canonicalize_label(positive_label)
        expected = self.query_labels[query_index]
        if positive != expected:
            raise ValueError(
                f"Query label mismatch at index {query_index}: dataset={positive!r}, embeddings={expected!r}"
            )
        return self.negative_labels[query_index]
