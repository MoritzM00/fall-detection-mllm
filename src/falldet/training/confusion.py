"""Confusion-guided negative labels mined from a previous model's predictions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from falldet.data.video_dataset import idx2label, label2idx
from falldet.training.preferences import (
    canonicalize_label,
    sample_identity,
    segment_for_index,
)
from falldet.utils.predictions import load_predictions_jsonl

logger = logging.getLogger(__name__)

SOURCE_PREDICTION = "prediction"
SOURCE_CONFUSION = "confusion"
SOURCE_RANDOM = "random"


def load_prediction_records(paths: Sequence[str | Path]) -> tuple[list[dict], list[str]]:
    """Load prediction rows from one or more inference JSONL files.

    Returns the prediction rows and the ``data.mode`` each file was generated on.
    """

    records: list[dict] = []
    modes: list[str] = []
    for path in paths:
        metadata, predictions = load_predictions_jsonl(Path(path).expanduser())
        data_config = metadata.get("config", {}).get("data", {})
        modes.append(str(data_config.get("mode", "unknown")))
        records.extend(predictions)
        logger.info(f"Loaded {len(predictions)} predictions from {path}")
    return records, modes


def _predicted_label(record: dict) -> str | None:
    try:
        return canonicalize_label(record["predicted_label"])
    except (KeyError, ValueError):
        return None


def confusion_counts(records: Sequence[dict]) -> np.ndarray:
    """Count ``[true, predicted]`` pairs over the full label vocabulary."""

    counts = np.zeros((len(label2idx), len(label2idx)), dtype=np.int64)
    skipped = 0
    for record in records:
        predicted = _predicted_label(record)
        if predicted is None:
            skipped += 1
            continue
        true = canonicalize_label(record.get("label_str", record.get("label")))
        counts[label2idx[true], label2idx[predicted]] += 1
    if skipped:
        logger.warning(f"Ignored {skipped} predictions with labels outside the vocabulary")
    return counts


def align_predictions_to_dataset(dataset: Dataset, records: Sequence[dict]) -> list[str | None]:
    """Return the previous model's predicted label per dataset row (None if not predicted)."""

    by_identity: dict[tuple, str | None] = {}
    for record in records:
        identity = sample_identity(record)
        if identity in by_identity:
            raise ValueError(f"Duplicate prediction sample identity: {identity}")
        by_identity[identity] = _predicted_label(record)

    length = len(dataset)  # ty: ignore[invalid-argument-type]
    aligned: list[str | None] = []
    for index in range(length):
        segment, dataset_name = segment_for_index(dataset, index)
        aligned.append(by_identity.get(sample_identity(segment, dataset_name)))
    return aligned


def negative_distribution(counts: np.ndarray, uniform_mix: float) -> np.ndarray:
    """Row-stochastic ``P(negative | positive)`` from off-diagonal confusion counts.

    Each row mixes the model's error distribution with a uniform distribution over
    all wrong labels. Rows without any recorded error fall back to uniform.
    """

    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    num_labels = counts.shape[0]
    off_diagonal = ~np.eye(num_labels, dtype=bool)
    uniform = off_diagonal / (num_labels - 1)

    errors = np.where(off_diagonal, counts, 0).astype(np.float64)
    totals = errors.sum(axis=1, keepdims=True)
    error_rows = np.divide(errors, totals, out=uniform.copy(), where=totals > 0)
    return (1.0 - uniform_mix) * error_rows + uniform_mix * uniform


class ConfusionNegativeSelector:
    """Pick the previous model's own mistake, else sample from ``distribution``.

    ``sampled_source`` names the sampled rows in ``sources`` (confusion or random).
    """

    def __init__(
        self,
        query_labels: Sequence[str],
        distribution: np.ndarray,
        seed: int,
        row_predictions: Sequence[str | None] | None = None,
        sampled_source: str = SOURCE_CONFUSION,
    ):
        if seed < 0:
            raise ValueError("Preference seed must be non-negative")
        if distribution.shape != (len(label2idx), len(label2idx)):
            raise ValueError(f"Distribution must be square over {len(label2idx)} labels")
        if row_predictions is not None and len(row_predictions) != len(query_labels):
            raise ValueError("Row predictions must match query labels")

        self.query_labels = tuple(canonicalize_label(label) for label in query_labels)
        negatives: list[str] = []
        sources: list[str] = []
        for index, positive in enumerate(self.query_labels):
            predicted = row_predictions[index] if row_predictions is not None else None
            if predicted is not None and predicted != positive:
                negatives.append(predicted)
                sources.append(SOURCE_PREDICTION)
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            choice = int(rng.choice(len(label2idx), p=distribution[label2idx[positive]]))
            negatives.append(idx2label[choice])
            sources.append(sampled_source)

        self.negative_labels = tuple(negatives)
        self.sources = tuple(sources)

    def select(self, query_index: int, positive_label: str) -> str:
        if query_index < 0 or query_index >= len(self.query_labels):
            raise IndexError(query_index)
        positive = canonicalize_label(positive_label)
        expected = self.query_labels[query_index]
        if positive != expected:
            raise ValueError(
                f"Query label mismatch at index {query_index}: dataset={positive!r}, selector={expected!r}"
            )
        return self.negative_labels[query_index]

    def prediction_fraction(self) -> float:
        return self.sources.count(SOURCE_PREDICTION) / max(len(self.sources), 1)
