"""Build train/validation negative selectors from the DPO preference config."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import Dataset

from falldet.data.video_dataset import label2idx
from falldet.embeddings import load_embeddings
from falldet.schemas import DPOTrainingConfig
from falldet.training.confusion import (
    SOURCE_CONFUSION,
    SOURCE_RANDOM,
    ConfusionNegativeSelector,
    align_predictions_to_dataset,
    confusion_counts,
    load_prediction_records,
    negative_distribution,
)
from falldet.training.preferences import (
    EmbeddingSimilarityNegativeSelector,
    NegativeSelector,
    RandomNegativeSelector,
    align_embeddings_to_dataset,
    label_for_index,
)

logger = logging.getLogger(__name__)

ManifestValue = str | int | float | None


@dataclass
class PreferenceSetup:
    train: NegativeSelector
    validation: NegativeSelector
    summary: dict
    manifest: list[dict[str, ManifestValue]] = field(default_factory=list)


def _labels_for_rows(dataset: Dataset) -> tuple[str, ...]:
    length = len(dataset)  # ty: ignore[invalid-argument-type]
    return tuple(label_for_index(dataset, index) for index in range(length))


def _build_similarity(
    config: DPOTrainingConfig, train_dataset: Dataset, validation_dataset: Dataset
) -> PreferenceSetup:
    assert config.preference.train_embeddings_path is not None
    assert config.preference.validation_embeddings_path is not None
    train_path = Path(config.preference.train_embeddings_path).expanduser().resolve()
    validation_path = Path(config.preference.validation_embeddings_path).expanduser().resolve()
    train_embeddings, train_samples = load_embeddings(train_path)
    validation_embeddings, validation_samples = load_embeddings(validation_path)
    train_embeddings = align_embeddings_to_dataset(train_dataset, train_embeddings, train_samples)
    validation_embeddings = align_embeddings_to_dataset(
        validation_dataset, validation_embeddings, validation_samples
    )
    train_labels = _labels_for_rows(train_dataset)

    train_selector = EmbeddingSimilarityNegativeSelector(
        query_embeddings=train_embeddings,
        corpus_embeddings=train_embeddings,
        query_labels=train_labels,
        corpus_labels=train_labels,
        chunk_size=config.preference.chunk_size,
    )
    validation_selector = EmbeddingSimilarityNegativeSelector(
        query_embeddings=validation_embeddings,
        corpus_embeddings=train_embeddings,
        query_labels=_labels_for_rows(validation_dataset),
        corpus_labels=train_labels,
        chunk_size=config.preference.chunk_size,
    )

    summary: dict = {
        "preference_strategy": "similarity",
        "train_embeddings_path": str(train_path),
        "validation_embeddings_path": str(validation_path),
    }
    manifest: list[dict[str, ManifestValue]] = []
    for split, selector in (("train", train_selector), ("validation", validation_selector)):
        scores = selector.scores
        summary[f"{split}_negative_similarity_mean"] = sum(scores) / len(scores)
        summary[f"{split}_negative_similarity_min"] = min(scores)
        summary[f"{split}_negative_similarity_max"] = max(scores)
        for query_index, (positive, negative, corpus_index, score) in enumerate(
            zip(
                selector.query_labels,
                selector.negative_labels,
                selector.corpus_indices,
                selector.scores,
                strict=True,
            )
        ):
            manifest.append(
                {
                    "split": split,
                    "query_index": query_index,
                    "positive_label": positive,
                    "corpus_index": corpus_index,
                    "negative_label": negative,
                    "cosine_similarity": score,
                }
            )
    return PreferenceSetup(train_selector, validation_selector, summary, manifest)


def _row_predictions(
    dataset: Dataset, paths: list[str], enabled: bool
) -> tuple[list[str | None] | None, float]:
    if not enabled or not paths:
        return None, 0.0
    records, _ = load_prediction_records(paths)
    aligned = align_predictions_to_dataset(dataset, records)
    coverage = sum(prediction is not None for prediction in aligned) / max(len(aligned), 1)
    return aligned, coverage


def _build_confusion(
    config: DPOTrainingConfig, train_dataset: Dataset, validation_dataset: Dataset
) -> PreferenceSetup:
    preference = config.preference
    records, modes = load_prediction_records(preference.train_predictions_paths)
    if any(mode != "train" for mode in modes):
        logger.warning(
            f"Confusion matrix is built from predictions on modes {modes}; use data.mode=train "
            "predictions to avoid leaking evaluation errors into training"
        )
    counts = confusion_counts(records)
    if preference.sample_from == "random":
        distribution = negative_distribution(counts, uniform_mix=1.0)
        sampled_source = SOURCE_RANDOM
    else:
        distribution = negative_distribution(counts, preference.uniform_mix)
        sampled_source = SOURCE_CONFUSION

    train_rows, train_coverage = _row_predictions(
        train_dataset, preference.train_predictions_paths, preference.use_row_predictions
    )
    validation_rows, validation_coverage = _row_predictions(
        validation_dataset, preference.validation_predictions_paths, preference.use_row_predictions
    )
    train_selector = ConfusionNegativeSelector(
        _labels_for_rows(train_dataset), distribution, preference.seed, train_rows, sampled_source
    )
    validation_selector = ConfusionNegativeSelector(
        _labels_for_rows(validation_dataset),
        distribution,
        preference.seed,
        validation_rows,
        sampled_source,
    )

    total = int(counts.sum())
    summary: dict = {
        "preference_strategy": "confusion",
        "train_predictions_paths": list(preference.train_predictions_paths),
        "validation_predictions_paths": list(preference.validation_predictions_paths),
        "confusion_prediction_modes": modes,
        "confusion_source_accuracy": float(counts.trace() / total) if total else 0.0,
        "confusion_matrix": {
            "labels": list(label2idx),
            "counts": counts.tolist(),
        },
        "sample_from": preference.sample_from,
        "uniform_mix": preference.uniform_mix,
        "train_prediction_coverage": train_coverage,
        "validation_prediction_coverage": validation_coverage,
        "train_negative_from_prediction": train_selector.prediction_fraction(),
        "validation_negative_from_prediction": validation_selector.prediction_fraction(),
    }
    logger.info(
        "Confusion negatives: source accuracy %.3f, train coverage %.3f, "
        "train rows using the model's own error %.3f",
        summary["confusion_source_accuracy"],
        train_coverage,
        summary["train_negative_from_prediction"],
    )

    manifest: list[dict[str, ManifestValue]] = []
    for split, selector in (("train", train_selector), ("validation", validation_selector)):
        for query_index, (positive, negative, source) in enumerate(
            zip(selector.query_labels, selector.negative_labels, selector.sources, strict=True)
        ):
            manifest.append(
                {
                    "split": split,
                    "query_index": query_index,
                    "positive_label": positive,
                    "negative_label": negative,
                    "source": source,
                }
            )
    return PreferenceSetup(train_selector, validation_selector, summary, manifest)


def build_negative_selectors(
    config: DPOTrainingConfig, train_dataset: Dataset, validation_dataset: Dataset
) -> PreferenceSetup:
    strategy = config.preference.strategy
    if strategy == "random":
        label_universe = tuple(label2idx)
        return PreferenceSetup(
            RandomNegativeSelector(label_universe, seed=config.preference.seed),
            RandomNegativeSelector(label_universe, seed=config.preference.seed),
            {"preference_strategy": "random"},
        )
    if strategy == "similarity":
        return _build_similarity(config, train_dataset, validation_dataset)
    return _build_confusion(config, train_dataset, validation_dataset)


def write_preference_manifest(output_dir: Path, rows: list[dict[str, ManifestValue]]) -> Path:
    path = output_dir / "preference_mining.jsonl"
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    return path
