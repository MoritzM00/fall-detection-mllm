"""Per-clip hard negatives from a model's teacher-forced label scores.

The rejected label of a clip is the wrong label the scoring model (usually the SFT
model DPO starts from) ranks highest for that clip, from ``scripts/score_labels.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from falldet.data.video_dataset import label2idx
from falldet.training.preferences import (
    NegativeSelector,
    PrecomputedNegativeSelector,
    canonicalize_label,
)
from falldet.utils.predictions import load_predictions_jsonl

logger = logging.getLogger(__name__)

SOURCE_SCORE = "score"
SOURCE_FALLBACK = "fallback"


def load_label_scores(path: str | Path) -> tuple[dict, list[dict]]:
    """Load a ``score_labels.py`` JSONL; every row must score every label."""

    metadata, records = load_predictions_jsonl(Path(path).expanduser())
    for record in records:
        scores = record.get("label_logprobs")
        if not isinstance(scores, dict) or set(scores) != set(label2idx):
            raise ValueError(f"{path}: row {record.get('idx')} does not score every label")
    logger.info(f"Loaded label scores for {len(records)} clips from {path}")
    return metadata, records


class ScoredNegativeSelector(PrecomputedNegativeSelector):
    """Reject the highest-scoring wrong label; rows without scores use ``fallback``."""

    def __init__(
        self,
        query_labels: Sequence[str],
        row_scores: Sequence[Mapping[str, float] | None],
        fallback: NegativeSelector,
    ):
        if len(row_scores) != len(query_labels):
            raise ValueError("Row scores must match query labels")

        self.query_labels = tuple(canonicalize_label(label) for label in query_labels)
        negatives: list[str] = []
        sources: list[str] = []
        margins: list[float | None] = []
        correct: list[bool | None] = []
        for index, (positive, scores) in enumerate(zip(self.query_labels, row_scores, strict=True)):
            if scores is None:
                negatives.append(fallback.select(index, positive))
                sources.append(SOURCE_FALLBACK)
                margins.append(None)
                correct.append(None)
                continue
            negative = max((label for label in scores if label != positive), key=scores.__getitem__)
            margin = scores[positive] - scores[negative]
            negatives.append(negative)
            sources.append(SOURCE_SCORE)
            margins.append(margin)
            correct.append(margin > 0)

        self.negative_labels = tuple(negatives)
        self.sources = tuple(sources)
        self.margins = tuple(margins)
        self.correct = tuple(correct)

    def summary(self, prefix: str) -> dict[str, float]:
        """Coverage, scoring-model accuracy and mean positive-minus-negative margin."""

        scored = [margin for margin in self.margins if margin is not None]
        hits = [hit for hit in self.correct if hit is not None]
        return {
            f"{prefix}_score_coverage": len(scored) / max(len(self.margins), 1),
            f"{prefix}_scoring_model_accuracy": sum(hits) / max(len(hits), 1),
            f"{prefix}_score_margin_mean": sum(scored) / max(len(scored), 1),
        }
