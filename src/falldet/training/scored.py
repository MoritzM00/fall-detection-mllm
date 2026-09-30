"""Per-clip hard negatives from a model's teacher-forced label scores.

The rejected label of a clip is the wrong label the scoring model (usually the SFT
model DPO starts from) ranks highest for that clip, from ``scripts/score_labels.py``.

With ``balance``, each label is rejected exactly as often as it is chosen, so DPO
cannot shift the label prior; within that constraint the summed score of the
rejected labels is maximized (an assignment of clips to rejection slots).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def balanced_negatives(
    positives: Sequence[str], row_scores: Sequence[Mapping[str, float]]
) -> list[str]:
    """Wrong labels with rejected counts equal to chosen counts, maximizing their scores."""

    labels = list(label2idx)
    slots = [label for label in labels for _ in range(positives.count(label))]
    slot_ids = np.array([label2idx[label] for label in slots])
    scores = np.array([[row[label] for label in labels] for row in row_scores])
    cost = -scores[:, slot_ids]
    own = np.array([label2idx[label] for label in positives])[:, None] == slot_ids[None, :]
    # Finite so the solver always returns a full assignment; used only if infeasible
    cost[own] = np.abs(cost).max() * len(positives) + 1.0
    rows, cols = linear_sum_assignment(cost)
    if own[rows, cols].any():
        raise ValueError("Balanced negatives are infeasible: one label is over half the rows")
    negatives = [""] * len(positives)
    for row, col in zip(rows, cols, strict=True):
        negatives[row] = slots[col]
    return negatives


class ScoredNegativeSelector(PrecomputedNegativeSelector):
    """Reject a high-scoring wrong label; rows without scores use ``fallback``.

    Without ``balance`` the highest-scoring wrong label; with it, the score-maximizing
    assignment in which every label is rejected as often as it is chosen.
    """

    def __init__(
        self,
        query_labels: Sequence[str],
        row_scores: Sequence[Mapping[str, float] | None],
        fallback: NegativeSelector,
        balance: bool = False,
    ):
        if len(row_scores) != len(query_labels):
            raise ValueError("Row scores must match query labels")

        self.query_labels = tuple(canonicalize_label(label) for label in query_labels)
        scored = {index: scores for index, scores in enumerate(row_scores) if scores is not None}
        hardest = {
            index: max(
                (label for label in scores if label != self.query_labels[index]),
                key=scores.__getitem__,
            )
            for index, scores in scored.items()
        }
        chosen = dict(hardest)
        if balance and scored:
            assigned = balanced_negatives(
                [self.query_labels[index] for index in scored], list(scored.values())
            )
            chosen = dict(zip(scored, assigned, strict=True))

        negatives: list[str] = []
        sources: list[str] = []
        margins: list[float | None] = []
        correct: list[bool | None] = []
        kept: list[bool] = []
        for index, (positive, scores) in enumerate(zip(self.query_labels, row_scores, strict=True)):
            if scores is None:
                negatives.append(fallback.select(index, positive))
                sources.append(SOURCE_FALLBACK)
                margins.append(None)
                correct.append(None)
                continue
            negative = chosen[index]
            negatives.append(negative)
            sources.append(SOURCE_SCORE)
            margins.append(scores[positive] - scores[negative])
            correct.append(scores[positive] > scores[hardest[index]])
            kept.append(negative == hardest[index])

        self.negative_labels = tuple(negatives)
        self.sources = tuple(sources)
        self.margins = tuple(margins)
        self.correct = tuple(correct)
        self.hardest_fraction = sum(kept) / max(len(kept), 1)

    def summary(self, prefix: str) -> dict[str, float]:
        """Coverage, scoring-model accuracy and mean positive-minus-negative margin."""

        scored = [margin for margin in self.margins if margin is not None]
        hits = [hit for hit in self.correct if hit is not None]
        return {
            f"{prefix}_score_coverage": len(scored) / max(len(self.margins), 1),
            f"{prefix}_scoring_model_accuracy": sum(hits) / max(len(hits), 1),
            f"{prefix}_score_margin_mean": sum(scored) / max(len(scored), 1),
            f"{prefix}_hardest_negative_fraction": self.hardest_fraction,
        }
