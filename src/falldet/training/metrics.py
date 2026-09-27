"""SFT/DPO eval metrics: map teacher-forced logits to classification metrics.

``preprocess_logits_for_metrics`` is passed to SFTTrainer to reduce the
per-step logit tensors from (N, seq_len, vocab_size) to (N, seq_len) argmax
IDs before they accumulate in CPU memory. DPO gets the same (argmax, gold)
pair from ``VideoDPOTrainer.prediction_step``.

``build_compute_metrics`` returns the ``compute_metrics`` callback shared by
SFT and DPO, so both report the same greedy-equivalent classification metrics.
"""

from __future__ import annotations

import numpy as np
import torch

from falldet.inference.prompts.parsers import KeywordOutputParser
from falldet.metrics.base import compute_metrics


def preprocess_logits_for_metrics(
    logits: torch.Tensor | tuple, labels: torch.Tensor
) -> torch.Tensor:
    """Reduce (N, seq_len, vocab_size) logits to (N, seq_len) argmax IDs.

    Shifts predictions right by one so ``preds[i]`` aligns with ``labels[i]``
    (since ``logits[i]`` predicts token ``i+1`` in a causal LM).
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(-1)
    shifted = torch.zeros_like(preds)
    shifted[:, 1:] = preds[:, :-1]
    return shifted


def build_compute_metrics(tokenizer, label2idx: dict[str, int], eos_token_id: int | None = None):
    """Return a compute_metrics callback for teacher-forced eval predictions.

    Inputs are ``(pred_ids, label_ids)``: argmax tokens aligned with the gold answer
    tokens (-100 outside the answer), from SFT's ``preprocess_logits_for_metrics`` or
    ``VideoDPOTrainer.prediction_step``. Gold is cut after the first ``eos_token_id``
    because generation stops there. A row is correct iff every answer token matches,
    which equals greedy decoding. For wrong
    rows, only the argmax tokens up to the first mismatch are decoded, because later
    positions are conditioned on the gold prefix (e.g. " fall" + "_up" -> "fall_up").
    A truncated answer maps to the shortest wrong label it prefixes, else "other"
    (as the inference parser does for unparseable output).
    """
    parser = KeywordOutputParser(label2idx)
    labels_by_length: list[str] = sorted(label2idx, key=lambda label: len(label))

    def _predicted_label(pred_ids: np.ndarray, gold_ids: np.ndarray, true_label: str) -> str:
        mismatches = np.flatnonzero(pred_ids != gold_ids)
        if mismatches.size == 0:
            return true_label
        text = tokenizer.decode(pred_ids[: mismatches[0] + 1], skip_special_tokens=True)
        answer = text.rsplit(":", 1)[-1].strip().lower()
        for label in labels_by_length:  # shortest first, so an exact label wins
            if answer and label != true_label and label.startswith(answer):
                return label
        return "other"  # like the inference parser for unparseable output

    def _answer_positions(label_row: np.ndarray) -> np.ndarray:
        positions = np.flatnonzero(label_row != -100)
        if eos_token_id is not None:
            eos = np.flatnonzero(label_row[positions] == eos_token_id)
            if eos.size:
                positions = positions[: eos[0] + 1]
        return positions

    def teacher_forced_compute_metrics(eval_pred) -> dict[str, float]:
        pred_ids, label_ids = eval_pred

        y_pred: list[str] = []
        y_true: list[str] = []
        for pred_row, label_row in zip(pred_ids, label_ids):
            keep = _answer_positions(label_row)
            gold = label_row[keep]
            true_label = parser.parse(tokenizer.decode(gold, skip_special_tokens=True)).label
            y_true.append(true_label)
            y_pred.append(_predicted_label(pred_row[keep], gold, true_label))

        return compute_metrics(y_pred, y_true)

    return teacher_forced_compute_metrics
