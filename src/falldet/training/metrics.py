"""SFT/DPO eval metrics: map teacher-forced logits to classification metrics.

``preprocess_logits_for_metrics`` is passed to SFTTrainer to reduce the
per-step logit tensors from (N, seq_len, vocab_size) to (N, seq_len) argmax
IDs before they accumulate in CPU memory.

``build_sft_compute_metrics`` returns the ``compute_metrics`` callback that
decodes the argmax predictions and ground-truth labels from the unmasked
completion tokens, then delegates to the project-wide ``compute_metrics``.
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


def build_sft_compute_metrics(tokenizer, label2idx: dict[str, int]):
    """Return a compute_metrics callback wired to the given tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer (processor.tokenizer).
        label2idx: Label-to-index mapping used to instantiate the parser.

    Returns:
        Callable compatible with Trainer's compute_metrics signature.
    """
    parser = KeywordOutputParser(label2idx)

    def _decode(token_ids: np.ndarray, mask: np.ndarray) -> str:
        unmasked = token_ids[mask != -100]
        return tokenizer.decode(unmasked, skip_special_tokens=True)

    def sft_compute_metrics(eval_pred) -> dict[str, float]:
        pred_ids, label_ids = eval_pred  # numpy (N, seq_len) after preprocessing

        y_pred: list[str] = []
        y_true: list[str] = []

        for pred_row, label_row in zip(pred_ids, label_ids):
            gt_text = _decode(label_row, label_row)
            pred_text = _decode(pred_row, label_row)

            y_true.append(parser.parse(gt_text).label)
            y_pred.append(parser.parse(pred_text).label)

        return compute_metrics(y_pred, y_true)

    return sft_compute_metrics


def build_dpo_compute_metrics(tokenizer, label2idx: dict[str, int]):
    """Return a compute_metrics callback for teacher-forced DPO eval predictions.

    Inputs are ``(pred_ids, label_ids)`` from ``VideoDPOTrainer.prediction_step``:
    argmax tokens and gold chosen-answer tokens (-100 outside the answer). A row is
    correct iff every answer token matches, which equals greedy decoding. For wrong
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
        if answer in label2idx and answer != true_label:
            return answer
        for label in labels_by_length:
            if answer and label != true_label and label.startswith(answer):
                return label
        return "other"  # like the inference parser for unparseable output

    def dpo_compute_metrics(eval_pred) -> dict[str, float]:
        pred_ids, label_ids = eval_pred

        y_pred: list[str] = []
        y_true: list[str] = []
        for pred_row, label_row in zip(pred_ids, label_ids):
            keep = label_row != -100
            gold = label_row[keep]
            true_label = parser.parse(tokenizer.decode(gold, skip_special_tokens=True)).label
            y_true.append(true_label)
            y_pred.append(_predicted_label(pred_row[keep], gold, true_label))

        return compute_metrics(y_pred, y_true)

    return dpo_compute_metrics
