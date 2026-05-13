"""SFT eval metrics: map teacher-forced logits to classification metrics.

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


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reduce (N, seq_len, vocab_size) logits to (N, seq_len) argmax IDs."""
    return logits.argmax(-1)


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
