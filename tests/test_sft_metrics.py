"""Tests for SFT eval metrics helpers."""

import numpy as np
import pytest
import torch

from falldet.training.metrics import build_sft_compute_metrics, preprocess_logits_for_metrics

LABEL2IDX = {
    "walk": 0,
    "fall": 1,
    "fallen": 2,
    "sit_down": 3,
    "sitting": 4,
    "lie_down": 5,
    "lying": 6,
    "stand_up": 7,
    "standing": 8,
    "other": 9,
}

# Token IDs used by the fake tokenizer (must not collide with -100)
_COMPLETION_TOKENS = {
    # "The best answer is: <label>" rendered as a sequence of small ints
    "walk": [10, 11, 12, 13, 14],  # ... is: walk
    "fall": [10, 11, 12, 13, 15],
    "fallen": [10, 11, 12, 13, 16, 17],  # two tokens: fall + en
    "sit_down": [10, 11, 12, 13, 18, 19],  # two tokens: sit + _down
    "lie_down": [10, 11, 12, 13, 20, 21],
    "stand_up": [10, 11, 12, 13, 22, 23],
    "sitting": [10, 11, 12, 13, 24],
    "lying": [10, 11, 12, 13, 25],
    "standing": [10, 11, 12, 13, 26],
    "other": [10, 11, 12, 13, 27],
}

# Map each label to the decoded string it would produce
_DECODED = {label: f"The best answer is: {label}" for label in LABEL2IDX}


class FakeTokenizer:
    """Minimal tokenizer stub that decodes token-id lists back to label strings."""

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = list(token_ids)
        for label, toks in _COMPLETION_TOKENS.items():
            if ids == toks:
                return _DECODED[label]
        return ""


# ---------------------------------------------------------------------------
# preprocess_logits_for_metrics
# ---------------------------------------------------------------------------


class TestPreprocessLogits:
    def test_argmax_shape(self):
        N, seq_len, vocab = 4, 20, 100
        logits = torch.randn(N, seq_len, vocab)
        labels = torch.zeros(N, seq_len, dtype=torch.long)
        out = preprocess_logits_for_metrics(logits, labels)
        assert out.shape == (N, seq_len)

    def test_argmax_values(self):
        logits = torch.zeros(2, 5, 10)
        logits[0, 2, 7] = 10.0  # sample 0, pos 2 → token 7
        logits[1, 4, 3] = 10.0  # sample 1, pos 4 → token 3
        labels = torch.zeros(2, 5, dtype=torch.long)
        out = preprocess_logits_for_metrics(logits, labels)
        assert out[0, 2].item() == 7
        assert out[1, 4].item() == 3


# ---------------------------------------------------------------------------
# build_sft_compute_metrics
# ---------------------------------------------------------------------------


def _make_batch(true_labels: list[str], pred_labels: list[str]):
    """Build (pred_ids, label_ids) numpy arrays for a batch of samples.

    Prompt tokens are represented as 1s with label -100; completion tokens
    carry their actual IDs in label_ids.
    """
    PROMPT_LEN = 5

    label_rows = []
    pred_rows = []

    for true_lbl, pred_lbl in zip(true_labels, pred_labels):
        gt_toks = _COMPLETION_TOKENS[true_lbl]
        pd_toks = _COMPLETION_TOKENS[pred_lbl]

        # Pad shorter completion so all rows have the same length
        max_comp = max(len(gt_toks), len(pd_toks))
        gt_padded = gt_toks + [0] * (max_comp - len(gt_toks))
        pd_padded = pd_toks + [0] * (max_comp - len(pd_toks))

        label_row = [-100] * PROMPT_LEN + gt_padded
        pred_row = [1] * PROMPT_LEN + pd_padded  # prompt preds don't matter

        label_rows.append(label_row)
        pred_rows.append(pred_row)

    # Pad rows to equal length across samples
    max_len = max(len(r) for r in label_rows)
    label_arr = np.array([r + [-100] * (max_len - len(r)) for r in label_rows])
    pred_arr = np.array([r + [0] * (max_len - len(r)) for r in pred_rows])

    return pred_arr, label_arr


class TestBuildSftComputeMetrics:
    def setup_method(self):
        self.tokenizer = FakeTokenizer()
        self.compute = build_sft_compute_metrics(self.tokenizer, LABEL2IDX)

    def test_perfect_predictions(self):
        labels = ["walk", "fall", "fallen", "sit_down"]
        pred_ids, label_ids = _make_batch(labels, labels)
        metrics = self.compute((pred_ids, label_ids))
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_all_wrong(self):
        true_labels = ["walk", "walk", "walk", "walk"]
        pred_labels = ["fall", "fall", "fall", "fall"]
        pred_ids, label_ids = _make_batch(true_labels, pred_labels)
        metrics = self.compute((pred_ids, label_ids))
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_mixed_predictions(self):
        true_labels = ["walk", "fall", "fallen", "sit_down"]
        pred_labels = ["walk", "fall", "walk", "sit_down"]  # 3/4 correct
        pred_ids, label_ids = _make_batch(true_labels, pred_labels)
        metrics = self.compute((pred_ids, label_ids))
        assert metrics["accuracy"] == pytest.approx(0.75)

    def test_multi_token_labels(self):
        """Multi-token labels (sit_down, lie_down, stand_up) parse correctly."""
        for label in ("sit_down", "lie_down", "stand_up"):
            pred_ids, label_ids = _make_batch([label], [label])
            metrics = self.compute((pred_ids, label_ids))
            assert metrics["accuracy"] == pytest.approx(1.0), f"failed for {label}"

    def test_returns_standard_keys(self):
        pred_ids, label_ids = _make_batch(["walk"], ["walk"])
        metrics = self.compute((pred_ids, label_ids))
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert key in metrics
