"""Tests for teacher-forced SFT/DPO eval metrics."""

import numpy as np

from falldet.training.metrics import build_compute_metrics

LABEL2IDX = {"walk": 0, "fall": 1, "fallen": 2, "lie_down": 5, "stand_up": 7, "other": 9}

VOCAB = {
    1: "The best answer is:",
    2: " walk",
    3: " fall",
    4: "en",
    5: " lie",
    6: "_down",
    7: " stand",
    8: "_up",
    9: " other",
    10: " xyz",
    11: "<eos>",
    12: "\n",
}


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(VOCAB[int(i)] for i in ids if not (skip_special_tokens and i == 11))


def _rows(pairs):
    """Build padded (pred_ids, label_ids) with a masked prompt position in front."""

    width = 1 + max(len(gold) for _, gold in pairs)
    preds = np.full((len(pairs), width), -100)
    labels = np.full((len(pairs), width), -100)
    for row, (pred, gold) in enumerate(pairs):
        preds[row, 0] = 0
        preds[row, 1 : 1 + len(pred)] = pred
        labels[row, 1 : 1 + len(gold)] = gold
    return preds, labels


def _metrics(pairs):
    return build_compute_metrics(FakeTokenizer(), LABEL2IDX, eos_token_id=11)(_rows(pairs))


def test_all_tokens_matching_counts_as_correct():
    metrics = _metrics([([1, 3, 11], [1, 3, 11]), ([1, 5, 6, 11], [1, 5, 6, 11])])

    assert metrics["accuracy"] == 1.0


def test_wrong_label_is_decoded_only_up_to_first_mismatch():
    # True stand_up; model picks " fall" first, then "_up" is forced by the gold prefix.
    metrics = _metrics([([1, 3, 8, 11], [1, 7, 8, 11])])

    assert metrics["accuracy"] == 0.0
    assert metrics["pred_dist_fall"] == 1.0


def test_mismatch_after_shared_prefix_yields_the_longer_label():
    # True fall; model continues " fall" with "en" instead of EOS -> fallen.
    metrics = _metrics([([1, 3, 4], [1, 3, 11])])

    assert metrics["accuracy"] == 0.0
    assert metrics["pred_dist_fallen"] == 1.0


def test_truncated_label_maps_to_label_it_prefixes():
    # True walk; model picks " lie" (first token of lie_down).
    metrics = _metrics([([1, 5, 11], [1, 2, 11])])

    assert metrics["pred_dist_lie_down"] == 1.0


def test_unparseable_prediction_maps_to_other():
    metrics = _metrics([([1, 10, 11], [1, 2, 11])])

    assert metrics["accuracy"] == 0.0
    assert metrics["pred_dist_other"] == 1.0


def test_tokens_after_eos_are_ignored():
    # Generation stops at EOS, so the chat-template newline after it is not scored.
    metrics = _metrics([([1, 3, 11, 9], [1, 3, 11, 12])])

    assert metrics["accuracy"] == 1.0
