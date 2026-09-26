import json

import numpy as np
import pytest
from torch.utils.data import Dataset, Subset

from falldet.data.video_dataset import label2idx
from falldet.training.confusion import (
    SOURCE_CONFUSION,
    SOURCE_PREDICTION,
    ConfusionNegativeSelector,
    align_predictions_to_dataset,
    confusion_counts,
    load_prediction_records,
    negative_distribution,
)


def _prediction(video, label, predicted, start=0.0, end=1.0):
    return {
        "dataset": "OOPS",
        "video_path": video,
        "start_time": start,
        "end_time": end,
        "label": label2idx[label],
        "label_str": label,
        "predicted_label": predicted,
    }


class SegmentDataset(Dataset):
    dataset_name = "OOPS"

    def __init__(self, rows):
        self.video_segments = [
            {"video_path": video, "start": 0.0, "end": 1.0, "label_str": label}
            for video, label in rows
        ]

    def __len__(self):
        return len(self.video_segments)


def test_confusion_counts_ignores_out_of_vocabulary_predictions():
    counts = confusion_counts(
        [
            _prediction("a", "fall", "fall"),
            _prediction("b", "fall", "lie_down"),
            _prediction("c", "fall", "not a label"),
            _prediction("d", "walk", "standing"),
        ]
    )

    assert counts.sum() == 3
    assert counts[label2idx["fall"], label2idx["fall"]] == 1
    assert counts[label2idx["fall"], label2idx["lie_down"]] == 1
    assert counts[label2idx["walk"], label2idx["standing"]] == 1


def test_negative_distribution_mixes_errors_with_uniform_and_excludes_positive():
    counts = np.zeros((len(label2idx), len(label2idx)), dtype=np.int64)
    fall, lie_down, sitting = label2idx["fall"], label2idx["lie_down"], label2idx["sitting"]
    counts[fall, fall] = 100
    counts[fall, lie_down] = 3
    counts[fall, sitting] = 1

    distribution = negative_distribution(counts, uniform_mix=0.2)

    np.testing.assert_allclose(distribution.sum(axis=1), 1.0)
    assert np.all(np.diag(distribution) == 0)
    uniform = 0.2 / (len(label2idx) - 1)
    assert distribution[fall, lie_down] == pytest.approx(0.8 * 0.75 + uniform)
    assert distribution[fall, sitting] == pytest.approx(0.8 * 0.25 + uniform)
    # A class the model never confused falls back to uniform over wrong labels.
    walk = label2idx["walk"]
    np.testing.assert_allclose(
        distribution[walk][np.arange(len(label2idx)) != walk], 1 / (len(label2idx) - 1)
    )


def test_pure_confusion_distribution_samples_only_observed_mistakes():
    counts = np.zeros((len(label2idx), len(label2idx)), dtype=np.int64)
    counts[label2idx["fall"], label2idx["lie_down"]] = 5
    distribution = negative_distribution(counts, uniform_mix=0.0)

    selector = ConfusionNegativeSelector(["fall"] * 50, distribution, seed=3)

    assert set(selector.negative_labels) == {"lie_down"}
    assert set(selector.sources) == {SOURCE_CONFUSION}


def test_selector_prefers_row_prediction_and_is_deterministic():
    distribution = negative_distribution(
        np.zeros((len(label2idx), len(label2idx)), dtype=np.int64), uniform_mix=0.1
    )
    labels = ["fall", "fall", "walk"]
    rows = ["sitting", "fall", None]

    first = ConfusionNegativeSelector(labels, distribution, seed=1, row_predictions=rows)
    second = ConfusionNegativeSelector(labels, distribution, seed=1, row_predictions=rows)

    assert first.negative_labels == second.negative_labels
    assert first.select(0, "fall") == "sitting"
    assert first.sources == (SOURCE_PREDICTION, SOURCE_CONFUSION, SOURCE_CONFUSION)
    assert first.select(1, "fall") != "fall"
    assert first.select(2, "walk") != "walk"
    assert first.prediction_fraction() == pytest.approx(1 / 3)


def test_selector_rejects_label_mismatch():
    distribution = negative_distribution(
        np.zeros((len(label2idx), len(label2idx)), dtype=np.int64), uniform_mix=1.0
    )
    selector = ConfusionNegativeSelector(["fall"], distribution, seed=0)

    with pytest.raises(ValueError, match="label mismatch"):
        selector.select(0, "walk")


def test_predictions_align_to_subset_rows_and_tolerate_missing_rows():
    dataset = SegmentDataset([("a", "fall"), ("b", "walk"), ("c", "sitting")])
    records = [_prediction("c", "sitting", "lying"), _prediction("a", "fall", "fall")]

    aligned = align_predictions_to_dataset(Subset(dataset, [2, 1, 0]), records)

    assert aligned == ["lying", None, "fall"]


def test_predictions_alignment_rejects_duplicates():
    dataset = SegmentDataset([("a", "fall")])
    records = [_prediction("a", "fall", "walk"), _prediction("a", "fall", "sitting")]

    with pytest.raises(ValueError, match="Duplicate"):
        align_predictions_to_dataset(dataset, records)


def test_load_prediction_records_reports_generation_mode(tmp_path):
    path = tmp_path / "run.jsonl"
    lines = [
        {"type": "metadata", "config": {"data": {"mode": "train"}}},
        {"type": "prediction", "idx": 0, **_prediction("a", "fall", "walk")},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    records, modes = load_prediction_records([path])

    assert modes == ["train"]
    assert records[0]["predicted_label"] == "walk"
