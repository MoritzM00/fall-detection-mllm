from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, Subset

from falldet.training.dataset import DPOConversationDataset
from falldet.training.preferences import (
    RandomNegativeSelector,
    label_for_index,
    observed_labels,
)


class MetadataDataset(Dataset):
    def __init__(self, labels):
        self.targets = torch.tensor(labels)
        self.decode_count = 0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.decode_count += 1
        raise AssertionError("metadata access must not decode video")


class SegmentDataset(Dataset):
    def __init__(self, labels):
        self.video_segments = [{"label": label, "label_str": name} for label, name in labels]

    def __len__(self):
        return len(self.video_segments)


class AggregateDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        total = 0
        self.cumulative_sizes = []
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]


class VideoDataset(Dataset):
    def __init__(self):
        self.video_segments = [
            {"label": 1, "label_str": "fall"},
            {"label": 0, "label_str": "walk"},
        ]

    def __len__(self):
        return len(self.video_segments)

    def __getitem__(self, index):
        segment = self.video_segments[index]
        return {"video": torch.full((2, 3, 4, 4), index), "label_str": segment["label_str"]}


class ConversationStub:
    def build(self, video):
        return SimpleNamespace(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "video", "video": video}],
                }
            ],
            videos=[SimpleNamespace(metadata={"fps": 7.5})],
        )


def test_observed_labels_reads_numeric_metadata_without_decoding():
    dataset = MetadataDataset([5, 1, 5, 0])

    assert observed_labels(dataset) == ("walk", "fall", "lie_down")
    assert dataset.decode_count == 0


def test_metadata_lookup_handles_aggregate_and_noncontiguous_subset():
    aggregate = AggregateDataset(
        [SegmentDataset([(0, "walk"), (1, "fall")]), SegmentDataset([(5, "lie_down")])]
    )
    subset = Subset(aggregate, [2, 0])

    assert label_for_index(subset, 0) == "lie_down"
    assert label_for_index(subset, 1) == "walk"
    assert observed_labels(subset) == ("walk", "lie_down")


def test_random_selector_is_stable_and_excludes_positive():
    selector = RandomNegativeSelector(["lie_down", "walk", "fall", "walk"], seed=7)

    first = [selector.select(index, "fall") for index in range(20)]
    second = [selector.select(index, "fall") for index in range(20)]

    assert first == second
    assert set(first) <= {"walk", "lie_down"}
    assert "fall" not in first


def test_random_selector_rejects_invalid_label_universe():
    with pytest.raises(ValueError, match="at least two"):
        RandomNegativeSelector(["fall", "fall"], seed=0)


def test_dpo_conversation_formats_one_distinct_pair_per_video():
    dataset = DPOConversationDataset(
        VideoDataset(),
        ConversationStub(),
        RandomNegativeSelector(["walk", "fall"], seed=0),
    )

    row = dataset[0]

    assert len(dataset) == 2
    assert row["chosen"][0]["content"][0]["text"] == "The best answer is: fall"
    assert row["rejected"][0]["content"][0]["text"] == "The best answer is: walk"
    assert row["chosen"] != row["rejected"]
    assert row["video_metadata"] == [{"fps": 7.5}]
