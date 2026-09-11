from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, Subset

from falldet.training.dataset import DPOConversationDataset
from falldet.training.preferences import (
    EmbeddingSimilarityNegativeSelector,
    RandomNegativeSelector,
    align_embeddings_to_dataset,
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


class IdentitySegmentDataset(Dataset):
    dataset_name = "OOPS"

    def __init__(self):
        self.video_segments = [
            {
                "video_path": "video-a",
                "start": 0.0,
                "end": 1.0,
                "label": 1,
                "label_str": "fall",
            },
            {
                "video_path": "video-b",
                "start": 1.0,
                "end": 2.0,
                "label": 0,
                "label_str": "walk",
            },
            {
                "video_path": "video-c",
                "start": 2.0,
                "end": 3.0,
                "label": 9,
                "label_str": "other",
            },
        ]

    def __len__(self):
        return len(self.video_segments)


def _artifact_sample(segment):
    return {
        "dataset": "OOPS",
        "video_path": segment["video_path"],
        "start_time": segment["start"],
        "end_time": segment["end"],
        "label": segment["label"],
        "label_str": segment["label_str"],
    }


def test_embedding_alignment_uses_manifest_identity_for_subset_rows():
    dataset = IdentitySegmentDataset()
    samples = [
        _artifact_sample(dataset.video_segments[1]),
        _artifact_sample(dataset.video_segments[2]),
        _artifact_sample(dataset.video_segments[0]),
    ]
    embeddings = torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])

    aligned = align_embeddings_to_dataset(Subset(dataset, [2, 0]), embeddings, samples)

    assert aligned.tolist() == [[20.0, 0.0], [30.0, 0.0]]


def test_embedding_alignment_rejects_missing_dataset_row():
    dataset = IdentitySegmentDataset()
    samples = [_artifact_sample(dataset.video_segments[0])]

    with pytest.raises(ValueError, match="absent from the embedding manifest"):
        align_embeddings_to_dataset(dataset, torch.ones(1, 2), samples)


def test_similarity_selector_chooses_nearest_wrong_class_in_chunks():
    selector = EmbeddingSimilarityNegativeSelector(
        query_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        corpus_embeddings=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        query_labels=["fall", "walk"],
        corpus_labels=["fall", "walk", "walk", "other"],
        chunk_size=1,
    )

    assert selector.select(0, "fall") == "walk"
    assert selector.select(1, "walk") == "other"
    assert selector.corpus_indices == (1, 3)
    assert selector.scores[0] == pytest.approx(0.9938837)
    assert selector.scores[1] == pytest.approx(0.9938837)


def test_similarity_selector_rejects_query_label_mismatch():
    selector = EmbeddingSimilarityNegativeSelector(
        query_embeddings=torch.tensor([[1.0, 0.0]]),
        corpus_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        query_labels=["fall"],
        corpus_labels=["fall", "walk"],
    )

    with pytest.raises(ValueError, match="Query label mismatch"):
        selector.select(0, "walk")
