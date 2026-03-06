"""Tests for WanfallVideoDataset.

These tests use mocked file I/O so no real data is needed.
"""

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import torch

from falldet.data.video_dataset import idx2label
from falldet.data.wanfall_video_dataset import WanfallVideoDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPLIT_CSV = "path\nvid_a\nvid_b\n"

ANNOTATIONS_DF = pd.DataFrame(
    {
        "path": ["vid_a", "vid_a", "vid_b"],
        "label": [0, 1, 2],
        "start": [0.0, 3.0, 0.0],
        "end": [3.0, 6.0, 4.0],
        "subject": ["s1", "s1", "s2"],
        "cam": ["c1", "c1", "c2"],
        "dataset": ["WanFall", "WanFall", "WanFall"],
        "age_group": ["young", "young", "old"],
        "gender": ["M", "M", "F"],
        "skin_tone": [1, 1, 3],
        "ethnicity": ["A", "A", "B"],
        "bmi_band": ["normal", "normal", "overweight"],
    }
)


def _build_dataset(mode="all", split_root=None, **overrides):
    """Build a WanfallVideoDataset with mocked I/O."""
    kwargs = dict(
        video_root="/fake/videos",
        annotations_file="/fake/annotations.csv",
        target_fps=8.0,
        vid_frame_count=16,
        dataset_name="WanFall",
        mode=mode,
        data_fps=16.0,
        fast=True,
        seed=0,
    )
    if split_root is not None:
        kwargs["split_root"] = split_root
    kwargs.update(overrides)

    with (
        patch(
            "falldet.data.wanfall_video_dataset.resolve_annotations_file",
            return_value="/fake/resolved_annotations.csv",
        ),
        patch(
            "falldet.data.wanfall_video_dataset.resolve_split_file",
            return_value="/fake/resolved_split.csv",
        ),
        patch("pandas.read_csv", return_value=ANNOTATIONS_DF.copy()),
        patch("builtins.open", mock_open(read_data=SPLIT_CSV)),
    ):
        return WanfallVideoDataset(**kwargs)


# ---------------------------------------------------------------------------
# Tests: construction and label loading
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_mode_all_loads_all_segments(self):
        ds = _build_dataset(mode="all")
        assert len(ds) == 3
        assert len(ds.samples) == 2  # vid_a, vid_b

    def test_mode_train_filters_by_split(self):
        ds = _build_dataset(mode="train", split_root="/fake/splits")
        # Split CSV lists vid_a and vid_b, annotations have 3 segments total
        assert len(ds) == 3

    def test_split_is_none(self):
        ds = _build_dataset(mode="all")
        assert ds.split is None

    def test_dataset_name(self):
        ds = _build_dataset(mode="all")
        assert ds.dataset_name == "WanFall"

    def test_inherits_from_omnifall(self):
        from falldet.data.video_dataset import OmnifallVideoDataset

        ds = _build_dataset(mode="all")
        assert isinstance(ds, OmnifallVideoDataset)

    def test_requires_split_root_unless_all(self):
        with pytest.raises(AssertionError, match="Split root must be provided"):
            _build_dataset(mode="train", split_root=None)


# ---------------------------------------------------------------------------
# Tests: label parsing
# ---------------------------------------------------------------------------


class TestLabelParsing:
    def test_integer_labels_mapped_correctly(self):
        ds = _build_dataset(mode="all")
        labels = [seg["label"] for seg in ds.video_segments]
        assert set(labels) == {0, 1, 2}

    def test_label_str_resolved(self):
        ds = _build_dataset(mode="all")
        for seg in ds.video_segments:
            assert seg["label_str"] == idx2label.get(seg["label"])

    def test_segments_sorted_by_path_and_start(self):
        ds = _build_dataset(mode="all")
        keys = [(s["video_path"], s["start"]) for s in ds.video_segments]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Tests: demographic metadata
# ---------------------------------------------------------------------------


class TestDemographicMetadata:
    def test_segments_contain_demographic_fields(self):
        ds = _build_dataset(mode="all")
        for seg in ds.video_segments:
            assert "age_group" in seg
            assert "gender" in seg
            assert "ethnicity" in seg
            assert "bmi_band" in seg

    def test_demographic_values_match_annotations(self):
        ds = _build_dataset(mode="all")
        # Find the vid_b segment (label=2, only one for vid_b)
        vid_b_segs = [s for s in ds.video_segments if s["video_path"] == "vid_b"]
        assert len(vid_b_segs) == 1
        seg = vid_b_segs[0]
        assert seg["age_group"] == "old"
        assert seg["gender"] == "F"
        assert seg["ethnicity"] == "B"
        assert seg["bmi_band"] == "overweight"


# ---------------------------------------------------------------------------
# Tests: _id2label
# ---------------------------------------------------------------------------


class TestId2Label:
    def test_returns_segment_and_label(self):
        ds = _build_dataset(mode="all")
        seg, label = ds._id2label(0)
        assert isinstance(seg, dict)
        assert label == seg["label"]

    def test_all_indices_valid(self):
        ds = _build_dataset(mode="all")
        for i in range(len(ds)):
            seg, label = ds._id2label(i)
            assert label in idx2label


# ---------------------------------------------------------------------------
# Tests: targets property
# ---------------------------------------------------------------------------


class TestTargets:
    def test_returns_tensor(self):
        ds = _build_dataset(mode="all")
        t = ds.targets
        assert isinstance(t, torch.Tensor)
        assert len(t) == len(ds)

    def test_target_values(self):
        ds = _build_dataset(mode="all")
        t = ds.targets
        assert set(t.tolist()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Tests: get_random_offset (inherited from OmnifallVideoDataset)
# ---------------------------------------------------------------------------


class TestGetRandomOffset:
    def test_short_segment_returns_start_frame(self):
        """When segment is shorter than the clip, offset should be segment start."""
        ds = _build_dataset(mode="all", vid_frame_count=160)
        # vid_frame_count=160 at target_fps=8 -> clip_duration = 159/8 = 19.875s
        # All our test segments are <=6s, so they're all too short.
        seg = ds.video_segments[0]
        fps = 16.0
        offset = ds.get_random_offset(None, 1, 0, fps)
        expected_start = int(seg["start"] * fps)
        assert offset == expected_start

    def test_offset_within_segment_bounds(self):
        """Offset should always produce a clip within segment boundaries."""
        ds = _build_dataset(mode="all")
        fps = 16.0
        for idx in range(len(ds)):
            seg = ds.video_segments[idx]
            offset = ds.get_random_offset(None, 1, idx, fps)
            seg_start_frame = int(seg["start"] * fps)
            seg_end_frame = int(seg["end"] * fps)
            assert offset >= seg_start_frame, (
                f"Segment {idx}: offset {offset} < segment start {seg_start_frame}"
            )
            # The clip end should not exceed the segment end
            clip_duration_sec = (ds.vid_frame_count - 1) / ds.target_fps
            required_frames = int(clip_duration_sec * fps) + 1
            clip_end = offset + required_frames
            # If clip fits, end should be within bounds; otherwise offset == start
            seg_frames = seg_end_frame - seg_start_frame
            if seg_frames > required_frames:
                assert clip_end <= seg_end_frame, (
                    f"Segment {idx}: clip end {clip_end} > segment end {seg_end_frame}"
                )

    def test_reproducible_with_seed(self):
        """Same seed + same idx should produce the same offset."""
        ds = _build_dataset(mode="all", seed=42)
        fps = 16.0
        offsets1 = [ds.get_random_offset(None, 1, i, fps) for i in range(len(ds))]
        offsets2 = [ds.get_random_offset(None, 1, i, fps) for i in range(len(ds))]
        assert offsets1 == offsets2


# ---------------------------------------------------------------------------
# Tests: load_item (with mocked video loading)
# ---------------------------------------------------------------------------


class TestLoadItem:
    def _make_dataset_with_mock_video(self):
        ds = _build_dataset(mode="all")
        # Mock load_video to return fake frames (16 frames, 64x48, RGB)
        fake_frames = [np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8) for _ in range(16)]
        ds.load_video = MagicMock(return_value=fake_frames)
        return ds

    def test_returns_base_segment_keys(self):
        ds = self._make_dataset_with_mock_video()
        item = ds.load_item(0)
        for key in (
            "video",
            "label",
            "label_str",
            "video_path",
            "start_time",
            "end_time",
            "segment_duration",
            "dataset",
        ):
            assert key in item, f"Missing key: {key}"

    def test_returns_demographic_keys(self):
        ds = self._make_dataset_with_mock_video()
        item = ds.load_item(0)
        for key in ("age_group", "gender", "ethnicity", "bmi_band"):
            assert key in item, f"Missing demographic key: {key}"

    def test_video_tensor_shape(self):
        ds = self._make_dataset_with_mock_video()
        item = ds.load_item(0)
        video = item["video"]
        assert video.shape[0] == 16  # T
        assert video.shape[1] == 3  # C

    def test_label_matches_segment(self):
        ds = self._make_dataset_with_mock_video()
        for idx in range(len(ds)):
            item = ds.load_item(idx)
            assert item["label"] == ds.video_segments[idx]["label"]
            assert item["dataset"] == "WanFall"


# ---------------------------------------------------------------------------
# Tests: __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_format(self):
        ds = _build_dataset(mode="all")
        r = repr(ds)
        assert "WanfallVideoDataset" in r
        assert "WanFall" in r
        assert "all" in r

    def test_repr_does_not_contain_split(self):
        """WanFall repr should not show a split field (unlike Omnifall)."""
        ds = _build_dataset(mode="all")
        r = repr(ds)
        assert "split=" not in r


# ---------------------------------------------------------------------------
# Tests: kwargs absorption (split=None from factory)
# ---------------------------------------------------------------------------


class TestFactoryCompatibility:
    def test_split_kwarg_absorbed(self):
        """Factory passes split=None for WanFall; ensure it doesn't raise."""
        ds = _build_dataset(mode="all", split=None)
        assert ds.split is None

    def test_extra_kwargs_absorbed(self):
        """Unexpected kwargs should be silently absorbed."""
        ds = _build_dataset(mode="all", split="cs", some_unknown_param=True)
        assert ds is not None
