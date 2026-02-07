import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from infreqact.data.video_dataset import OmnifallVideoDataset


@pytest.fixture(scope="module")
def omnifall_root():
    of_root = os.getenv("OMNIFALL_ROOT")
    if of_root is None:
        pytest.skip("OMNIFALL_ROOT environment variable not set.")
    return of_root


@pytest.fixture(scope="module")
def test_omnifall_video_dataset(omnifall_root):
    config = {
        "video_root": f"{omnifall_root}/OOPS/video",
        "annotations_file": "hf://simplexsigil2/omnifall/labels/OOPS.csv",
        "split_root": "hf://simplexsigil2/omnifall/splits",
        "dataset_name": "OOPS",
        "mode": "test",
        "split": "cs",
        "target_fps": 8.0,
        "vid_frame_count": 16,
        "data_fps": 30.0,
        "ext": ".mp4",
        "fast": True,
        "size": 224,
        "seed": 0,
    }
    dataset = OmnifallVideoDataset(**config)
    return dataset


def test_random_offset_seed(test_omnifall_video_dataset):
    """Test that random offset is consistent across runs with the same seed."""
    dataset = test_omnifall_video_dataset
    frame_count = 512  # Example frame count
    fps = dataset.data_fps
    target_interval = 1

    n_offsets = 100

    offsets1 = [
        dataset.get_random_offset(
            length=frame_count, target_interval=target_interval, idx=i, fps=fps
        )[0]  # Extract offset from (offset, is_too_short) tuple
        for i in range(n_offsets)
    ]

    offsets2 = [
        dataset.get_random_offset(
            length=frame_count, target_interval=target_interval, idx=i, fps=fps
        )[0]  # Extract offset from (offset, is_too_short) tuple
        for i in range(n_offsets)
    ]
    assert offsets1 == offsets2, "Random offsets differ between runs with the same seed."

    dataset.seed = 42
    offsets3 = [
        dataset.get_random_offset(
            length=frame_count, target_interval=target_interval, idx=i, fps=fps
        )[0]  # Extract offset from (offset, is_too_short) tuple
        for i in range(n_offsets)
    ]
    assert offsets1 != offsets3, "Random offsets are the same for different seeds."
    assert offsets2 != offsets3, "Random offsets are the same for different seeds."


# ============================================================================
# Tests for compute_actual_frame_count
# ============================================================================


class TestComputeActualFrameCount:
    """Tests for the compute_actual_frame_count method."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset with configurable parameters."""
        dataset = MagicMock(spec=OmnifallVideoDataset)
        dataset.target_fps = 8.0
        dataset.vid_frame_count = 16
        dataset.data_fps = 30.0
        # Bind the actual method to our mock
        dataset.compute_actual_frame_count = (
            lambda duration: OmnifallVideoDataset.compute_actual_frame_count(dataset, duration)
        )
        return dataset

    @pytest.mark.parametrize(
        "duration,expected",
        [
            (4.0, 16),  # long segment, capped to vid_frame_count
            (2.0, 16),  # medium segment, capped to vid_frame_count
            (1.875, 15),  # exact boundary, minus margin for start-frame rounding
            (1.0, 8),  # short segment, minus margin for start-frame rounding
            (0.5, 4),  # very short segment, minus margin for start-frame rounding
            (0.1, 1),  # minimal segment
        ],
        ids=["long", "medium", "exact_boundary", "short", "very_short", "minimal"],
    )
    def test_compute_actual_frame_count(self, mock_dataset, duration, expected):
        """Test compute_actual_frame_count with various durations."""
        result = mock_dataset.compute_actual_frame_count(duration)
        assert result == expected, (
            f"Expected {expected} frames for {duration}s segment, got {result}"
        )

    def test_negative_duration_returns_at_least_one(self, mock_dataset):
        """Test that negative duration returns at least 1 frame."""
        result = mock_dataset.compute_actual_frame_count(-1.0)
        assert result == 1, f"Expected at least 1 frame for negative duration, got {result}"

    def test_zero_duration(self, mock_dataset):
        """Test that zero duration returns at least 1 frame."""
        result = mock_dataset.compute_actual_frame_count(0.0)
        assert result == 1, f"Expected 1 frame for zero duration, got {result}"

    def test_vid_frame_count_none_returns_none(self):
        """When vid_frame_count is None, should return None."""
        dataset = MagicMock(spec=OmnifallVideoDataset)
        dataset.vid_frame_count = None
        dataset.target_fps = 8.0
        dataset.compute_actual_frame_count = (
            lambda duration: OmnifallVideoDataset.compute_actual_frame_count(dataset, duration)
        )
        result = dataset.compute_actual_frame_count(4.0)
        assert result is None, "Expected None when vid_frame_count is None"


# ============================================================================
# Tests for get_random_offset with frame_count parameter
# ============================================================================


class TestGetRandomOffsetWithFrameCount:
    """Tests for get_random_offset with the frame_count parameter."""

    def test_frame_count_parameter_used(self, test_omnifall_video_dataset):
        """Test that frame_count parameter overrides vid_frame_count."""
        dataset = test_omnifall_video_dataset
        fps = 30.0

        # Collect multiple offsets to verify different ranges
        offsets_default = [
            dataset.get_random_offset(length=512, target_interval=1, idx=i, fps=fps)[0]
            for i in range(100)
        ]

        offsets_small = [
            dataset.get_random_offset(length=512, target_interval=1, idx=i, fps=fps, frame_count=4)[
                0
            ]
            for i in range(100)
        ]

        # With vid_frame_count=16: required_span=15, max_offset=512-1-15=496
        # With frame_count=4: required_span=3, max_offset=512-1-3=508
        max_default = max(offsets_default)
        max_small = max(offsets_small)

        # frame_count=4 should allow larger offsets
        assert max_small > max_default, (
            f"frame_count=4 (max={max_small}) should allow larger offsets than "
            f"vid_frame_count=16 (max={max_default})"
        )

    def test_frame_count_none_uses_default(self, test_omnifall_video_dataset):
        """Test that frame_count=None uses vid_frame_count."""
        dataset = test_omnifall_video_dataset
        fps = 30.0

        offset1, _ = dataset.get_random_offset(length=512, target_interval=1, idx=0, fps=fps)

        offset2, _ = dataset.get_random_offset(
            length=512, target_interval=1, idx=0, fps=fps, frame_count=None
        )

        # Should produce identical results
        assert offset1 == offset2, "frame_count=None should use default vid_frame_count"


# ============================================================================
# Tests for decord-based video loading
# ============================================================================


class TestDecordVideoLoading:
    """Tests for the decord-based video loading implementation."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock OmnifallVideoDataset for testing."""
        dataset = MagicMock(spec=OmnifallVideoDataset)
        dataset.target_fps = 8.0
        dataset.vid_frame_count = 16
        dataset.data_fps = 30.0
        # Bind the actual method
        dataset.load_video_fast = (
            lambda path, idx, frame_count=None: OmnifallVideoDataset.load_video_fast(
                dataset, path, idx, frame_count
            )
        )
        # Use real get_random_offset
        from infreqact.data.dataset import GenericVideoDataset

        dataset.get_random_offset = GenericVideoDataset.get_random_offset.__get__(dataset)
        return dataset

    @pytest.mark.parametrize(
        "frame_count,expected_len", [(16, 16), (8, 8)], ids=["default_count", "short_segment"]
    )
    def test_load_video_fast_frame_count(
        self, mock_dataset, mock_video_reader_factory, frame_count, expected_len
    ):
        """Test that load_video_fast returns the expected number of frames."""
        # Use 224x224 frames to match expected shape
        mock_vr = mock_video_reader_factory(fps=30.0, total_frames=300, height=224, width=224)

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_vr):
            frames = mock_dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=frame_count)

        assert isinstance(frames, np.ndarray), "Should return an ndarray"
        assert frames.shape[0] == expected_len, (
            f"Expected {expected_len} frames, got {frames.shape[0]}"
        )

    def test_load_video_fast_frame_count_none_uses_default(
        self, mock_dataset, mock_video_reader_factory
    ):
        """Test that frame_count=None uses vid_frame_count."""
        mock_vr = mock_video_reader_factory(fps=30.0, total_frames=300, height=224, width=224)

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_vr):
            frames = mock_dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=None)

        assert isinstance(frames, np.ndarray), "Should return an ndarray"
        assert frames.shape[0] == mock_dataset.vid_frame_count, (
            f"Expected {mock_dataset.vid_frame_count} frames, got {frames.shape[0]}"
        )

    def test_frames_are_numpy_arrays(self, mock_dataset, mock_video_reader_factory):
        """Test that returned frames are numpy arrays with correct shape."""
        mock_vr = mock_video_reader_factory(fps=30.0, total_frames=300, height=224, width=224)

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_vr):
            frames = mock_dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=4)

        assert isinstance(frames, np.ndarray), "Should return an ndarray"
        assert frames.shape == (4, 224, 224, 3), f"Expected (4, 224, 224, 3), got {frames.shape}"


# ============================================================================
# Integration tests (require OMNIFALL_ROOT)
# ============================================================================


class TestVideoLoadingIntegration:
    """Integration tests that require actual video files."""

    def test_load_item_returns_correct_num_frames(self, test_omnifall_video_dataset):
        """Test that load_item returns frames matching compute_actual_frame_count."""
        dataset = test_omnifall_video_dataset

        # Get the first item
        item = dataset[0]

        # Check that num_frames is in the output
        assert "num_frames" in item, "num_frames should be in output"
        assert "video" in item, "video should be in output"

        # Check the video tensor shape
        video = item["video"]
        assert video.shape[0] == item["num_frames"], (
            f"Video tensor frames ({video.shape[0]}) should match num_frames ({item['num_frames']})"
        )

    def test_short_segment_has_fewer_frames(self, test_omnifall_video_dataset):
        """Test that short segments return fewer frames than vid_frame_count."""
        dataset = test_omnifall_video_dataset

        # Find a short segment (< 1.5 seconds)
        short_segment_idx = None
        for i, segment in enumerate(dataset.video_segments):
            if segment["duration"] < 1.5:
                short_segment_idx = i
                break

        if short_segment_idx is None:
            pytest.skip("No short segments found in test dataset")

        # Compute expected frame count
        segment = dataset.video_segments[short_segment_idx]
        expected_frames = dataset.compute_actual_frame_count(segment["duration"])

        # Load the item
        item = dataset[short_segment_idx]

        assert item["num_frames"] == expected_frames, (
            f"Short segment should have {expected_frames} frames, got {item['num_frames']}"
        )
        assert item["num_frames"] < dataset.vid_frame_count, (
            "Short segment should have fewer frames than vid_frame_count"
        )

    def test_long_segment_has_full_frames(self, test_omnifall_video_dataset):
        """Test that long segments return vid_frame_count frames."""
        dataset = test_omnifall_video_dataset

        # Find a long segment (>= 3 seconds)
        long_segment_idx = None
        for i, segment in enumerate(dataset.video_segments):
            if segment["duration"] >= 3.0:
                long_segment_idx = i
                break

        if long_segment_idx is None:
            pytest.skip("No long segments found in test dataset")

        # Load the item
        item = dataset[long_segment_idx]

        assert item["num_frames"] == dataset.vid_frame_count, (
            f"Long segment should have {dataset.vid_frame_count} frames, got {item['num_frames']}"
        )

    def test_no_frame_repetition_in_short_segment(self, test_omnifall_video_dataset):
        """Test that short segments don't have repeated frames at the end."""
        dataset = test_omnifall_video_dataset

        # Find a short segment
        short_segment_idx = None
        for i, segment in enumerate(dataset.video_segments):
            if 0.5 < segment["duration"] < 1.5:
                short_segment_idx = i
                break

        if short_segment_idx is None:
            pytest.skip("No suitable short segments found")

        item = dataset[short_segment_idx]
        video = item["video"].numpy()  # (T, C, H, W)

        # Check that consecutive frames are different
        # (at least some should be different if no repetition)
        n_frames = video.shape[0]
        if n_frames > 1:
            differences = []
            for i in range(n_frames - 1):
                diff = np.abs(video[i] - video[i + 1]).mean()
                differences.append(diff)

            # At least some frames should be different
            assert any(d > 0 for d in differences), (
                "All consecutive frames are identical, suggesting unwanted repetition"
            )

    def test_load_video_304(self, test_omnifall_video_dataset):
        """Test loading a specific video segment (index 304) to check for issues."""
        dataset = test_omnifall_video_dataset

        # Load the item at index 304
        try:
            item = dataset[304]
            video = item["video"]
            assert video.shape[0] == item["num_frames"], (
                f"Video frames ({video.shape[0]}) should match num_frames ({item['num_frames']})"
            )
        except Exception as e:
            pytest.fail(f"Failed to load video at index 304: {str(e)}")
