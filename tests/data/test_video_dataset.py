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
        )
        for i in range(n_offsets)
    ]

    offsets2 = [
        dataset.get_random_offset(
            length=frame_count, target_interval=target_interval, idx=i, fps=fps
        )
        for i in range(n_offsets)
    ]
    assert offsets1 == offsets2, "Random offsets differ between runs with the same seed."

    dataset.seed = 42
    offsets3 = [
        dataset.get_random_offset(
            length=frame_count, target_interval=target_interval, idx=i, fps=fps
        )
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
        # Bind the actual method to our mock
        dataset.compute_actual_frame_count = (
            lambda duration: OmnifallVideoDataset.compute_actual_frame_count(dataset, duration)
        )
        return dataset

    def test_long_segment_returns_vid_frame_count(self, mock_dataset):
        """A 4-second segment at 8 fps can fit 33 frames, should return 16."""
        result = mock_dataset.compute_actual_frame_count(4.0)
        assert result == 16, f"Expected 16 frames for 4s segment, got {result}"

    def test_medium_segment_returns_vid_frame_count(self, mock_dataset):
        """A 2-second segment at 8 fps can fit 17 frames, should return 16."""
        result = mock_dataset.compute_actual_frame_count(2.0)
        assert result == 16, f"Expected 16 frames for 2s segment, got {result}"

    def test_short_segment_returns_fewer_frames(self, mock_dataset):
        """A 1-second segment at 8 fps can fit 9 frames, should return 9."""
        result = mock_dataset.compute_actual_frame_count(1.0)
        # available = floor(1.0 * 8) + 1 = 9
        assert result == 9, f"Expected 9 frames for 1s segment, got {result}"

    def test_very_short_segment(self, mock_dataset):
        """A 0.5-second segment at 8 fps can fit 5 frames, should return 5."""
        result = mock_dataset.compute_actual_frame_count(0.5)
        # available = floor(0.5 * 8) + 1 = 5
        assert result == 5, f"Expected 5 frames for 0.5s segment, got {result}"

    def test_minimal_segment(self, mock_dataset):
        """A very short segment should return at least 1 frame."""
        result = mock_dataset.compute_actual_frame_count(0.1)
        # available = floor(0.1 * 8) + 1 = 1
        assert result == 1, f"Expected 1 frame for 0.1s segment, got {result}"

    def test_exact_boundary(self, mock_dataset):
        """Test segment duration that exactly fits vid_frame_count frames."""
        # For 16 frames at 8 fps, need (16-1)/8 = 1.875 seconds
        # But formula is: available = floor(duration * fps) + 1
        # So we need floor(duration * 8) + 1 >= 16, meaning floor(duration * 8) >= 15
        # duration >= 15/8 = 1.875
        result = mock_dataset.compute_actual_frame_count(1.875)
        # available = floor(1.875 * 8) + 1 = floor(15) + 1 = 16
        assert result == 16, f"Expected 16 frames for 1.875s segment, got {result}"

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

        # With default vid_frame_count=16
        offset1 = dataset.get_random_offset(length=512, target_interval=1, idx=0, fps=fps)

        # With explicit frame_count=8 (shorter, more room for offset)
        offset2 = dataset.get_random_offset(
            length=512, target_interval=1, idx=0, fps=fps, frame_count=8
        )

        # Both should be valid (non-negative)
        assert offset1 >= 0, "Offset should be non-negative"
        assert offset2 >= 0, "Offset should be non-negative"

    def test_frame_count_none_uses_default(self, test_omnifall_video_dataset):
        """Test that frame_count=None uses vid_frame_count."""
        dataset = test_omnifall_video_dataset
        fps = 30.0

        offset1 = dataset.get_random_offset(length=512, target_interval=1, idx=0, fps=fps)

        offset2 = dataset.get_random_offset(
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
    def mock_video_reader(self):
        """Create a mock VideoReader."""
        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)  # 10 seconds of video

        # Mock get_batch to return fake frames
        def mock_get_batch(indices):
            frames = np.random.randint(0, 255, size=(len(indices), 224, 224, 3), dtype=np.uint8)
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

        mock_vr.get_batch = mock_get_batch
        return mock_vr

    def test_load_video_fast_returns_correct_frame_count(
        self, test_omnifall_video_dataset, mock_video_reader
    ):
        """Test that load_video_fast returns the expected number of frames."""
        dataset = test_omnifall_video_dataset

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=16)

        assert len(frames) == 16, f"Expected 16 frames, got {len(frames)}"

    def test_load_video_fast_short_segment(self, test_omnifall_video_dataset, mock_video_reader):
        """Test that load_video_fast respects frame_count for short segments."""
        dataset = test_omnifall_video_dataset

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            # Request only 8 frames (simulating a short segment)
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=8)

        assert len(frames) == 8, f"Expected 8 frames for short segment, got {len(frames)}"

    def test_load_video_fast_frame_count_none_uses_default(
        self, test_omnifall_video_dataset, mock_video_reader
    ):
        """Test that frame_count=None uses vid_frame_count."""
        dataset = test_omnifall_video_dataset

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=None)

        assert len(frames) == dataset.vid_frame_count, (
            f"Expected {dataset.vid_frame_count} frames, got {len(frames)}"
        )

    def test_frames_are_numpy_arrays(self, test_omnifall_video_dataset, mock_video_reader):
        """Test that returned frames are numpy arrays with correct shape."""
        dataset = test_omnifall_video_dataset

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=4)

        for i, frame in enumerate(frames):
            assert isinstance(frame, np.ndarray), f"Frame {i} should be numpy array"
            assert frame.shape == (224, 224, 3), f"Frame {i} has wrong shape: {frame.shape}"


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
