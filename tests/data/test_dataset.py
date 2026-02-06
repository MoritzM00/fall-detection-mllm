"""Tests for the GenericVideoDataset base class and decord video loading."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from infreqact.data.dataset import GenericVideoDataset


class TestGenericVideoDatasetGetRandomOffset:
    """Tests for the base class get_random_offset method."""

    @pytest.fixture
    def dataset(self):
        """Create a minimal GenericVideoDataset for testing."""
        with patch.object(GenericVideoDataset, "__init__", lambda x: None):
            ds = GenericVideoDataset.__new__(GenericVideoDataset)
            ds.vid_frame_count = 16
            ds.target_fps = 8.0
            return ds

    def test_default_uses_vid_frame_count(self, dataset):
        """Test that default behavior uses vid_frame_count."""
        offset = dataset.get_random_offset(length=100, target_interval=1, idx=0, fps=30.0)
        # With 100 frames and vid_frame_count=16, required_span=(16-1)*1=15, max_offset=100-1-15=84
        assert 0 <= offset <= 84, f"Offset {offset} out of expected range [0, 84]"

    def test_frame_count_parameter(self, dataset):
        """Test that frame_count parameter is used when provided."""
        offset = dataset.get_random_offset(
            length=100, target_interval=1, idx=0, fps=30.0, frame_count=8
        )
        # With 100 frames and frame_count=8, required_span=(8-1)*1=7, max_offset=100-1-7=92
        assert 0 <= offset <= 92, f"Offset {offset} out of expected range [0, 92]"

    def test_returns_zero_when_too_short(self, dataset):
        """Test that offset is 0 when video is too short."""
        offset = dataset.get_random_offset(length=10, target_interval=1, idx=0, fps=30.0)
        # With only 10 frames and needing 16, required_span=15, max_offset=10-1-15=-6, should return 0
        assert offset == 0, f"Expected 0 for short video, got {offset}"

    def test_frame_count_none_uses_default(self, dataset):
        """Test that frame_count=None behaves like not passing it."""
        offset1 = dataset.get_random_offset(length=100, target_interval=1, idx=0, fps=30.0)
        offset2 = dataset.get_random_offset(
            length=100, target_interval=1, idx=0, fps=30.0, frame_count=None
        )
        # Both should work (they may differ due to randomness, but both should be valid)
        assert 0 <= offset1 <= 84
        assert 0 <= offset2 <= 84

    def test_fractional_target_interval(self, dataset):
        """Test with fractional target_interval (e.g., fps/target_fps)."""
        # fps=30, target_fps=8 → interval = 3.75
        offset = dataset.get_random_offset(
            length=200, target_interval=3.75, idx=0, fps=30.0, frame_count=16
        )
        # required_span = (16-1) * 3.75 = 56.25
        # max_offset = int(200 - 1 - 56.25) = int(142.75) = 142
        assert 0 <= offset <= 142, f"Offset {offset} out of expected range [0, 142]"

    def test_fractional_interval_prevents_out_of_bounds(self, dataset):
        """Test that fractional intervals don't produce out-of-bounds indices."""
        # Run 100 iterations to catch any edge cases
        for i in range(100):
            offset = dataset.get_random_offset(
                length=200, target_interval=3.75, idx=i, fps=30.0, frame_count=16
            )
            # Last frame index: offset + (16-1) * 3.75
            last_index = int(offset + 15 * 3.75)
            assert last_index < 200, (
                f"Last frame index {last_index} >= total_frames 200 (offset={offset}, iter={i})"
            )

    def test_frame_count_zero(self, dataset):
        """Test that frame_count=0 returns offset 0."""
        offset = dataset.get_random_offset(
            length=100, target_interval=1, idx=0, fps=30.0, frame_count=0
        )
        assert offset == 0, f"Expected 0 for frame_count=0, got {offset}"

    def test_frame_count_one(self, dataset):
        """Test that frame_count=1 allows any offset in valid range."""
        offset = dataset.get_random_offset(
            length=100, target_interval=1, idx=0, fps=30.0, frame_count=1
        )
        # required_span = (1-1) * 1 = 0, max_offset = 100 - 1 - 0 = 99
        assert 0 <= offset <= 99, f"Offset {offset} out of expected range [0, 99]"


class TestDecordVideoLoaderFast:
    """Tests for the decord-based load_video_fast method."""

    @pytest.fixture
    def dataset(self, generic_dataset):
        """Use generic_dataset fixture and add get_random_offset mock."""
        generic_dataset.get_random_offset = (
            lambda length, interval, idx, fps, start=0, frame_count=None: 0
        )
        return generic_dataset

    @pytest.mark.parametrize(
        "frame_count,expected_len", [(None, 16), (8, 8)], ids=["default", "explicit"]
    )
    def test_frame_count(self, dataset, mock_video_reader, frame_count, expected_len):
        """Test that frame_count parameter controls output size."""
        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=frame_count)

        assert isinstance(frames, list), "Should return a list"
        assert len(frames) == expected_len, (
            f"Should return {expected_len} frames, got {len(frames)}"
        )
        for frame in frames:
            assert isinstance(frame, np.ndarray), "Each frame should be numpy array"

    def test_frame_indices_calculation(self, dataset, mock_video_reader, make_capture_get_batch):
        """Test that frame indices are calculated correctly."""
        capture_get_batch, captured_indices = make_capture_get_batch()
        mock_video_reader.get_batch = capture_get_batch

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=4)

        # With fps=30, target_fps=8, frame spacing = 30/8 = 3.75
        # Frame indices: [0, 3, 7, 11] (int(0 + n * 3.75) for n in range(4))
        expected = [int(0 + n * 30.0 / 8.0) for n in range(4)]
        assert captured_indices == expected, f"Expected {expected}, got {captured_indices}"

    def test_indices_clamped_to_valid_range(
        self, dataset, mock_video_reader, make_capture_get_batch
    ):
        """Test that frame indices are clamped to valid range."""
        capture_get_batch, captured_indices = make_capture_get_batch()

        # Short video with only 20 frames
        mock_video_reader.__len__ = MagicMock(return_value=20)
        mock_video_reader.get_batch = capture_get_batch

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=16)

        # All indices should be <= 19 (max valid index)
        assert all(i <= 19 for i in captured_indices), (
            f"Indices should be clamped to max 19, got {captured_indices}"
        )

    def test_raises_on_empty_video(self, dataset):
        """Test that empty video raises appropriate error."""
        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=0)

        with (
            patch("infreqact.data.dataset.VideoReader", return_value=mock_vr),
            pytest.raises(RuntimeError, match="Failed to process video"),
        ):
            dataset.load_video_fast("/fake/path.mp4", idx=0)


class TestDecordVideoLoaderSlow:
    """Tests for the decord-based load_video_slow method."""

    @pytest.fixture
    def dataset(self, generic_dataset):
        """Use generic_dataset fixture and add get_random_offset mock."""
        generic_dataset.get_random_offset = (
            lambda length, interval, idx, fps, start=0, frame_count=None: 0
        )
        return generic_dataset

    def test_samples_at_interval(self, dataset, mock_video_reader, make_capture_get_batch):
        """Test that slow loader samples at correct interval."""
        capture_get_batch, captured_indices = make_capture_get_batch()
        mock_video_reader.get_batch = capture_get_batch

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            dataset.load_video_slow("/fake/path.mp4", idx=0)

        # With fps=30, target_fps=8, interval = round(30/8) = 4
        # Should sample every 4th frame: 0, 4, 8, 12, ...
        expected_interval = 4
        for i in range(1, len(captured_indices)):
            diff = captured_indices[i] - captured_indices[i - 1]
            assert diff == expected_interval, f"Expected interval {expected_interval}, got {diff}"

    @pytest.mark.parametrize(
        "frame_count,expected_len", [(8, 8), (None, 75)], ids=["with_count", "load_all"]
    )
    def test_frame_count(self, dataset, mock_video_reader, frame_count, expected_len):
        """Test that frame_count parameter controls output size."""
        if frame_count is None:
            dataset.vid_frame_count = None  # Override to None for load_all test

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_slow("/fake/path.mp4", idx=0, frame_count=frame_count)

        # For load_all: 300 frames / 4 interval = 75 frames
        assert len(frames) == expected_len, f"Expected {expected_len} frames, got {len(frames)}"

    def test_load_video_slow_cycling(self, dataset, mock_video_reader_factory):
        """Test that short videos cycle frames correctly."""
        # Create a short video with only 40 frames (5 seconds at 8fps = 5 sampled frames)
        short_vr = mock_video_reader_factory(fps=30.0, total_frames=40)

        with patch("infreqact.data.dataset.VideoReader", return_value=short_vr):
            frames = dataset.load_video_slow("/fake/path.mp4", idx=0, frame_count=16)

        # Should cycle the 10 sampled frames (40 / 4 interval = 10) to get 16 frames
        assert len(frames) == 16, f"Expected 16 frames (with cycling), got {len(frames)}"

    def test_load_video_slow_random_window(self, dataset, mock_video_reader_factory):
        """Test that random window selection works with fixed interval."""
        # Use real get_random_offset instead of mocked version
        from infreqact.data.dataset import GenericVideoDataset

        dataset.get_random_offset = GenericVideoDataset.get_random_offset.__get__(dataset)

        # Create video with enough frames for random selection
        long_vr = mock_video_reader_factory(fps=30.0, total_frames=300)

        with patch("infreqact.data.dataset.VideoReader", return_value=long_vr):
            # 300 frames / 4 interval = 75 sampled frames, need 16, so 59 possible offsets
            frames = dataset.load_video_slow("/fake/path.mp4", idx=0, frame_count=16)

        assert len(frames) == 16, f"Expected 16 frames, got {len(frames)}"
