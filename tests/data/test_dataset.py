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
        # With 100 frames and vid_frame_count=16, max offset is 100-16=84
        assert 0 <= offset <= 84, f"Offset {offset} out of expected range [0, 84]"

    def test_frame_count_parameter(self, dataset):
        """Test that frame_count parameter is used when provided."""
        offset = dataset.get_random_offset(
            length=100, target_interval=1, idx=0, fps=30.0, frame_count=8
        )
        # With 100 frames and frame_count=8, max offset is 100-8=92
        assert 0 <= offset <= 92, f"Offset {offset} out of expected range [0, 92]"

    def test_returns_zero_when_too_short(self, dataset):
        """Test that offset is 0 when video is too short."""
        offset = dataset.get_random_offset(length=10, target_interval=1, idx=0, fps=30.0)
        # With only 10 frames and needing 16, should return 0
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


class TestDecordVideoLoaderFast:
    """Tests for the decord-based load_video_fast method."""

    @pytest.fixture
    def dataset(self):
        """Create a minimal GenericVideoDataset for testing."""
        with patch.object(GenericVideoDataset, "__init__", lambda x: None):
            ds = GenericVideoDataset.__new__(GenericVideoDataset)
            ds.vid_frame_count = 16
            ds.target_fps = 8.0
            ds.data_fps = 30.0
            # Set up the get_random_offset method
            ds.get_random_offset = lambda length, interval, idx, fps, start=0, frame_count=None: 0
            return ds

    @pytest.fixture
    def mock_video_reader(self):
        """Create a mock VideoReader that simulates decord behavior."""
        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)  # 10 seconds at 30fps

        def mock_get_batch(indices):
            # Return fake RGB frames
            frames = np.random.randint(0, 255, size=(len(indices), 480, 640, 3), dtype=np.uint8)
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

        mock_vr.get_batch = mock_get_batch
        return mock_vr

    def test_returns_list_of_frames(self, dataset, mock_video_reader):
        """Test that load_video_fast returns a list of numpy arrays."""
        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0)

        assert isinstance(frames, list), "Should return a list"
        assert len(frames) == 16, f"Should return 16 frames, got {len(frames)}"
        for frame in frames:
            assert isinstance(frame, np.ndarray), "Each frame should be numpy array"

    def test_frame_count_parameter(self, dataset, mock_video_reader):
        """Test that frame_count parameter controls output size."""
        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=8)

        assert len(frames) == 8, f"Should return 8 frames, got {len(frames)}"

    def test_frame_count_none_uses_default(self, dataset, mock_video_reader):
        """Test that frame_count=None uses vid_frame_count."""
        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=None)

        assert len(frames) == dataset.vid_frame_count

    def test_frame_indices_calculation(self, dataset, mock_video_reader):
        """Test that frame indices are calculated correctly."""
        captured_indices = []

        def capture_get_batch(indices):
            captured_indices.extend(indices)
            frames = np.zeros((len(indices), 480, 640, 3), dtype=np.uint8)
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

        mock_video_reader.get_batch = capture_get_batch

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            dataset.load_video_fast("/fake/path.mp4", idx=0, frame_count=4)

        # With fps=30, target_fps=8, frame spacing = 30/8 = 3.75
        # Frame indices: [0, 3, 7, 11] (int(0 + n * 3.75) for n in range(4))
        expected = [int(0 + n * 30.0 / 8.0) for n in range(4)]
        assert captured_indices == expected, f"Expected {expected}, got {captured_indices}"

    def test_indices_clamped_to_valid_range(self, dataset, mock_video_reader):
        """Test that frame indices are clamped to valid range."""
        captured_indices = []

        # Short video with only 20 frames
        mock_video_reader.__len__ = MagicMock(return_value=20)

        def capture_get_batch(indices):
            captured_indices.extend(indices)
            frames = np.zeros((len(indices), 480, 640, 3), dtype=np.uint8)
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

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
    def dataset(self):
        """Create a minimal GenericVideoDataset for testing."""
        with patch.object(GenericVideoDataset, "__init__", lambda x: None):
            ds = GenericVideoDataset.__new__(GenericVideoDataset)
            ds.vid_frame_count = 16
            ds.target_fps = 8.0
            ds.data_fps = 30.0
            ds.get_random_offset = lambda length, interval, idx, fps, start=0, frame_count=None: 0
            return ds

    @pytest.fixture
    def mock_video_reader(self):
        """Create a mock VideoReader."""
        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)

        def mock_get_batch(indices):
            frames = np.random.randint(0, 255, size=(len(indices), 480, 640, 3), dtype=np.uint8)
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

        mock_vr.get_batch = mock_get_batch
        return mock_vr

    def test_samples_at_interval(self, dataset, mock_video_reader):
        """Test that slow loader samples at correct interval."""
        captured_indices = []

        def capture_get_batch(indices):
            captured_indices.extend(indices)
            frames = np.zeros((len(indices), 480, 640, 3), dtype=np.uint8)
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

        mock_video_reader.get_batch = capture_get_batch

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            dataset.load_video_slow("/fake/path.mp4", idx=0)

        # With fps=30, target_fps=8, interval = round(30/8) = 4
        # Should sample every 4th frame: 0, 4, 8, 12, ...
        expected_interval = 4
        for i in range(1, len(captured_indices)):
            diff = captured_indices[i] - captured_indices[i - 1]
            assert diff == expected_interval, f"Expected interval {expected_interval}, got {diff}"

    def test_frame_count_parameter(self, dataset, mock_video_reader):
        """Test that frame_count parameter is respected."""
        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_slow("/fake/path.mp4", idx=0, frame_count=8)

        assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"

    def test_frame_count_none_loads_all(self, dataset, mock_video_reader):
        """Test that frame_count=None loads all sampled frames."""
        dataset.vid_frame_count = None  # Override to None

        with patch("infreqact.data.dataset.VideoReader", return_value=mock_video_reader):
            frames = dataset.load_video_slow("/fake/path.mp4", idx=0, frame_count=None)

        # Should load all frames at the sampling interval
        # 300 frames / 4 interval = 75 frames
        expected = len(list(range(0, 300, 4)))
        assert len(frames) == expected, f"Expected {expected} frames, got {len(frames)}"
