"""Shared test fixtures for data tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from infreqact.data.dataset import GenericVideoDataset


@pytest.fixture
def mock_video_reader_factory():
    """Factory fixture to create mock VideoReader instances with custom parameters.

    Returns:
        Callable that creates a mock VideoReader with specified parameters
    """

    def _create_mock(fps=30.0, total_frames=300, height=480, width=640):
        """Create a mock VideoReader.

        Args:
            fps: Video frame rate
            total_frames: Total number of frames in video
            height: Frame height in pixels
            width: Frame width in pixels

        Returns:
            MagicMock configured to simulate decord VideoReader
        """
        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = fps
        mock_vr.__len__ = MagicMock(return_value=total_frames)

        def mock_get_batch(indices):
            """Simulate decord get_batch returning RGB frames."""
            frames = np.random.randint(
                0, 255, size=(len(indices), height, width, 3), dtype=np.uint8
            )
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

        mock_vr.get_batch = mock_get_batch
        return mock_vr

    return _create_mock


@pytest.fixture
def mock_video_reader(mock_video_reader_factory):
    """Create a default mock VideoReader (30fps, 300 frames, 480x640)."""
    return mock_video_reader_factory()


@pytest.fixture
def generic_dataset():
    """Create a minimal GenericVideoDataset for testing.

    Returns:
        GenericVideoDataset with patched __init__ and default test parameters
    """
    with patch.object(GenericVideoDataset, "__init__", lambda x: None):
        ds = GenericVideoDataset.__new__(GenericVideoDataset)
        ds.vid_frame_count = 16
        ds.target_fps = 8.0
        ds.data_fps = 30.0
        return ds


@pytest.fixture
def make_capture_get_batch():
    """Factory to create a get_batch capturing function.

    Returns:
        Callable that returns (capture_fn, indices_list) tuple
    """

    def _make_capture(height=480, width=640):
        """Create a capturing get_batch function.

        Args:
            height: Frame height
            width: Frame width

        Returns:
            Tuple of (capture_function, indices_list)
        """
        captured_indices = []

        def capture_get_batch(indices):
            """Capture indices and return mock frames."""
            captured_indices.extend(indices)
            frames = np.zeros((len(indices), height, width, 3), dtype=np.uint8)
            mock_result = MagicMock()
            mock_result.asnumpy.return_value = frames
            return mock_result

        return capture_get_batch, captured_indices

    return _make_capture
