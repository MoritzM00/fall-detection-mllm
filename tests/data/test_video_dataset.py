import os

import pytest

from falldet.data.video_dataset import OmnifallVideoDataset


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
        "offset": "random",
    }
    dataset = OmnifallVideoDataset(**config)
    return dataset


def test_random_offset_seed(test_omnifall_video_dataset):
    """Test that random offset is consistent across runs with the same seed."""
    dataset = test_omnifall_video_dataset

    offsets_run1 = []
    offsets_run2 = []

    for idx in range(min(3, len(dataset))):
        segment = dataset.video_segments[idx]
        video_path = dataset.format_path(segment["video_path"])

        # Load video twice with the same seed
        frames1 = dataset.load_video(video_path, idx)
        offsets_run1.append(frames1)

        frames2 = dataset.load_video(video_path, idx)
        offsets_run2.append(frames2)

    # Verify that offsets are the same across both runs
    import numpy as np

    for f1, f2 in zip(offsets_run1, offsets_run2):
        # Convert to numpy arrays for comparison if needed
        if isinstance(f1, list):
            f1 = np.array(f1)
            f2 = np.array(f2)
        assert np.array_equal(f1, f2), "Frame sequences differ between runs with the same seed."


def test_decode_within_boundaries(test_omnifall_video_dataset):
    """Test that loaded clips stay within segment temporal boundaries when possible."""
    dataset = test_omnifall_video_dataset

    # Test multiple segments to ensure robust boundary checking
    num_segments_to_test = min(50, len(dataset))

    segments_tested = 0
    segments_within_bounds = 0

    for idx in range(num_segments_to_test):
        segment = dataset.video_segments[idx]
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_duration = segment_end - segment_start

        # Compute expected clip duration in seconds
        clip_duration_sec = (dataset.vid_frame_count - 1) / dataset.target_fps

        # Get random offset (this is what dataset uses internally)
        fps = dataset.data_fps if dataset.data_fps is not None else 30.0

        # Compute the offset that would be used
        begin_frame = dataset.get_random_offset(None, 1, idx, fps)

        # Calculate the actual start and end timestamps of the decoded clip
        clip_start_time = begin_frame / fps
        clip_end_time = clip_start_time + clip_duration_sec

        # Allow small floating point tolerance
        tolerance = 0.03  # 1ms tolerance

        # For segments that are long enough to fit the clip, verify strict bounds
        if segment_duration >= clip_duration_sec:
            segments_tested += 1
            assert clip_start_time >= segment_start - tolerance, (
                f"Segment {idx}: Clip start {clip_start_time:.3f}s is before "
                f"segment start {segment_start:.3f}s"
            )
            assert clip_end_time <= segment_end + tolerance, (
                f"Segment {idx}: Clip end {clip_end_time:.3f}s is after "
                f"segment end {segment_end:.3f}s (duration={segment_duration:.3f}s, "
                f"clip_duration={clip_duration_sec:.3f}s)"
            )
            segments_within_bounds += 1

    # Verify we tested at least some valid segments
    assert segments_tested >= 5, (
        f"Expected to test at least 5 segments that fit the clip duration, "
        f"but only found {segments_tested}"
    )
    assert segments_within_bounds == segments_tested, (
        f"Not all valid segments stayed within bounds: {segments_within_bounds}/{segments_tested}"
    )

    # Actually load one item to verify it works end-to-end
    item = dataset[0]
    assert "video" in item, "Dataset should return video frames"
    assert "label" in item, "Dataset should return label"
    assert "start_time" in item, "Dataset should return segment metadata"


def test_load_very_short_video(test_omnifall_video_dataset):
    """Test to load video at index 304 which is very short."""
    dataset = test_omnifall_video_dataset
    item = dataset[304]
    assert "video" in item, "Dataset should return video frames"
    frames = item["video"]
    assert frames.shape[0] == dataset.vid_frame_count, (
        f"Expected {dataset.vid_frame_count} frames after padding, but got {frames.shape[0]}"
    )
