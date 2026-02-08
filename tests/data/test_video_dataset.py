import os

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
        "offset": "random",
    }
    dataset = OmnifallVideoDataset(**config)
    return dataset


def test_random_offset_seed(test_omnifall_video_dataset):
    """Test that random offset is consistent across runs with the same seed."""
    dataset = test_omnifall_video_dataset

    offsets_run1 = []
    offsets_run2 = []

    for idx in range(len(dataset)):
        segment = dataset.video_segments[idx]
        start_sec = segment["start"]
        end_sec = segment["end"]
        video_path = dataset.format_path(segment["video_path"])

        # Load video twice with the same seed
        frames1 = dataset.load_video(video_path, start_sec=start_sec, end_sec=end_sec)
        offsets_run1.append(frames1)

        frames2 = dataset.load_video(video_path, start_sec=start_sec, end_sec=end_sec)
        offsets_run2.append(frames2)
        if idx > 2:
            break  # Limit to first 3 segments for speed

    # Verify that offsets are the same across both runs
    for f1, f2 in zip(offsets_run1, offsets_run2):
        assert (f1 == f2).all(), "Frame sequences differ between runs with the same seed."
