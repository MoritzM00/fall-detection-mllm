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
