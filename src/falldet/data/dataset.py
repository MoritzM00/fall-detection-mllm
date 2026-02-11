import csv
import logging
import time

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchvision import tv_tensors

from falldet.inference import load_video_clip

logger = logging.getLogger(__name__)


class GenericVideoDataset(Dataset):
    def __init__(
        self,
        video_root,
        annotations_file,
        target_fps,
        vid_frame_count,
        data_fps=None,
        path_format="{video_root}/{filename}.mp4",
        max_retries=10,
        mode="train",
        fast=True,
        size=None,
        max_size=None,
        offset="random",
        seed=0,
    ):
        self.video_root = video_root
        self.slow_video_file = annotations_file.replace(".csv", "_slow.csv")
        self.target_fps = target_fps
        self.vid_frame_count = vid_frame_count
        self.data_fps = data_fps
        self.path_format = path_format
        self.max_retries = max_retries
        self.mode = mode
        self.annotations = {}
        self.size = size
        self.max_size = max_size
        self.offset = offset
        self.seed = seed

    def load_annotations(self, annotations_file):
        # call manually after init if needed
        annotations = {}
        with open(annotations_file) as file:
            reader = csv.reader(file)
            for row in reader:
                annotations[row[0]] = int(row[1])
        self.annotations = annotations
        self.paths = list(sorted(self.annotations.keys()))

    def __len__(self):
        return len(self.paths)

    def _id2label(self, idx):
        path = self.paths[idx]
        label = self.annotations[path]
        return path, label

    def load_item(self, idx):
        path, label = self._id2label(idx)

        # Measure video IO time
        video_io_start = time.time()
        video_path = self.path_format.format(video_root=self.video_root, filename=path)
        frames = self.load_video(video_path)
        video_io_end = time.time()

        # Measure video processing time
        video_processing_start = time.time()
        inputs = self.transform_frames(frames)
        video_processing_end = time.time()

        inputs.update(
            {
                "label": label,
                "video_io_time": video_io_end - video_io_start,
                "video_processing_time": video_processing_end - video_processing_start,
            }
        )
        return inputs

    def __getitem__(self, idx):
        retries = 0
        while retries < self.max_retries:
            try:
                return self.load_item(idx)

            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    logging.error(f"Error loading video at index {idx}: {str(e)}")

        raise RuntimeError(f"Failed to load a valid video after {self.max_retries} attempts")

    def transform_frames(self, frames):
        # frames is a ndarrays (T, H, W, C)
        # Stack and convert to tensor: (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        frames = tv_tensors.Video(frames)

        if self.size is not None:
            transform = v2.Compose(
                [
                    v2.Resize(
                        self.size,
                        max_size=self.max_size,
                        antialias=True,
                        interpolation=v2.InterpolationMode.BILINEAR,
                    ),
                    v2.CenterCrop(self.size),
                    v2.ToDtype(torch.float32),
                ]
            )
            frames = transform(frames)

        return {"video": frames}

    def load_video(
        self, path: str, start_sec: float = 0.0, end_sec: float = float("inf")
    ) -> np.ndarray:
        frames, _ = load_video_clip(
            path,
            start_sec=start_sec,
            end_sec=end_sec,
            offset=self.offset,
            target_fps=self.target_fps,
            max_frames=self.vid_frame_count,
            seed=self.seed,
        )
        return frames
