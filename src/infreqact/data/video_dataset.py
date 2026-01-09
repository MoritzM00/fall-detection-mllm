import logging
import random
from collections import OrderedDict

import pandas as pd
import torch

from infreqact.data.dataset import GenericVideoDataset
from infreqact.data.hf_utils import resolve_annotations_file, resolve_split_file

label2idx = {
    "walk": 0,
    "fall": 1,
    "fallen": 2,
    "sit_down": 3,
    "sitting": 4,
    "lie_down": 5,
    "lying": 6,
    "stand_up": 7,
    "standing": 8,
    "other": 9,
    # WanFall-specific classes (10-15)
    "kneel_down": 10,
    "kneeling": 11,
    "squat_down": 12,
    "squatting": 13,
    "crawl": 14,
    "jump": 15,
    "no_fall": -1,
}

idx2label = {v: k for k, v in label2idx.items()}


class OmnifallVideoDataset(GenericVideoDataset):
    """
    Video dataset for Omnifall that handles temporal segmentation annotations.
    Extends GenericVideoDataset to support start/end times and proper segment sampling.
    """

    def __init__(
        self,
        video_root,
        annotations_file,
        target_fps,
        vid_frame_count,
        split_root=None,
        dataset_name="UnnamedVideoDataset",
        mode="train",
        split="cs",
        data_fps=None,
        path_format="{video_root}/{video_path}{ext}",
        max_retries=10,
        fast=True,
        ext=".mp4",
        size=None,
        **kwargs,
    ):
        """
        Initialize Omnifall video dataset with temporal segmentation support.

        Args:
            video_root: Root directory for video files
            annotations_file: CSV file with temporal labels (path,label,start,end,subsect,cam)
            target_fps: Target FPS for frame sampling
            vid_frame_count: Number of frames to extract per segment
            split_root: Root directory for split files
            dataset_name: Name of the dataset
            mode: Dataset mode ("train", "val", "test", or "all")
            split: Split type ("cs" for cross-subject, "cv" for cross-view)
            data_fps: Original FPS of the videos (if known)
            path_format: Format string for video paths
            max_retries: Maximum retries for loading a video
            fast: Whether to use fast video loading
        """
        super().__init__(
            video_root=video_root,
            annotations_file=annotations_file,
            target_fps=target_fps,
            vid_frame_count=vid_frame_count,
            data_fps=data_fps,
            path_format=path_format,
            max_retries=max_retries,
            mode=mode,
            fast=fast,
        )
        self.dataset_name = dataset_name
        self.split = split
        self.mode = mode
        self.split_root = split_root
        self.ext = ext
        self.size = size

        logging.info(
            f"Initializing {self.dataset_name} dataset in {self.mode} mode with split {self.split}"
        )

        # Video segments with temporal annotations
        self.video_segments = []
        self.samples = OrderedDict()

        # Load split file if provided
        assert mode == "all" or split_root is not None, (
            "Split root must be provided unless mode is 'all'"
        )

        if mode != "all":
            # Resolve split file (supports local paths and HF dataset references)
            # Omnifall has structure: split_root/{split}/{dataset_name}/{mode}.csv
            self.split_file = resolve_split_file(
                split_root, mode, dataset_name=dataset_name, split_type=split
            )
            with open(self.split_file) as f:
                paths = sorted(list(f.read().splitlines()))
                for p in paths:
                    self.samples[p] = {"id": p}

        # Load temporal segmentation labels
        self._load_temporal_labels(annotations_file)

        # Set paths to segment indices
        self.paths = list(range(len(self.video_segments)))

    def _load_temporal_labels(self, annotations_file):
        """Load temporal segmentation labels from CSV and create segment index."""
        # Resolve annotations file (supports local paths and HF dataset references)
        resolved_path = resolve_annotations_file(annotations_file)
        df = pd.read_csv(resolved_path)

        for _, row in df.iterrows():
            path = row.iloc[0]  # Video path
            label = row.iloc[1]  # Label string
            start = float(row.iloc[2])  # Start time in seconds
            end = float(row.iloc[3])  # End time in seconds
            subject = row.iloc[4]  # Subsection
            cam = row.iloc[5]  # Camera

            # Convert label to index
            label_str = idx2label.get(label)

            # Only process videos that are in our split
            if path in self.samples or self.mode == "all":
                if path not in self.samples:
                    self.samples[path] = {"id": path}

                if "segments" not in self.samples[path]:
                    self.samples[path]["segments"] = []

                segment = {
                    "video_path": path,
                    "label": label,
                    "label_str": label_str,
                    "start": start,
                    "end": end,
                    "subsect": subject,
                    "cam": cam,
                    "duration": end - start,
                }

                self.samples[path]["segments"].append(segment)
                self.video_segments.append(segment)

        # Sort segments by video path and start time for consistency
        self.video_segments.sort(key=lambda x: (x["video_path"], x["start"]))

        logging.info(
            f"Loaded {len(self.video_segments)} segments from {len(self.samples)} videos for {self.mode} split"
        )

    def __len__(self):
        return len(self.video_segments)

    def _id2label(self, idx):
        """Get segment info and label for given index."""
        segment = self.video_segments[idx]
        return segment, segment["label"]

    def get_random_offset(self, length, target_interval, idx, fps, start=0):
        """
        Get random offset for temporal segment sampling.
        Ensures we sample within the annotated segment boundaries.
        """
        segment = self.video_segments[idx]
        segment_start_frame = int(segment["start"] * fps)
        segment_end_frame = int(segment["end"] * fps)
        segment_frames = segment_end_frame - segment_start_frame

        required_frames = self.vid_frame_count * target_interval

        if segment_frames <= required_frames:
            # Segment is too short, start from beginning of segment
            return segment_start_frame
        else:
            # Random offset within the segment
            max_offset = segment_frames - required_frames
            random_offset = random.randint(0, int(max_offset))
            return segment_start_frame + random_offset

    def load_item(self, idx):
        """Load video segment with temporal boundaries."""
        segment, label = self._id2label(idx)

        # Construct video path
        video_path = self.path_format.format(
            video_root=self.video_root, video_path=segment["video_path"], ext=self.ext
        )

        # Load frames from the video
        frames = self.load_video(video_path, idx)

        # Transform frames
        inputs = self.transform_frames(frames)

        # Add segment information
        inputs.update(
            {
                "label": label,
                "label_str": segment["label_str"],
                "video_path": segment["video_path"],
                "start_time": segment["start"],
                "end_time": segment["end"],
                "segment_duration": segment["duration"],
                "dataset": self.dataset_name,
            }
        )

        return inputs

    @property
    def targets(self):
        """Return all class labels for segments in this dataset."""
        return torch.tensor([segment["label"] for segment in self.video_segments])

    def __repr__(self):
        return (
            f"OmnifallVideoDataset(name='{self.dataset_name}', "
            f"split='{self.split}', mode='{self.mode}', "
            f"videos={len(self.samples)}, segments={len(self.video_segments)})"
        )
