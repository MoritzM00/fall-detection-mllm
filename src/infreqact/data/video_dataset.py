import logging
import math
from collections import OrderedDict

import numpy as np
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
        max_size=None,
        seed=0,
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
            size=size,
            max_size=max_size,
        )
        self.seed = seed
        self.dataset_name = dataset_name
        self.split = split
        self.split_root = split_root
        self.ext = ext

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

    def format_path(self, rel_path):
        """Format relative video path to full path."""
        return self.path_format.format(
            video_root=self.video_root, video_path=rel_path, ext=self.ext
        )

    def compute_actual_frame_count(self, segment_duration_sec):
        """
        Compute actual frames to extract based on segment duration.
        For short segments, returns fewer frames to avoid repetition.

        Formula: available_frames = floor(segment_duration * target_fps) + 1
        (The +1 accounts for the frame at t=0)

        Args:
            segment_duration_sec: Duration of the segment in seconds

        Returns:
            Number of frames to extract, or None to load all frames
        """
        if self.vid_frame_count is None:
            return None  # Load all frames

        # How many frames can fit in the actual segment at target_fps
        available_frames = max(1, int(segment_duration_sec * self.target_fps) + 1)

        # Return the minimum of desired and available
        return min(self.vid_frame_count, available_frames)

    def get_random_offset(self, length, target_interval, idx, fps, start=0, frame_count=None):
        """
        Get random offset for temporal segment sampling.
        Ensures we sample within the annotated segment boundaries.

        Uses index-based seeding for reproducibility across DataLoader workers.

        Args:
            length: Total number of frames in video (used to constrain offset when annotations are inaccurate)
            target_interval: Target interval (unused, kept for API compatibility)
            idx: Sample index for segment lookup and random seeding
            fps: Video frame rate
            start: Start offset (unused, kept for API compatibility)
            frame_count: Number of frames to extract (defaults to self.vid_frame_count)
        """
        segment = self.video_segments[idx]
        # Use provided frame_count or fall back to default
        num_frames = frame_count if frame_count is not None else self.vid_frame_count

        # IMPORTANT:
        # `load_video_fast()` interprets this return value as a *begin_frame* in the
        # original video FPS domain, then samples `num_frames` timestamps spaced
        # by 1/self.target_fps seconds:
        #   ts_n = begin_frame/fps + n/self.target_fps
        # Therefore, to guarantee we do NOT sample past the segment boundary, we must
        # constrain begin_frame such that the last timestamp stays within [start, end).
        segment_start_sec = float(segment["start"])
        segment_end_sec = float(segment["end"])

        # Convert start bound to a safe frame index (inclusive).
        segment_start_frame = int(math.ceil(segment_start_sec * fps))

        # If we don't have the information to compute a safe offset, fall back to
        # the segment start.
        if num_frames is None:
            return segment_start_frame
        if self.target_fps is None or self.target_fps <= 0:
            return segment_start_frame
        if num_frames <= 1:
            return segment_start_frame

        # Maximum allowed begin time so that the *last* sampled timestamp is still
        # within the segment.
        required_duration_sec = (num_frames - 1) / float(self.target_fps)
        max_begin_time_sec = segment_end_sec - required_duration_sec

        # Segment too short: clamp to segment start and rely on padding (repeat last
        # decoded frame) rather than sampling outside the segment.
        if max_begin_time_sec <= segment_start_sec:
            return segment_start_frame

        max_begin_frame = int(math.floor(max_begin_time_sec * fps))

        # Constrain by actual video length to handle annotation inaccuracies
        # With spacing fps/target_fps, last frame index = begin_frame + (num_frames-1) * spacing
        # So max safe begin_frame where last index < length
        if length is not None and length > 0:
            spacing = fps / self.target_fps
            required_span = (num_frames - 1) * spacing
            max_safe_begin_frame = int(length - 1 - required_span)
            # Only apply constraint if it doesn't conflict with segment boundaries
            if max_safe_begin_frame >= segment_start_frame:
                max_begin_frame = min(max_begin_frame, max_safe_begin_frame)

        if max_begin_frame < segment_start_frame:
            return segment_start_frame

        max_offset = int(max_begin_frame - segment_start_frame)

        if self.seed is not None:
            # Use index-based seeding: same idx always produces same offset
            idx_rng = np.random.default_rng(self.seed + idx)
            random_offset = int(idx_rng.integers(0, max_offset + 1, dtype=int))
        else:
            # No seed: truly random offset each time (for training augmentation)
            random_offset = int(np.random.randint(0, max_offset + 1))

        return segment_start_frame + random_offset

    def load_item(self, idx):
        """Load video segment with temporal boundaries."""
        segment, label = self._id2label(idx)

        video_path = self.format_path(segment["video_path"])

        # Compute actual frame count based on segment duration
        # For short segments, this returns fewer frames to avoid repetition
        actual_frame_count = self.compute_actual_frame_count(segment["duration"])

        # Load frames from the video with computed frame count
        frames = self.load_video(video_path, idx, frame_count=actual_frame_count)

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
                "num_frames": len(frames),  # Actual number of frames extracted
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
