import csv
import logging
import random
from collections import OrderedDict

import pandas as pd
import torch

from .dataset import GenericVideoDataset, idx2label
from .hf_utils import resolve_annotations_file, resolve_split_file


class WanfallVideoDataset(GenericVideoDataset):
    """
    Video dataset for WanFall that handles temporal segmentation annotations.
    Extends GenericVideoDataset to support start/end times and proper segment sampling.

    Unlike OmnifallVideoDataset, WanFall uses a simpler split structure without
    cross-subject/cross-view subdirectories.
    """

    def __init__(
        self,
        video_root,
        annotations_file,
        target_fps,
        vid_frame_count,
        split_root=None,
        dataset_name="WanFall",
        mode="train",
        data_fps=None,
        path_format="{video_root}/{video_path}{ext}",
        max_retries=10,
        image_processor=None,
        normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        fast=True,
        ext=".mp4",
        **kwargs,
    ):
        """
        Initialize WanFall video dataset with temporal segmentation support.

        Args:
            video_root: Root directory for video files
            annotations_file: CSV file with temporal labels (path,label,start,end,subject,cam,dataset)
            target_fps: Target FPS for frame sampling
            vid_frame_count: Number of frames to extract per segment
            split_root: Root directory for split files (contains train.csv, val.csv, test.csv)
            dataset_name: Name of the dataset
            mode: Dataset mode ("train", "val", "test", or "all")
            data_fps: Original FPS of the videos (if known, default 16.0 for WanFall)
            path_format: Format string for video paths
            max_retries: Maximum retries for loading a video
            image_processor: Image processor for frame transformation
            normalize: Normalization parameters
            fast: Whether to use fast video loading
        """
        self.dataset_name = dataset_name
        self.mode = mode
        self.split_root = split_root
        self.ext = ext

        # Extract split type from split_root for logging
        split_type = "unknown"
        if split_root and "config=" in split_root:
            split_type = split_root.split("config=")[-1].split("/")[0]

        logging.info(
            f"Initializing {self.dataset_name} dataset in {self.mode} mode (split_type={split_type})"
        )

        # Video segments with temporal annotations
        self.video_segments = []
        self.samples = OrderedDict()

        # Load split file if provided
        assert mode == "all" or split_root is not None, (
            "Split root must be provided unless mode is 'all'"
        )

        if mode != "all":
            # WanFall has simpler split structure: split_root/{mode}.csv
            # Resolve split file (supports local paths and HF dataset references)
            self.split_file = resolve_split_file(split_root, mode)
            with open(self.split_file) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                paths = sorted([row[0] for row in reader])
                for p in paths:
                    self.samples[p] = {"id": p}

        # Initialize parent class (but we'll override the annotations loading)
        self.video_root = video_root
        self.target_fps = target_fps
        self.vid_frame_count = vid_frame_count
        self.data_fps = data_fps if data_fps is not None else 16.0  # WanFall default
        self.path_format = path_format
        self.max_retries = max_retries
        self.image_processor = image_processor
        self.normalize = normalize
        self.mode = mode
        self.fast = fast

        # Load temporal segmentation labels
        self._load_temporal_labels(annotations_file)

        # Initialize GenericVideoDataset attributes without calling __init__
        # (to avoid loading annotations twice)
        self.load_video = (
            self.load_video_fast if fast and vid_frame_count is not None else self.load_video_slow
        )

        # Import necessary methods from parent class
        from data.dataset import GenericVideoDataset

        self.load_video_fast = GenericVideoDataset.load_video_fast.__get__(self)
        self.load_video_slow = GenericVideoDataset.load_video_slow.__get__(self)
        self.transform_frames = GenericVideoDataset.transform_frames.__get__(self)

        # Initialize image processor if not provided
        if self.image_processor is None:
            from transformers import AutoImageProcessor

            self.image_processor = AutoImageProcessor.from_pretrained(
                "MCG-NJU/videomae-small-finetuned-kinetics"
            )

        self.image_processor.image_mean = (normalize or {}).get("mean", [0.485, 0.456, 0.406])
        self.image_processor.image_std = (normalize or {}).get("std", [0.229, 0.224, 0.225])

        # Initialize transforms (from parent class)
        from data.transforms.fourier import PhaseOnly
        from data.transforms.tube_masking import TubeMasker

        self.fourier_trans = PhaseOnly(in_arrangement="t c h w")
        input_size = (8, 14, 14)  # Patches in (T, H, W)
        mask_ratio = 0.9  # Ratio of patches to mask
        self.tube_masker = TubeMasker(input_size, mask_ratio)

        # Set paths to segment indices
        self.paths = list(range(len(self.video_segments)))
        self.annotations = {}  # Not used in our case

    def _load_temporal_labels(self, annotations_file):
        """Load temporal segmentation labels from CSV and create segment index."""
        # Resolve annotations file (supports local paths and HF dataset references)
        resolved_path = resolve_annotations_file(annotations_file)
        df = pd.read_csv(resolved_path)

        for _, row in df.iterrows():
            path = row.iloc[0]  # Video path
            label = row.iloc[1]  # Label integer
            start = float(row.iloc[2])  # Start time in seconds
            end = float(row.iloc[3])  # End time in seconds
            subject = row.iloc[4]  # Subject ID
            cam = row.iloc[5]  # Camera ID

            # Optional demographic metadata (WanFall-specific, may not exist in all datasets)
            # Using .get() to safely handle missing columns
            dataset_name = row.iloc[6] if len(row) > 6 else None  # noqa: F841
            age_group = row.iloc[7] if len(row) > 7 else None
            gender = row.iloc[8] if len(row) > 8 else None
            skin_tone = row.iloc[9] if len(row) > 9 else None  # noqa: F841
            ethnicity = row.iloc[10] if len(row) > 10 else None
            bmi_band = row.iloc[11] if len(row) > 11 else None

            # WanFall uses 16 classes (0-15)
            # Classes 0-9 match Omnifall, classes 10-15 are WanFall-specific
            label_str = idx2label[label]

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
                    "subject": subject,
                    "cam": cam,
                    "duration": end - start,
                    # Demographic metadata (None for datasets without this info)
                    "age_group": age_group,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "bmi_band": bmi_band,
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
            random_offset = random.randint(0, max_offset)
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
                # Add demographic metadata (will be None for datasets without this info)
                "age_group": segment.get("age_group"),
                "gender": segment.get("gender"),
                "ethnicity": segment.get("ethnicity"),
                "bmi_band": segment.get("bmi_band"),
            }
        )

        return inputs

    @property
    def targets(self):
        """Return all class labels for segments in this dataset."""
        return torch.tensor([segment["label"] for segment in self.video_segments])

    def __repr__(self):
        return (
            f"WanfallVideoDataset(name='{self.dataset_name}', "
            f"mode='{self.mode}', "
            f"videos={len(self.samples)}, segments={len(self.video_segments)})"
        )
