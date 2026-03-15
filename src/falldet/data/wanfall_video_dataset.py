import csv
import logging
from collections import OrderedDict

import pandas as pd

from .hf_utils import resolve_annotations_file, resolve_split_file
from .video_dataset import OmnifallVideoDataset, idx2label

logger = logging.getLogger(__name__)


class WanfallVideoDataset(OmnifallVideoDataset):
    """
    Video dataset for WanFall that extends OmnifallVideoDataset.

    Differences from OmnifallVideoDataset:
    - No cross-subject/cross-view split concept (split parameter is ignored)
    - Split files have a CSV header row that must be skipped
    - Annotations CSV uses integer label indices instead of string labels
    - Annotations include demographic metadata (age_group, gender, ethnicity, bmi_band)
    - load_item returns demographic metadata alongside segment info
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
        data_fps=16.0,
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
        Initialize WanFall video dataset with temporal segmentation support.

        Args:
            video_root: Root directory for video files
            annotations_file: CSV file with temporal labels (path,label,start,end,subject,cam,dataset,...)
            target_fps: Target FPS for frame sampling
            vid_frame_count: Number of frames to extract per segment
            split_root: Root directory for split files (contains train.csv, val.csv, test.csv)
            dataset_name: Name of the dataset
            mode: Dataset mode ("train", "val", "test", or "all")
            data_fps: Original FPS of the videos (default 16.0 for WanFall)
            path_format: Format string for video paths
            max_retries: Maximum retries for loading a video
            fast: Whether to use fast video loading
            ext: Video file extension
            size: Optional tuple specifying the (height, width) to resize frames to
            max_size: Optional maximum size for resizing
            seed: Random seed for reproducibility
        """
        # Skip OmnifallVideoDataset.__init__ — we call GenericVideoDataset directly
        # and handle split loading ourselves because WanFall uses a different split
        # structure (no cross-subject/cross-view, CSV header in split files).
        # We then reuse the rest of Omnifall's machinery (load_item, get_random_offset, etc.)
        super(OmnifallVideoDataset, self).__init__(
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
            seed=seed,
            disk_cache=kwargs.get("disk_cache"),
            cache_in_memory=kwargs.get("cache_in_memory", False),
        )
        self.dataset_name = dataset_name
        self.split = None  # WanFall doesn't use cross-subject/cross-view splits
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

        # Load temporal segmentation labels
        self._load_temporal_labels(annotations_file)

        # Set paths to segment indices
        self.paths = list(range(len(self.video_segments)))

    def _load_temporal_labels(self, annotations_file):
        """Load temporal segmentation labels from CSV and create segment index.

        WanFall annotations differ from OmniFall:
        - Label column contains integer indices (not string labels)
        - Extra demographic metadata columns (age_group, gender, skin_tone, ethnicity, bmi_band)
        """
        resolved_path = resolve_annotations_file(annotations_file)
        df = pd.read_csv(resolved_path)

        for _, row in df.iterrows():
            path = row.iloc[0]  # Video path
            label = row.iloc[1]  # Label integer index
            start = float(row.iloc[2])  # Start time in seconds
            end = float(row.iloc[3])  # End time in seconds
            subject = row.iloc[4]  # Subject ID
            cam = row.iloc[5]  # Camera ID

            # Optional demographic metadata (WanFall-specific)
            age_group = row.iloc[7] if len(row) > 7 else None
            gender = row.iloc[8] if len(row) > 8 else None
            ethnicity = row.iloc[10] if len(row) > 10 else None
            bmi_band = row.iloc[11] if len(row) > 11 else None

            # WanFall uses 16 classes (0-15)
            # Classes 0-9 match Omnifall, classes 10-15 are WanFall-specific
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

    def load_item(self, idx):
        """Load video segment with temporal boundaries and demographic metadata."""
        # Reuse parent's load_item for video loading and base segment info
        inputs = super().load_item(idx)

        # Add WanFall-specific demographic metadata
        segment = self.video_segments[idx]
        inputs.update(
            {
                "age_group": segment.get("age_group"),
                "gender": segment.get("gender"),
                "ethnicity": segment.get("ethnicity"),
                "bmi_band": segment.get("bmi_band"),
            }
        )

        return inputs

    def __repr__(self):
        return (
            f"WanfallVideoDataset(name='{self.dataset_name}', "
            f"mode='{self.mode}', "
            f"videos={len(self.samples)}, segments={len(self.video_segments)})"
        )
