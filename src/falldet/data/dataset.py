import csv
import logging
import random
import time
from typing import TYPE_CHECKING

import av
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchvision import tv_tensors

if TYPE_CHECKING:
    from falldet.data.cache import TensorDiskCache

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
        seed=0,
        disk_cache: "TensorDiskCache | None" = None,
        cache_in_memory: bool = False,
    ):
        self.video_root = video_root
        self.slow_video_file = annotations_file.replace(".csv", "_slow.csv")
        self.target_fps = target_fps
        self.load_video = (
            self.load_video_fast if fast and vid_frame_count is not None else self.load_video_slow
        )
        self.vid_frame_count = vid_frame_count
        self.data_fps = data_fps
        self.path_format = path_format
        self.max_retries = max_retries
        self.mode = mode
        self.annotations = {}
        self.size = size
        self.max_size = max_size
        self.seed = seed
        self._disk_cache = disk_cache
        self._cache_in_memory = cache_in_memory
        self._memory_cache: dict[int, dict] = {}
        self._stat_disk_hits = 0
        self._stat_disk_misses = 0
        self._stat_mem_hits = 0
        self._stat_log_interval = 500  # log every N cache-layer accesses

    def enable_memory_cache(self) -> None:
        """Enable the lazy in-memory cache. Safe to call after construction."""
        self._cache_in_memory = True

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
        frames = self.load_video(video_path, idx)
        video_io_end = time.time()

        # to numpy
        frames = np.array(frames)

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

    def _compute_cache_key(self, idx: int) -> str:
        """Compute a cache key for the item at idx. Override in subclasses for segment-aware keys."""
        from falldet.data.cache import compute_cache_key

        path, _ = self._id2label(idx)
        return compute_cache_key(video_path=str(path), idx=idx)

    def _load_item_cached(self, idx: int) -> dict:
        """Load item with optional caching: memory → disk → decode."""
        has_disk = self._disk_cache is not None
        has_mem = self._cache_in_memory

        if not has_disk and not has_mem:
            return self.load_item(idx)

        # Layer 1: in-memory uses idx directly — no hash computation on hit.
        if has_mem and idx in self._memory_cache:
            self._stat_mem_hits += 1
            self._maybe_log_cache_stats()
            return self._memory_cache[idx]

        # Layer 2: disk (hash computed only when needed).
        if has_disk:
            disk_key = self._compute_cache_key(idx)
            cached = self._disk_cache.get(disk_key)
            if cached is not None:
                self._stat_disk_hits += 1
                self._maybe_log_cache_stats()
                if has_mem:
                    self._memory_cache[idx] = cached
                return cached
            self._stat_disk_misses += 1
            self._maybe_log_cache_stats()
        else:
            disk_key = None

        # Cache miss: decode from video.
        result = self.load_item(idx)
        if has_disk:
            self._disk_cache.put(disk_key, result)  # type: ignore[arg-type]
        if has_mem:
            self._memory_cache[idx] = result
        return result

    def _maybe_log_cache_stats(self) -> None:
        total = self._stat_disk_hits + self._stat_disk_misses + self._stat_mem_hits
        if total % self._stat_log_interval == 0:
            self.log_cache_stats()

    def log_cache_stats(self) -> None:
        """Log a one-line cache hit-rate summary."""
        disk_total = self._stat_disk_hits + self._stat_disk_misses
        disk_rate = self._stat_disk_hits / disk_total if disk_total else 0.0
        parts = []
        if self._cache_in_memory:
            parts.append(f"mem={self._stat_mem_hits}")
        if self._disk_cache is not None:
            parts.append(f"disk={self._stat_disk_hits}/{disk_total} ({disk_rate:.0%})")
        if parts:
            label = getattr(self, "dataset_name", type(self).__name__)
            logger.info(f"Cache stats [{label}]: {', '.join(parts)}")

    def __getitem__(self, idx):
        original_idx = idx
        retries = 0
        while retries < self.max_retries:
            try:
                return self._load_item_cached(idx)

            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    logging.error(
                        f"Error loading video at original index {original_idx} "
                        f"(last tried {idx}): {str(e)}"
                    )
                else:
                    idx = random.randint(0, len(self) - 1)
                    logger.warning(
                        f"Failed to load video at index {idx}, "
                        f"retrying with a different video (attempt {retries}/{self.max_retries})"
                    )

        raise RuntimeError(
            f"Failed to load video at original index {original_idx} "
            f"after {self.max_retries} attempts"
        )

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
                ]
            )
            frames = transform(frames)

        return {"video": frames}

    def get_random_offset(self, length, target_interval, idx, fps, start=0):
        if self.vid_frame_count is None or length < self.vid_frame_count * target_interval:
            return 0
        else:
            return random.randint(0, length - self.vid_frame_count * target_interval)

    def load_video_fast(self, path, idx):
        try:
            with av.open(path) as container:
                vs = next(s for s in container.streams if s.type == "video")

                # robust fps
                rate = vs.average_rate or vs.base_rate
                if not rate or rate.denominator == 0:
                    raise ValueError("Cannot determine FPS")
                fps = float(rate)

                # stream time‑base
                tb = vs.time_base
                if not tb:
                    raise ValueError("Missing time_base")
                tb = float(tb)

                frame_cnt = None if vs.frames in (0, None) else int(vs.frames)

                # random start (in frames) for augmentation
                begin_frame = self.get_random_offset(frame_cnt, 1, idx, fps) if frame_cnt else 0

                # Calculate desired timestamps
                desired_timestamps = [
                    (begin_frame / fps) + n / self.target_fps for n in range(self.vid_frame_count)
                ]

                desired_pts = [int(ts / tb) for ts in desired_timestamps]

                # Try seeking to just before first desired frame
                if desired_pts:
                    try:
                        container.seek(desired_pts[0], any_frame=False, backward=True, stream=vs)
                    except av.error.FFmpegError:
                        # Fallback if no keyframe found
                        container.seek(0, stream=vs)

                frames, want_idx, prev = [], 0, None
                for f in container.decode(vs):
                    if f.pts is None:
                        continue

                    # Collect frames matching desired PTS values
                    while want_idx < len(desired_pts) and f.pts >= desired_pts[want_idx]:
                        if prev and abs(prev.pts - desired_pts[want_idx]) < abs(
                            f.pts - desired_pts[want_idx]
                        ):
                            frames.append(prev.to_ndarray(format="rgb24"))
                        else:
                            frames.append(f.to_ndarray(format="rgb24"))
                        want_idx += 1

                    if want_idx == len(desired_pts):
                        break
                    prev = f

                if not frames:
                    logging.warning(f"{idx}: fallback to slow loader")
                    return self.load_video_slow(path, idx)

                # Better frame padding strategy
                if len(frames) < self.vid_frame_count:
                    if len(frames) > 0:
                        # Repeat last frame instead of cycling
                        last_frame = frames[-1]
                        while len(frames) < self.vid_frame_count:
                            frames.append(last_frame)
                    else:
                        raise ValueError("No frames decoded")

                return frames

        except Exception as e:
            logging.error(f"Error reading video {path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process video {path}") from e

    def load_video_slow(self, video_path, idx):
        try:
            with av.open(video_path) as container:
                video_stream = next(s for s in container.streams if s.type == "video")

                frame_rate = video_stream.average_rate  # Detect actual frame rate

                if frame_rate:
                    fps = float(frame_rate.numerator / frame_rate.denominator)
                elif self.data_fps is not None:
                    fps = self.data_fps
                else:
                    raise ValueError(
                        "Cannot determine fps: video has no average_rate and data_fps is None"
                    )

                target_interval = max(
                    round(fps / self.target_fps), 1
                )  # Calculate downsampling interval

                frames = []
                for i, frame in enumerate(container.decode(video_stream)):
                    if i % target_interval == 0:  # Keep only frames at the target interval
                        img = frame.to_ndarray(format="rgb24")
                        frames.append(img)

        except Exception as e:
            logging.error(f"Error reading video {video_path}: {e}")
            raise RuntimeError("Failed to process video")

        if not frames:
            raise RuntimeError(f"No frames decoded from video {video_path}")

        if self.vid_frame_count is None:
            # Load full video, no cycling required
            return frames

        if len(frames) < self.vid_frame_count:
            # Handle short videos by cycling frames
            logging.debug(
                f"Video {video_path} is too short. "
                + f"Got {len(frames)} sampled at {self.target_fps} instead of {self.vid_frame_count}. "
                + f"Cycling frames to match {self.vid_frame_count} frames."
            )
            frames = (frames * ((self.vid_frame_count // len(frames)) + 1))[: self.vid_frame_count]
        else:
            # Select a random consecutive sequence of frames
            start_index = self.get_random_offset(len(frames), 1, idx, fps)
            frames = frames[start_index : start_index + self.vid_frame_count]

        return frames
