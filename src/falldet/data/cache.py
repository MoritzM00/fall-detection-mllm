"""Persistent disk cache and in-memory lazy cache for preprocessed video tensors.

Two independent cache layers (can be used separately or together):

- TensorDiskCache: saves .pt files under a namespace keyed by dataset params.
  Changing any param (fps, size, seed) shifts the namespace, invalidating stale
  entries automatically.

- In-memory cache: a plain dict[str, dict] managed by the dataset via the
  cache_in_memory flag.  Populated lazily on first access.
"""

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def compute_cache_key(
    video_path: str,
    idx: int,
    segment_start: float | None = None,
    segment_end: float | None = None,
) -> str:
    """Compute a deterministic SHA-256 cache key from video identity parameters."""
    key_data = json.dumps(
        {
            "video_path": video_path,
            "idx": idx,
            "start": segment_start,
            "end": segment_end,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(key_data).hexdigest()


class TensorDiskCache:
    """Persistent disk cache for preprocessed video tensors.

    Layout: ``{cache_dir}/{namespace}/{key[:2]}/{key}.pt``

    The namespace is derived from dataset_params so that changing target_fps,
    vid_frame_count, size, or seed automatically invalidates old entries.
    Tensors are stored in their original dtype (uint8 for video frames).
    Writes are atomic (temp-file + os.rename) and thread-safe.
    """

    def __init__(self, cache_dir: Path, dataset_params: dict) -> None:
        self._base = Path(cache_dir)
        namespace = hashlib.sha256(json.dumps(dataset_params, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        self._root = self._base / namespace
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorDiskCache: root={self._root} params={dataset_params}")

    def _key_path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.pt"

    def get(self, key: str) -> dict | None:
        path = self._key_path(key)
        try:
            return torch.load(path, map_location="cpu", weights_only=False)  # noqa: S301
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"TensorDiskCache: corrupt entry {path}, ignoring: {e}")
            return None

    def put(self, key: str, item: dict) -> None:
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Strip tv_tensors subclass (e.g. tv_tensors.Video) before pickling.
        save_item = {
            k: v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v
            for k, v in item.items()
        }

        # Atomic write: save to a temp file then rename into place.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.close(tmp_fd)
            torch.save(save_item, tmp_path)
            os.rename(tmp_path, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def log_estimated_disk_usage(self, num_items: int) -> None:
        """Log disk usage: actual current size + projected total for num_items."""
        existing = list(self._root.rglob("*.pt"))
        if existing:
            actual_bytes = sum(p.stat().st_size for p in existing)
            avg_bytes = actual_bytes / len(existing)
            projected_gb = (avg_bytes * num_items) / (1024**3)
            actual_gb = actual_bytes / (1024**3)
            logger.info(
                f"TensorDiskCache: {len(existing)}/{num_items} entries cached "
                f"({actual_gb:.1f} GB on disk), "
                f"projected total: {projected_gb:.1f} GB"
            )
        else:
            # Rough estimate: 16 frames × size × size × 3 channels × 1 byte (uint8)
            size = 448  # default; actual may differ
            rough_bytes = 16 * size * size * 3
            estimated_gb = (rough_bytes * num_items) / (1024**3)
            logger.info(
                f"TensorDiskCache: no cached entries yet. "
                f"Estimated disk usage for {num_items} items: ~{estimated_gb:.1f} GB"
            )
