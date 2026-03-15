"""Tests for TensorDiskCache and dataset-level caching.

No real video files needed — load_item is patched to return synthetic tensors.
"""

import csv
from pathlib import Path
from unittest.mock import patch

import torch
from torchvision import tv_tensors

from falldet.data.cache import TensorDiskCache, compute_cache_key
from falldet.data.dataset import GenericVideoDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARAMS = {"target_fps": 8.0, "vid_frame_count": 16, "size": 224, "seed": 0}


def _make_item(label: int = 0) -> dict:
    """Synthetic item matching what GenericVideoDataset.load_item returns."""
    video = tv_tensors.Video(torch.randint(0, 255, (16, 3, 224, 224), dtype=torch.uint8))
    return {"video": video, "label": label}


def _make_dataset(
    tmp_path: Path,
    cache_dir: Path | None = None,
    cache_in_memory: bool = False,
    read_only: bool = False,
):
    """GenericVideoDataset with a 3-item CSV annotation file and optional cache."""
    ann_file = tmp_path / "annotations.csv"
    video_root = tmp_path / "videos"
    video_root.mkdir(exist_ok=True)
    with open(ann_file, "w", newline="") as f:
        csv.writer(f).writerows([["vid_a", 0], ["vid_b", 1], ["vid_c", 2]])

    disk_cache = (
        TensorDiskCache(cache_dir, PARAMS, read_only=read_only) if cache_dir is not None else None
    )
    ds = GenericVideoDataset(
        video_root=str(video_root),
        annotations_file=str(ann_file),
        target_fps=PARAMS["target_fps"],
        vid_frame_count=PARAMS["vid_frame_count"],
        size=PARAMS["size"],
        seed=PARAMS["seed"],
        disk_cache=disk_cache,
        cache_in_memory=cache_in_memory,
    )
    ds.load_annotations(str(ann_file))
    return ds


# ---------------------------------------------------------------------------
# TensorDiskCache unit tests
# ---------------------------------------------------------------------------


def _writable(tmp_path, params=None) -> TensorDiskCache:
    return TensorDiskCache(tmp_path, params or PARAMS, read_only=False)


def test_get_returns_none_on_miss(tmp_path):
    assert _writable(tmp_path).get("nonexistent_key") is None


def test_put_then_get_roundtrip(tmp_path):
    cache = _writable(tmp_path)
    item = _make_item(label=1)
    key = compute_cache_key("vid_a.mp4", 0)

    cache.put(key, item)
    loaded = cache.get(key)

    assert loaded is not None
    assert loaded["label"] == 1
    assert torch.equal(loaded["video"], item["video"])


def test_put_preserves_uint8_dtype(tmp_path):
    cache = _writable(tmp_path)
    key = compute_cache_key("vid_a.mp4", 0)
    cache.put(key, _make_item())
    assert cache.get(key)["video"].dtype == torch.uint8


def test_put_strips_tv_tensors_subclass(tmp_path):
    """Loaded tensor must be a plain torch.Tensor, not tv_tensors.Video."""
    cache = _writable(tmp_path)
    item = _make_item()
    assert isinstance(item["video"], tv_tensors.Video)
    key = compute_cache_key("vid_a.mp4", 0)
    cache.put(key, item)
    assert type(cache.get(key)["video"]) is torch.Tensor


def test_read_only_put_is_noop(tmp_path):
    """put() on a read-only cache must not write anything."""
    cache = TensorDiskCache(tmp_path, PARAMS, read_only=True)
    cache.put(compute_cache_key("vid_a.mp4", 0), _make_item())
    assert list(tmp_path.rglob("*.pt")) == []


def test_namespace_differs_for_different_params(tmp_path):
    assert (
        _writable(tmp_path, {**PARAMS, "size": 224})._root
        != _writable(tmp_path, {**PARAMS, "size": 448})._root
    )


def test_namespace_stable_for_same_params(tmp_path):
    assert _writable(tmp_path)._root == _writable(tmp_path)._root


# ---------------------------------------------------------------------------
# Dataset-level disk cache tests
# ---------------------------------------------------------------------------


def test_disk_cache_equivalence(tmp_path):
    """Item loaded via disk cache must be tensor-equal to the original."""
    cache_dir = tmp_path / "cache"

    # First pass: cold cache (read_only=False) — write synthetic items to disk.
    ds_cold = _make_dataset(tmp_path, cache_dir=cache_dir, read_only=False)
    items_cold = {}
    with patch.object(ds_cold, "load_item", side_effect=lambda idx: _make_item(idx)):
        for i in range(len(ds_cold)):
            items_cold[i] = ds_cold[i]

    # Second pass: warm cache (read_only=True) — load_item must not be called.
    ds_warm = _make_dataset(tmp_path, cache_dir=cache_dir, read_only=True)
    load_called: list[int] = []
    with patch.object(
        ds_warm, "load_item", side_effect=lambda idx: load_called.append(idx) or _make_item(idx)
    ):
        for i in range(len(ds_warm)):
            item = ds_warm[i]
            assert item["label"] == items_cold[i]["label"]
            assert torch.equal(item["video"], items_cold[i]["video"])

    assert load_called == [], f"load_item called on warm cache for indices: {load_called}"


def test_disk_cache_load_item_called_once_per_idx(tmp_path):
    """With disk cache (write mode), load_item fires exactly once per idx across multiple passes."""
    cache_dir = tmp_path / "cache"
    ds = _make_dataset(tmp_path, cache_dir=cache_dir, read_only=False)
    call_counts: dict[int, int] = {}

    with patch.object(
        ds,
        "load_item",
        side_effect=lambda idx: [
            call_counts.__setitem__(idx, call_counts.get(idx, 0) + 1),
            _make_item(idx),
        ][1],
    ):
        for _ in range(3):
            for i in range(len(ds)):
                ds[i]

    assert all(v == 1 for v in call_counts.values()), f"Expected 1 call per idx, got: {call_counts}"


# ---------------------------------------------------------------------------
# Dataset-level memory cache tests
# ---------------------------------------------------------------------------


def test_memory_cache_equivalence(tmp_path):
    """Item loaded via memory cache must be identical object (same reference)."""
    ds = _make_dataset(tmp_path, cache_in_memory=True)

    with patch.object(ds, "load_item", side_effect=lambda idx: _make_item(idx)):
        first = [ds[i] for i in range(len(ds))]

    # Second pass hits memory — no patching needed, same ds instance.
    second = [ds[i] for i in range(len(ds))]

    for a, b in zip(first, second):
        assert a is b, "Memory cache should return the exact same dict object"


def test_memory_cache_load_item_called_once_per_idx(tmp_path):
    """With memory cache, load_item fires exactly once per idx across multiple passes."""
    ds = _make_dataset(tmp_path, cache_in_memory=True)
    call_counts: dict[int, int] = {}

    with patch.object(
        ds,
        "load_item",
        side_effect=lambda idx: [
            call_counts.__setitem__(idx, call_counts.get(idx, 0) + 1),
            _make_item(idx),
        ][1],
    ):
        for _ in range(3):
            for i in range(len(ds)):
                ds[i]

    assert all(v == 1 for v in call_counts.values()), f"Expected 1 call per idx, got: {call_counts}"


def test_enable_memory_cache(tmp_path):
    """enable_memory_cache() activates caching after construction."""
    ds = _make_dataset(tmp_path)
    assert not ds._cache_in_memory
    ds.enable_memory_cache()
    assert ds._cache_in_memory

    call_counts: dict[int, int] = {}
    with patch.object(
        ds,
        "load_item",
        side_effect=lambda idx: [
            call_counts.__setitem__(idx, call_counts.get(idx, 0) + 1),
            _make_item(idx),
        ][1],
    ):
        for _ in range(3):
            for i in range(len(ds)):
                ds[i]

    assert all(v == 1 for v in call_counts.values())


# ---------------------------------------------------------------------------
# No-cache baseline
# ---------------------------------------------------------------------------


def test_no_cache_calls_load_item_every_access(tmp_path):
    """Without any cache, load_item is called on every __getitem__."""
    ds = _make_dataset(tmp_path)
    call_counts: dict[int, int] = {}

    n_passes = 3
    with patch.object(
        ds,
        "load_item",
        side_effect=lambda idx: [
            call_counts.__setitem__(idx, call_counts.get(idx, 0) + 1),
            _make_item(idx),
        ][1],
    ):
        for _ in range(n_passes):
            for i in range(len(ds)):
                ds[i]

    assert all(v == n_passes for v in call_counts.values()), (
        f"Expected {n_passes} calls per idx, got: {call_counts}"
    )
