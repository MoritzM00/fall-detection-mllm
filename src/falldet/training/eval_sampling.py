"""Stratified-random sampling of evaluation subsets.

Provides ``stratified_sample_indices`` which selects a reproducible,
class-balanced subset of indices from a video dataset, used to cap the
validation set during training.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sized

import numpy as np


def stratified_sample_indices(ds: Sized, n: int, seed: int = 0) -> list[int]:
    """Return *n* reproducible, class-stratified indices from *ds*.

    Labels are read from ``ds.video_segments[i]["label"]`` (available on
    ``OmnifallVideoDataset`` and ``WanfallVideoDataset``).  Falls back to
    plain random sampling when ``video_segments`` is unavailable.

    Each class present in the dataset is guaranteed at least one slot so that
    rare classes (e.g. ``fall``, ``lie_down``) are never dropped from the
    capped eval subset.  Allocation across classes is proportional to class
    size; any over- or under-allocation from the ``max(1, floor)`` rounding is
    reconciled from a shared leftover pool before returning exactly *n* indices.
    """
    n = min(n, len(ds))
    rng = np.random.default_rng(seed)

    segments = getattr(ds, "video_segments", None)
    if segments is None or len(segments) != len(ds):
        chosen = rng.choice(len(ds), n, replace=False)
        return sorted(int(i) for i in chosen)

    # Group indices by label, shuffled for within-class randomness
    groups: dict[object, list[int]] = defaultdict(list)
    for i, seg in enumerate(segments):
        groups[seg["label"]].append(i)
    for idxs in groups.values():
        rng.shuffle(idxs)

    total = len(ds)
    sampled: list[int] = []
    leftover: list[int] = []

    for idxs in groups.values():
        k = max(1, int(n * len(idxs) / total))
        k = min(k, len(idxs))
        sampled.extend(idxs[:k])
        leftover.extend(idxs[k:])

    if len(sampled) > n:
        # Over-allocated (can happen when many tiny classes each get floor=1)
        arr = np.array(sampled)
        rng.shuffle(arr)
        sampled = arr[:n].tolist()
    elif len(sampled) < n:
        # Under-allocated — top up from leftover pool
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])

    return sorted(int(i) for i in sampled)
