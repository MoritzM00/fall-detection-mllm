"""Random-tensor dataset matching the GenericVideoDataset return shape.

For wiring/smoke tests only — no real video I/O.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyVideoDataset(Dataset):
    def __init__(
        self,
        n: int = 8,
        T: int = 16,
        H: int = 224,
        W: int = 224,
        labels: tuple[str, ...] = ("fall", "walk", "sitting", "lying"),
        seed: int = 0,
    ):
        self.n = n
        self.T = T
        self.H = H
        self.W = W
        self.labels = list(labels)
        self._rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        video = torch.randint(
            0, 256, (self.T, 3, self.H, self.W), dtype=torch.uint8, generator=self._rng
        )
        label_idx = idx % len(self.labels)
        return {
            "video": video,
            "label": label_idx,
            "label_str": self.labels[label_idx],
            "video_path": f"dummy/{idx}.mp4",
            "start_time": 0.0,
            "end_time": 1.0,
            "segment_duration": 1.0,
            "dataset": "dummy",
        }
