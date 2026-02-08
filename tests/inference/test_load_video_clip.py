"""Tests for load_video_clip in infreqact.inference.base."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from infreqact.inference.base import load_video_clip

# ---------------------------------------------------------------------------
# Synthetic video fixture
# ---------------------------------------------------------------------------
DURATION_SEC = 3.0
NATIVE_FPS = 30
WIDTH, HEIGHT = 64, 48
TOTAL_FRAMES = int(DURATION_SEC * NATIVE_FPS)  # 90


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a 3-second, 30 fps, 64x48 MP4 with unique per-frame colors."""
    video_dir = tmp_path_factory.mktemp("video")
    video_path = video_dir / "test.mp4"

    rng = np.random.RandomState(42)
    frames = []
    for _ in range(TOTAL_FRAMES):
        # unique solid color per frame so we can verify frame identity
        color = rng.randint(0, 256, size=3, dtype=np.uint8)
        frame = np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)
        frames.append(frame)

    iio.imwrite(
        str(video_path),
        np.stack(frames),
        fps=NATIVE_FPS,
        codec="libx264",
        plugin="pyav",
    )
    return video_path


# ===================================================================
# Basic contract
# ===================================================================
class TestBasicContract:
    def test_returns_tuple_of_ndarray_and_dict(self, synthetic_video):
        result = load_video_clip(str(synthetic_video))
        assert isinstance(result, tuple)
        assert len(result) == 2
        frames, meta = result
        assert isinstance(frames, np.ndarray)
        assert isinstance(meta, dict)

    def test_frame_shape(self, synthetic_video):
        frames, _ = load_video_clip(str(synthetic_video))
        assert frames.ndim == 4
        n, h, w, c = frames.shape
        assert c == 3
        assert h == HEIGHT
        assert w == WIDTH
        assert n > 0

    def test_metadata_keys(self, synthetic_video):
        _, meta = load_video_clip(str(synthetic_video))
        expected_keys = {
            "requested_frames",
            "actual_frames",
            "frame_indices",
            "timestamps",
            "available_sec",
            "clip_start_sec",
            "clip_end_sec",
            "shift_sec",
        }
        assert expected_keys == set(meta.keys())


# ===================================================================
# Frame extraction
# ===================================================================
class TestFrameExtraction:
    def test_full_video_extraction(self, synthetic_video):
        """3s video at target_fps=8, max_frames=16 → should yield 16 frames."""
        frames, meta = load_video_clip(str(synthetic_video), target_fps=8, max_frames=16)
        assert frames.shape[0] == 16
        assert meta["requested_frames"] == 16

    def test_temporal_segment(self, synthetic_video):
        """1-second segment at 8 fps → at most 9 frames (fence-post)."""
        frames, meta = load_video_clip(
            str(synthetic_video), start_sec=1.0, end_sec=2.0, target_fps=8, max_frames=16
        )
        assert frames.shape[0] <= 9

    def test_short_segment_returns_fewer_frames(self, synthetic_video):
        """Very short segment returns fewer than max_frames."""
        frames, meta = load_video_clip(
            str(synthetic_video), end_sec=0.3, target_fps=8, max_frames=16
        )
        assert frames.shape[0] < 16

    def test_max_frames_limits_output(self, synthetic_video):
        """max_frames=4 limits the output regardless of segment length."""
        frames, meta = load_video_clip(str(synthetic_video), target_fps=8, max_frames=4)
        assert frames.shape[0] <= 4


# ===================================================================
# Offset behavior
# ===================================================================
class TestOffsetBehavior:
    def test_offset_start(self, synthetic_video):
        """offset=0.0 → clip starts at the beginning of the segment."""
        _, meta = load_video_clip(str(synthetic_video), target_fps=8, max_frames=16, offset=0.0)
        assert meta["clip_start_sec"] == pytest.approx(0.0, abs=0.05)

    def test_offset_end(self, synthetic_video):
        """offset=1.0 → clip anchored at end of segment, clip_start_sec > 0."""
        _, meta = load_video_clip(str(synthetic_video), target_fps=8, max_frames=16, offset=1.0)
        assert meta["clip_start_sec"] > 0.0

    def test_offset_middle(self, synthetic_video):
        """offset=0.5 → clip_start_sec between the start-anchored and end-anchored positions."""
        _, meta_start = load_video_clip(
            str(synthetic_video), target_fps=8, max_frames=16, offset=0.0
        )
        _, meta_end = load_video_clip(str(synthetic_video), target_fps=8, max_frames=16, offset=1.0)
        _, meta_mid = load_video_clip(str(synthetic_video), target_fps=8, max_frames=16, offset=0.5)
        assert (
            meta_start["clip_start_sec"] <= meta_mid["clip_start_sec"] <= meta_end["clip_start_sec"]
        )

    def test_offset_random_reproducible(self, synthetic_video):
        """Same seed → identical frames."""
        frames1, meta1 = load_video_clip(
            str(synthetic_video), target_fps=8, max_frames=16, offset="random", seed=123
        )
        frames2, meta2 = load_video_clip(
            str(synthetic_video), target_fps=8, max_frames=16, offset="random", seed=123
        )
        np.testing.assert_array_equal(frames1, frames2)
        assert meta1["frame_indices"] == meta2["frame_indices"]

    def test_offset_random_different_seeds(self, synthetic_video):
        """Different seeds → (very likely) different clip positions."""
        _, meta1 = load_video_clip(
            str(synthetic_video), target_fps=8, max_frames=16, offset="random", seed=1
        )
        _, meta2 = load_video_clip(
            str(synthetic_video), target_fps=8, max_frames=16, offset="random", seed=999
        )
        # With a 3s video and 16/8=1.875s clip, there's ~1.125s of slack,
        # so two different random seeds should almost certainly differ.
        assert meta1["clip_start_sec"] != meta2["clip_start_sec"]


# ===================================================================
# Error handling
# ===================================================================
class TestErrorHandling:
    def test_invalid_segment_raises_valueerror(self, synthetic_video):
        """start > end → ValueError."""
        with pytest.raises(ValueError, match="Invalid segment"):
            load_video_clip(str(synthetic_video), start_sec=2.0, end_sec=1.0)

    def test_segment_clamped_to_duration(self, synthetic_video):
        """end=100 is silently clamped to video duration (~3s)."""
        frames, meta = load_video_clip(
            str(synthetic_video), end_sec=100.0, target_fps=8, max_frames=16
        )
        assert meta["available_sec"] == pytest.approx(DURATION_SEC, abs=0.15)
        assert frames.shape[0] > 0


# ===================================================================
# Bug regressions
# ===================================================================
class TestBugRegressions:
    def test_shift_sec_in_metadata_short_segment(self, synthetic_video):
        """Bug 2: shift_sec must be present (and 0.0) for short segments."""
        _, meta = load_video_clip(str(synthetic_video), end_sec=0.3, target_fps=8, max_frames=16)
        assert "shift_sec" in meta
        assert meta["shift_sec"] == 0.0

    def test_shift_sec_in_metadata_full_segment(self, synthetic_video):
        """Bug 2: shift_sec must be present for full-length segments too."""
        _, meta = load_video_clip(str(synthetic_video), target_fps=8, max_frames=16, offset=0.5)
        assert "shift_sec" in meta
        assert isinstance(meta["shift_sec"], float)

    def test_dedup_metadata_actual_frames(self, synthetic_video):
        """Bug 3: actual_frames must match the real number of returned frames."""
        frames, meta = load_video_clip(str(synthetic_video), target_fps=8, max_frames=16)
        assert meta["actual_frames"] == frames.shape[0]

    def test_dedup_metadata_actual_frames_short(self, synthetic_video):
        """Bug 3: actual_frames matches when dedup might reduce the count."""
        # Use a very high target_fps relative to native_fps to trigger dedup
        frames, meta = load_video_clip(
            str(synthetic_video), end_sec=0.5, target_fps=60, max_frames=64
        )
        assert meta["actual_frames"] == frames.shape[0]
        assert meta["actual_frames"] == len(meta["frame_indices"])
