import logging
import random
import warnings
from typing import Literal

import numpy as np
from decord import VideoReader, cpu

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logger = logging.getLogger(__name__)


def load_video_clip(
    path: str,
    start_sec: float = 0.0,
    end_sec: float = float("inf"),
    target_fps: float = 8.0,
    max_frames: int = 16,
    offset: Literal["random"] | float = 0.0,
    seed: int | None = None,
    width: int = -1,
    height: int = -1,
    num_threads: int = 0,
) -> tuple[np.ndarray, dict]:
    """Extract a clip from a video with temporal down-sampling and random access.

    Parameters
    ----------
    path : str
        Path to the video file.
    start_sec, end_sec : float
        Temporal boundaries of the segment (in seconds).
        Default is the entire video (0.0 to inf).
    target_fps : float
        Desired output frame rate (e.g. 8 → 8 frames per second).
    max_frames : int
        Maximum number of frames to return.  If the segment is shorter than
        ``max_frames / target_fps`` seconds the actual (smaller) number of
        frames is returned.
    offset : "random" or float
        Controls *where* inside the segment the sub-clip is anchored when the
        segment is longer than needed.

        * ``0.0``  – start of the segment (default).
        * ``1.0``  – end of the segment.
        * Any float in [0, 1] – linearly interpolated position.
        * ``"random"`` – a random offset is chosen; use *seed* for
        reproducibility.
    seed : int | None
        RNG seed used when ``offset="random"``.
    width, height : int
        Resize dimensions passed to decord. -1 means keep original.
    num_threads : int
        Number of decoding threads (0 = auto).

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple of (frames, metadata) where frames is an array of shape
        ``(N, H, W, 3)`` with ``N <= max_frames``, and metadata is a dict
        containing keys like ``requested_frames``, ``frame_indices``,
        ``timestamps``, ``available_sec``, ``clip_start_sec``,
        ``clip_end_sec``, ``shift_sec``, and ``actual_frames``.
    """
    # ------------------------------------------------------------------
    # 1. Open video & read metadata
    # ------------------------------------------------------------------
    vr = VideoReader(path, ctx=cpu(0), width=width, height=height, num_threads=num_threads)

    native_fps: float = vr.get_avg_fps()
    total_frames: int = len(vr)
    duration: float = total_frames / native_fps

    # clamp boundaries
    start_sec = max(0.0, start_sec)
    end_sec = min(duration, end_sec)
    if end_sec <= start_sec:
        raise ValueError(
            f"Invalid segment: start_sec={start_sec}, end_sec={end_sec}, "
            f"video duration={duration:.3f}s"
        )

    # ------------------------------------------------------------------
    # 2. Determine how many frames the segment can provide at target_fps
    # ------------------------------------------------------------------
    segment_duration = end_sec - start_sec
    # fence-post: a 1s segment at 2 FPS can provide 3 frames at timestamps [0s, 0.5s, 1s]
    available_frames = max(1, 1 + int(segment_duration * target_fps))
    n_frames = min(max_frames, available_frames)

    # ------------------------------------------------------------------
    # 3. Compute the offset (temporal anchor inside the segment)
    # ------------------------------------------------------------------
    # Offset only applies when the segment is long enough to fully provide
    # max_frames.  If the segment is too short (n_frames < max_frames), we
    # use every available frame starting from the segment start — no slack.
    shift_sec = 0.0
    if n_frames < max_frames:
        clip_start_sec = start_sec
    else:
        clip_duration = (n_frames - 1) / target_fps
        slack = segment_duration - clip_duration

        if slack <= 0:
            effective_offset = 0.0
        elif offset == "random":
            rng = random.Random(seed)
            effective_offset = rng.random()
        else:
            effective_offset = float(max(0.0, min(1.0, offset)))

        shift_sec = slack * effective_offset if slack > 0 else 0.0
        clip_start_sec = start_sec + shift_sec

    # ------------------------------------------------------------------
    # 4. Map desired timestamps → native frame indices
    # ------------------------------------------------------------------
    timestamps = [clip_start_sec + i / target_fps for i in range(n_frames)]

    seen: set[int] = set()
    indices: list[int] = []
    for t in timestamps:
        idx = max(0, min(total_frames - 1, int(round(t * native_fps))))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)

    if len(indices) < n_frames:
        logger.warning(
            f"Frame deduplication reduced count from {n_frames} to {len(indices)} "
            f"(target_fps={target_fps}, native_fps={native_fps:.1f})"
        )

    if not indices:
        raise RuntimeError("Could not compute any valid frame indices.")

    # ------------------------------------------------------------------
    # 5. Fetch frames via random-access get_batch
    # ------------------------------------------------------------------
    frames = vr.get_batch(indices).asnumpy()

    # key metadata for logging/debugging
    metadata = dict(
        requested_frames=n_frames,
        actual_frames=len(indices),
        frame_indices=indices,
        timestamps=timestamps,
        available_sec=segment_duration,
        clip_start_sec=clip_start_sec,
        clip_end_sec=timestamps[-1],
        shift_sec=shift_sec,
    )

    return frames, metadata


def prepare_inputs_for_vllm(frames, messages, processor, model_fps=8, needs_video_metadata=True):
    """
    Prepare inputs for vLLM.

    Args:
        frames: Video frames tensor
        messages: List of message dicts (system + user) or single message dict
        processor: AutoProcessor instance
        model_fps: Frame rate to use for video metadata
        needs_video_metadata: Whether to include video metadata in the multi-modal data

    Returns:
        dict: Input format required by vLLM
    """
    # Ensure messages is a list
    if isinstance(messages, dict):
        messages = [messages]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if needs_video_metadata:
        video_meta = dict(
            total_num_frames=frames.shape[0],
            fps=model_fps,
            frames_indices=list(range(frames.shape[0])),
        )
        mm_data = dict(video=(frames, video_meta))
    else:
        mm_data = dict(video=frames)

    video_kwargs = dict(do_sample_frames=False)

    return dict(prompt=text, multi_modal_data=mm_data, mm_processor_kwargs=video_kwargs)
