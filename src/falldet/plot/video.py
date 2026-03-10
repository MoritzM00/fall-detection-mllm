"""Video visualization helpers."""

import logging

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

logger = logging.getLogger(__name__)


def video_to_image_grid(
    video: torch.Tensor, nrow: int | None = None, padding: int = 2, normalize: bool = True
) -> torch.Tensor:
    """Convert a video tensor to an image grid."""
    num_frames = video.shape[0]

    if nrow is None:
        nrow = int(torch.ceil(torch.sqrt(torch.tensor(num_frames, dtype=torch.float))).item())

    grid = vutils.make_grid(video.float(), nrow=nrow, padding=padding, normalize=normalize)
    return grid


def visualize_video(
    video: torch.Tensor | None = None,
    dataset=None,
    idx: int | None = None,
    figsize: tuple[int, int] = (12, 12),
    **kwargs,
):
    """Visualize a video segment at the given index."""
    if video is None:
        assert dataset is not None and idx is not None, (
            "Must provide either video tensor or dataset and index"
        )
        segment = dataset[idx]
        logger.info("Label: %s", segment["label_str"])
        video = segment["video"]
    else:
        assert isinstance(video, torch.Tensor) and video.ndim == 4, (
            "Video must be a tensor of shape (T, C, H, W)"
        )

    grid_tensor = video_to_image_grid(video, **kwargs)
    grid_image = grid_tensor.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid_image)
    ax.axis("off")

    return fig, ax


__all__ = ["video_to_image_grid", "visualize_video"]
