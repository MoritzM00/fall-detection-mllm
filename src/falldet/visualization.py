import logging

import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def video_to_image_grid(
    video: torch.Tensor, nrow: int | None = None, padding: int = 2, normalize: bool = True
) -> torch.Tensor:
    """
    Convert a video tensor to an image grid.

    Args:
        video: Tensor of shape (T, C, H, W) where T is number of frames
        nrow: Number of images per row. If None, uses ceil(sqrt(T))
        padding: Padding between images
        normalize: Whether to normalize the output to [0, 1]

    Returns:
        Image grid tensor of shape (C, H', W') suitable for display
    """
    T, C, H, W = video.shape

    if nrow is None:
        nrow = int(torch.ceil(torch.sqrt(torch.tensor(T, dtype=torch.float))).item())

    grid = vutils.make_grid(video.float(), nrow=nrow, padding=padding, normalize=normalize)

    return grid


def visualize_video(
    video: torch.Tensor | None = None,
    dataset=None,
    idx: int | None = None,
    figsize: tuple[int, int] = (12, 12),
    **kwargs,
):
    """
    Visualize a video segment at the given index.

    Args:
        video: Optional tensor of shape (T, C, H, W) to visualize directly. If None, will load from dataset and idx.
        dataset: Dataset to load from if video is not provided
        idx: Index of the video segment to visualize if video is not provided
        figsize: Size of the output image in inches
        kwargs: Additional keyword arguments to pass to video_to_image_grid

    Returns:
        fig, ax: Matplotlib figure and axis containing the visualization
    """
    if video is None:
        assert dataset is not None and idx is not None, (
            "Must provide either video tensor or dataset and index"
        )
        segment = dataset[idx]
        logger.info(f"Label: {segment['label_str']}")
        video = segment["video"]  # shape (T, C, H, W)
    else:
        assert isinstance(video, torch.Tensor) and video.ndim == 4, (
            "Video must be a tensor of shape (T, C, H, W)"
        )

    grid_tensor = video_to_image_grid(video, **kwargs)  # shape (C, H', W')
    grid_image = grid_tensor.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid_image)
    ax.axis("off")

    return fig, ax
