"""Visualization utilities for video data and evaluation results.

This module provides matplotlib-based visualization functions including
video frame grids and confusion matrix plots.
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def _validate_confusion_inputs(y_true: list[str], y_pred: list[str]) -> None:
    """Validate a pair of label arrays for confusion-matrix plotting."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}."
        )


def _resolve_display_subset(
    all_labels: list[str],
    display_values: np.ndarray,
    subset: list[str] | None,
) -> tuple[list[str], np.ndarray]:
    """Select the subset of labels to display from a full matrix."""
    if subset is not None:
        unknown = set(subset) - set(all_labels)
        if unknown:
            raise ValueError(f"subset contains labels not present in the data: {unknown}")

        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        subset_indices = [label_to_idx[s] for s in subset]
        return list(subset), display_values[np.ix_(subset_indices, subset_indices)]

    return list(all_labels), display_values


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


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    normalize: str | None = None,
    subset: list[str] | None = None,
    title: str | None = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot a confusion matrix computed from string label arrays.

    The confusion matrix is always computed over all classes present in
    ``y_true`` and ``y_pred``.  When *subset* is given, only the
    requested classes are shown in the plot (adjacently, in the order
    provided).  Leading and/or trailing ``...`` labels indicate that
    omitted classes exist outside the displayed range.

    Args:
        y_true: Ground-truth labels (e.g. ``["fall", "walk", ...]``).
        y_pred: Predicted labels, same length as *y_true*.
        normalize: How to normalize cell values.  One of

            * ``None`` -- raw counts (default).
            * ``"true"`` -- normalize over each **row** (true label),
              so that rows sum to 1.  Equivalent to recall per class.
            * ``"pred"`` -- normalize over each **column** (predicted
              label), so that columns sum to 1.
            * ``"all"`` -- normalize over the entire matrix so that all
              cells sum to 1.

        subset: Optional list of class names to display.  The order of
            classes in the plot matches the order in this list.  Classes
            not listed are omitted from the visualization but still
            contribute to the computed matrix values.
        title: Title shown above the plot.
        cmap: Matplotlib / seaborn colormap name used for the heatmap.
        figsize: ``(width, height)`` in inches.  When ``None`` the size
            is derived from the number of displayed classes.
        ax: An existing :class:`~matplotlib.axes.Axes` to draw into.
            When ``None`` a new figure and axes are created.

    Returns:
        A ``(fig, ax)`` tuple with the matplotlib
        :class:`~matplotlib.figure.Figure` and
        :class:`~matplotlib.axes.Axes` containing the plot.

    Raises:
        ValueError: If *y_true* and *y_pred* differ in length, if
            either is empty, if *normalize* is not a recognized value,
            or if *subset* contains labels not present in the data.

    Example::

        y_true = ["fall", "walk", "fallen", "walk", "sitting"]
        y_pred = ["fall", "walk", "walk",   "walk", "fall"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")
        fig.savefig("cm.pdf", bbox_inches="tight")
    """
    import seaborn as sns

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    _validate_confusion_inputs(y_true, y_pred)
    valid_normalize = {None, "true", "pred", "all"}
    if normalize not in valid_normalize:
        raise ValueError(f"normalize must be one of {valid_normalize}, got {normalize!r}.")

    # ------------------------------------------------------------------
    # Compute the full confusion matrix (with optional normalization)
    # ------------------------------------------------------------------
    all_labels: list[str] = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize=normalize)
    display_values = cm.astype(np.float64)

    # ------------------------------------------------------------------
    # Subset selection
    # ------------------------------------------------------------------
    display_labels, display_matrix = _resolve_display_subset(all_labels, display_values, subset)

    # ------------------------------------------------------------------
    # Annotation formatting
    # ------------------------------------------------------------------
    annot = np.empty_like(display_matrix, dtype=object)
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            if normalize is None:
                # Raw counts -- show integer
                annot[i, j] = f"{int(display_matrix[i, j])}"
            else:
                # Normalized -- show proportion to two decimals
                annot[i, j] = f"{display_matrix[i, j]:.2f}"

    # ------------------------------------------------------------------
    # Figure / axes setup
    # ------------------------------------------------------------------
    n = len(display_labels)
    if ax is None:
        if figsize is None:
            side = max(4.0, 0.6 * n + 2.0)
            figsize = (side, side)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # ------------------------------------------------------------------
    # Draw heatmap
    # ------------------------------------------------------------------
    sns.heatmap(
        display_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    if title is not None:
        ax.set_title(title)

    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    return fig, ax


def plot_relative_confusion_matrix(
    y_true_a: list[str],
    y_pred_a: list[str],
    y_true_b: list[str],
    y_pred_b: list[str],
    subset: list[str] | None = None,
    title: str | None = "Relative Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot a relative row-normalized confusion matrix for two runs.

    The matrix compares run *B* against run *A*. Cell color shows the
    absolute change in row-normalized confusion. On the diagonal, a
    larger value is better; off-diagonal, a smaller value is better
    because those cells represent misclassifications. The annotation is
    ``+`` when run *B* is better, ``-`` when run *A* is better, and
    empty when the cell is unchanged.
    """
    import seaborn as sns

    _validate_confusion_inputs(y_true_a, y_pred_a)
    _validate_confusion_inputs(y_true_b, y_pred_b)

    if y_true_a != y_true_b:
        raise ValueError(
            "y_true_a and y_true_b must be identical — comparing runs "
            "evaluated on different ground-truth samples is not meaningful."
        )

    all_labels: list[str] = sorted(set(y_true_a) | set(y_pred_a) | set(y_true_b) | set(y_pred_b))
    cm_a_norm = confusion_matrix(y_true_a, y_pred_a, labels=all_labels, normalize="true")
    cm_b_norm = confusion_matrix(y_true_b, y_pred_b, labels=all_labels, normalize="true")
    diff_matrix = cm_b_norm - cm_a_norm

    magnitude_matrix = np.abs(diff_matrix)
    sign_annot = np.empty_like(magnitude_matrix, dtype=object)
    tol = 1e-12

    for i in range(magnitude_matrix.shape[0]):
        for j in range(magnitude_matrix.shape[1]):
            diff = diff_matrix[i, j]
            if abs(diff) <= tol:
                sign_annot[i, j] = ""
            elif i == j:
                # Diagonal: higher is better for B → positive pp
                pct = round(diff * 100)
                sign_annot[i, j] = f"{pct:+d}"
            else:
                # Off-diagonal: lower is better for B → flip sign
                pct = round(-diff * 100)
                sign_annot[i, j] = f"{pct:+d}"

    display_labels, display_matrix = _resolve_display_subset(all_labels, magnitude_matrix, subset)
    _, display_annot = _resolve_display_subset(all_labels, sign_annot, subset)

    n = len(display_labels)
    if ax is None:
        if figsize is None:
            side = max(4.0, 0.6 * n + 2.0)
            figsize = (side, side)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vmax = float(display_matrix.max()) if display_matrix.size else 0.0
    sns.heatmap(
        display_matrix,
        annot=display_annot,
        fmt="",
        cmap=cmap,
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
        vmin=0.0,
        vmax=vmax if vmax > 0 else 1.0,
        cbar_kws={"label": "Absolute difference (pp)", "shrink": 0.85},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.0, vmax if vmax > 0 else 1.0])
    cbar.set_ticklabels([f"{0.0:.2f}", f"{vmax:.2f}" if vmax > 0 else f"{1.0:.2f}"])

    fig.tight_layout()
    return fig, ax
