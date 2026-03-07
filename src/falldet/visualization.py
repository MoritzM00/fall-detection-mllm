"""Visualization utilities for video data and evaluation results.

This module provides matplotlib-based visualization functions including
video frame grids and confusion matrix plots.
"""

import logging
from collections.abc import Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


PUBLICATION_TARGET_DEFAULTS: dict[str, dict[str, float | int]] = {
    "paper": {
        "text_width_pt": 246.0,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    },
    "thesis": {
        "text_width_pt": 427.43153,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    },
}


def _latex_pt_to_inches(value_pt: float) -> float:
    """Convert TeX points to inches."""
    return value_pt / 72.27


def _default_figsize() -> tuple[float, float]:
    """Return the current matplotlib default figure size as a tuple."""
    width, height = plt.rcParams["figure.figsize"]
    return float(width), float(height)


def compute_publication_figsize(
    *,
    target: str = "thesis",
    text_width_pt: float | None = None,
    width_fraction: float = 1.0,
    height_ratio: float = 0.66,
) -> tuple[float, float]:
    """Compute a manuscript-oriented matplotlib figure size in inches."""
    if target not in PUBLICATION_TARGET_DEFAULTS:
        raise ValueError(
            f"target must be one of {set(PUBLICATION_TARGET_DEFAULTS)}, got {target!r}."
        )

    default_text_width_pt = float(PUBLICATION_TARGET_DEFAULTS[target]["text_width_pt"])
    resolved_text_width_pt = default_text_width_pt if text_width_pt is None else text_width_pt
    if resolved_text_width_pt <= 0:
        raise ValueError(f"text_width_pt must be positive, got {resolved_text_width_pt!r}.")
    if width_fraction <= 0:
        raise ValueError(f"width_fraction must be positive, got {width_fraction!r}.")
    if height_ratio <= 0:
        raise ValueError(f"height_ratio must be positive, got {height_ratio!r}.")

    figure_width_in = _latex_pt_to_inches(resolved_text_width_pt) * width_fraction
    figure_height_in = figure_width_in * height_ratio
    return figure_width_in, figure_height_in


def set_publication_rc_defaults(
    *,
    target: str = "thesis",
    context: str = "paper",
    style: str = "whitegrid",
    palette: str = "colorblind",
    font_scale: float = 1.0,
    use_tex: bool = False,
    text_width_pt: float | None = None,
    width_fraction: float = 1.0,
    height_ratio: float = 0.66,
    rc: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], tuple[float, float]]:
    """Set seaborn and matplotlib defaults for paper-ready plots.

    The defaults favor manuscript-friendly figures, serif typography, vector-friendly
    output, and restrained grid styling suitable for LaTeX manuscripts.

    Args:
        target: Preset sizing target. Supported values are ``"paper"``
            and ``"thesis"``.
        context: Seaborn plotting context, usually ``"paper"``.
        style: Seaborn style preset.
        palette: Seaborn color palette name.
        font_scale: Seaborn font scaling factor.
        use_tex: Enable matplotlib's LaTeX text rendering. This requires
            a working LaTeX installation.
        text_width_pt: LaTeX text width in TeX points. When omitted,
            a target-specific default is used.
        width_fraction: Fraction of the LaTeX text width used for the
            default matplotlib figure width.
        height_ratio: Figure height divided by figure width.
        rc: Optional rcParams overrides applied after the defaults.

    Returns:
        The resolved rcParams dictionary applied to matplotlib and
        seaborn, along with the computed figure size tuple.
    """
    if target not in PUBLICATION_TARGET_DEFAULTS:
        raise ValueError(
            f"target must be one of {set(PUBLICATION_TARGET_DEFAULTS)}, got {target!r}."
        )

    preset = PUBLICATION_TARGET_DEFAULTS[target]
    figure_width_in, figure_height_in = compute_publication_figsize(
        target=target,
        text_width_pt=text_width_pt,
        width_fraction=width_fraction,
        height_ratio=height_ratio,
    )

    resolved_rc: dict[str, object] = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.figsize": (figure_width_in, figure_height_in),
        "figure.constrained_layout.use": False,
        "axes.titlesize": preset["axes.titlesize"],
        "axes.labelsize": preset["axes.labelsize"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": preset["legend.fontsize"],
        "legend.title_fontsize": preset["legend.title_fontsize"],
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "patch.linewidth": 0.8,
        "xtick.labelsize": preset["xtick.labelsize"],
        "ytick.labelsize": preset["ytick.labelsize"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": use_tex,
    }
    if use_tex:
        resolved_rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    if rc is not None:
        resolved_rc.update(dict(rc))

    sns.set_theme(
        context=context,
        style=style,
        palette=palette,
        font="serif",
        font_scale=font_scale,
        rc=resolved_rc,
    )
    plt.rcParams.update(resolved_rc)

    return resolved_rc, (figure_width_in, figure_height_in)


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
    title: str | None = None,
    cmap: str = "Blues",
    cbar: bool = False,
    annot_threshold: float | int | None = None,
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
        cbar: Whether to show the color bar.
        annot_threshold: Minimum cell value required before showing a
            text annotation. For normalized matrices, use proportions
            (for example ``0.01`` for 1%). For raw-count matrices, use
            counts.
        figsize: ``(width, height)`` in inches.  When ``None`` the
            current matplotlib rc default is used.
        ax: An existing :class:`~matplotlib.axes.Axes` to draw into.
            When ``None`` a new figure and axes are created.

    Returns:
        A ``(fig, ax)`` tuple with the matplotlib
        :class:`~matplotlib.figure.Figure` and
        :class:`~matplotlib.axes.Axes` containing the plot.

    Raises:
        ValueError: If *y_true* and *y_pred* differ in length, if
            either is empty, if *normalize* is not a recognized value,
            if *annot_threshold* is negative, or if *subset* contains
            labels not present in the data.

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
    if annot_threshold is not None and annot_threshold < 0:
        raise ValueError(f"annot_threshold must be non-negative, got {annot_threshold!r}.")

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
            if annot_threshold is not None and display_matrix[i, j] < annot_threshold:
                annot[i, j] = ""
                continue
            if normalize is None:
                # Raw counts -- show integer
                annot[i, j] = f"{int(display_matrix[i, j])}"
            else:
                # Normalized -- show proportion to two decimals
                annot[i, j] = f"{display_matrix[i, j]:.2f}"

    # ------------------------------------------------------------------
    # Figure / axes setup
    # ------------------------------------------------------------------
    if ax is None:
        if figsize is None:
            figsize = _default_figsize()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # ------------------------------------------------------------------
    # Draw heatmap
    # ------------------------------------------------------------------
    heatmap_kwargs = {
        "data": display_matrix,
        "annot": annot,
        "fmt": "",
        "cmap": cmap,
        "xticklabels": display_labels,
        "yticklabels": display_labels,
        "square": True,
        "cbar": cbar,
        "linewidths": 0.5,
        "linecolor": "white",
        "ax": ax,
    }
    if cbar:
        heatmap_kwargs["cbar_kws"] = {"label": "Proportion" if normalize else "Count"}

    sns.heatmap(
        **heatmap_kwargs,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title is not None:
        ax.set_title(title)

    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    return fig, ax


def plot_relative_confusion_matrix(
    y_true_a: list[str],
    y_pred_a: list[str],
    y_true_b: list[str],
    y_pred_b: list[str],
    subset: list[str] | None = None,
    title: str | None = None,
    cmap: str = "Blues",
    cbar: bool = False,
    figsize: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot a relative row-normalized confusion matrix for two runs.

    The matrix compares run *B* against run *A*. Cell color shows the
    absolute change in row-normalized confusion. The annotation shows
    the raw difference (B − A) in percentage points: positive means B's
    value is higher, negative means B's value is lower. On the diagonal,
    positive is good (higher recall); off-diagonal, negative is good
    (fewer misclassifications).
    """

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
    diff_pp = np.round((cm_b_norm - cm_a_norm) * 100).astype(int)
    magnitude_matrix = np.abs(diff_pp).astype(float)
    sign_annot = np.where(diff_pp == 0, "", np.vectorize(lambda v: f"{v:+d}")(diff_pp))

    display_labels, display_matrix = _resolve_display_subset(all_labels, magnitude_matrix, subset)
    _, display_annot = _resolve_display_subset(all_labels, sign_annot, subset)

    if ax is None:
        if figsize is None:
            figsize = _default_figsize()
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
        vmax=vmax if vmax > 0 else 100.0,
        cbar=cbar,
        cbar_kws={"label": "Absolute difference (pp)", "shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title is not None:
        ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if cbar:
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.0, vmax if vmax > 0 else 100.0])
        colorbar.set_ticklabels(["0", f"{vmax:.0f}" if vmax > 0 else "100"])

    fig.tight_layout()
    return fig, ax
