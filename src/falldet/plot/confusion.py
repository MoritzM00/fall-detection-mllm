"""Confusion-matrix plotting helpers and public plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .base import (
    NORMAL_CONFUSION_OFFDIAGONAL_CMAP,
    RELATIVE_CONFUSION_DIAGONAL_CMAP,
    _default_figsize,
)


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


def _annotation_text_color_for_rgba(rgba: tuple[float, float, float, float]) -> str:
    """Choose black or white annotation text based on cell luminance."""
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.45 else "black"


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
    ``y_true`` and ``y_pred``. When *subset* is given, only the
    requested classes are shown in the plot, in the order provided.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels, same length as *y_true*.
        normalize: One of ``None``, ``"true"``, ``"pred"``, or ``"all"``.
        subset: Optional list of class names to display.
        title: Title shown above the plot.
        cmap: Colormap name used for diagonal cells.
        cbar: Whether to show the color bar(s).
        annot_threshold: Minimum cell value required before showing a text annotation.
        figsize: ``(width, height)`` in inches.
        ax: Existing axes to draw into.

    Returns:
        A ``(fig, ax)`` tuple.
    """
    _validate_confusion_inputs(y_true, y_pred)
    valid_normalize = {None, "true", "pred", "all"}
    if normalize not in valid_normalize:
        raise ValueError(f"normalize must be one of {valid_normalize}, got {normalize!r}.")
    if annot_threshold is not None and annot_threshold < 0:
        raise ValueError(f"annot_threshold must be non-negative, got {annot_threshold!r}.")

    all_labels: list[str] = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize=normalize)
    display_values = cm.astype(np.float64)

    display_labels, display_matrix = _resolve_display_subset(all_labels, display_values, subset)

    annot = np.empty_like(display_matrix, dtype=object)
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            if annot_threshold is not None and display_matrix[i, j] < annot_threshold:
                annot[i, j] = ""
                continue
            if normalize is None:
                annot[i, j] = f"{int(display_matrix[i, j])}"
            else:
                annot[i, j] = f"{display_matrix[i, j]:.2f}"

    n_labels = len(display_labels)
    diagonal_mask = ~np.eye(n_labels, dtype=bool)
    off_diagonal_mask = np.eye(n_labels, dtype=bool)

    diagonal_values = display_matrix
    off_diagonal_values = display_matrix
    diagonal_vmax = float(np.max(np.diag(display_matrix))) if n_labels else 0.0
    diagonal_vmax = diagonal_vmax if diagonal_vmax > 0 else 1.0
    if n_labels > 1:
        off_diagonal_vmax = float(np.max(off_diagonal_values[~off_diagonal_mask]))
    else:
        off_diagonal_vmax = 0.0
    off_diagonal_vmax = off_diagonal_vmax if off_diagonal_vmax > 0 else 1.0

    diagonal_cax = None
    off_diagonal_cax = None
    uses_gridspec_colorbars = cbar and ax is None
    if ax is None:
        if figsize is None:
            figsize = _default_figsize()
        if uses_gridspec_colorbars:
            fig = plt.figure(figsize=figsize)
            outer_gs = matplotlib.gridspec.GridSpec(
                1,
                2,
                figure=fig,
                width_ratios=[24, 1.8],
                wspace=0.2,
            )
            cbar_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer_gs[1],
                hspace=0.35,
            )
            ax = fig.add_subplot(outer_gs[0])
            diagonal_cax = fig.add_subplot(cbar_gs[0])
            off_diagonal_cax = fig.add_subplot(cbar_gs[1])
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        diagonal_values,
        mask=diagonal_mask,
        annot=False,
        fmt="",
        cmap=cmap,
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
        vmin=0.0,
        vmax=diagonal_vmax,
        cbar=diagonal_cax is not None,
        cbar_ax=diagonal_cax,
        cbar_kws={"ticks": [0.0, diagonal_vmax]} if diagonal_cax is not None else None,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    diagonal_mesh = ax.collections[-1]

    off_diagonal_mesh = None
    if n_labels > 1:
        sns.heatmap(
            off_diagonal_values,
            mask=off_diagonal_mask,
            annot=False,
            fmt="",
            cmap=NORMAL_CONFUSION_OFFDIAGONAL_CMAP,
            xticklabels=display_labels,
            yticklabels=display_labels,
            square=True,
            vmin=0.0,
            vmax=off_diagonal_vmax,
            cbar=off_diagonal_cax is not None,
            cbar_ax=off_diagonal_cax,
            cbar_kws={"ticks": [0.0, off_diagonal_vmax]} if off_diagonal_cax is not None else None,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        off_diagonal_mesh = ax.collections[-1]
    elif off_diagonal_cax is not None:
        off_diagonal_cax.set_visible(False)

    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                rgba = diagonal_mesh.cmap(diagonal_mesh.norm(display_matrix[i, j]))
            elif off_diagonal_mesh is not None:
                rgba = off_diagonal_mesh.cmap(off_diagonal_mesh.norm(display_matrix[i, j]))
            else:
                rgba = (1.0, 1.0, 1.0, 1.0)

            ax.text(
                j + 0.5,
                i + 0.5,
                annot[i, j],
                ha="center",
                va="center",
                color=_annotation_text_color_for_rgba(rgba),
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title is not None:
        ax.set_title(title)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if cbar:
        value_label = "Proportion" if normalize else "Count"
        diagonal_ticklabels = ["0", f"{diagonal_vmax:.2f}" if normalize else f"{diagonal_vmax:.0f}"]
        off_diagonal_ticklabels = [
            "0",
            f"{off_diagonal_vmax:.2f}" if normalize else f"{off_diagonal_vmax:.0f}",
        ]
        if uses_gridspec_colorbars:
            diagonal_colorbar = diagonal_mesh.colorbar
            diagonal_colorbar.set_ticks([0.0, diagonal_vmax])
            diagonal_colorbar.set_ticklabels(diagonal_ticklabels)
            diagonal_cax.set_title("Diagonal", pad=6)
            if off_diagonal_mesh is not None and off_diagonal_cax is not None:
                off_diagonal_colorbar = off_diagonal_mesh.colorbar
                off_diagonal_colorbar.set_ticks([0.0, off_diagonal_vmax])
                off_diagonal_colorbar.set_ticklabels(off_diagonal_ticklabels)
                off_diagonal_cax.set_title("Off-diagonal", pad=6)
        else:
            diagonal_colorbar = fig.colorbar(diagonal_mesh, ax=ax, fraction=0.046, pad=0.04)
            diagonal_colorbar.set_label(f"Diagonal {value_label}")
            diagonal_colorbar.set_ticks([0.0, diagonal_vmax])
            diagonal_colorbar.set_ticklabels(diagonal_ticklabels)

            if off_diagonal_mesh is not None:
                off_diagonal_colorbar = fig.colorbar(
                    off_diagonal_mesh,
                    ax=ax,
                    fraction=0.046,
                    pad=0.12,
                )
                off_diagonal_colorbar.set_label(f"Off-diagonal {value_label}")
                off_diagonal_colorbar.set_ticks([0.0, off_diagonal_vmax])
                off_diagonal_colorbar.set_ticklabels(off_diagonal_ticklabels)

    if not uses_gridspec_colorbars:
        fig.tight_layout()
    return fig, ax


def plot_relative_confusion_matrix(
    y_true_a: list[str],
    y_pred_a: list[str],
    y_true_b: list[str],
    y_pred_b: list[str],
    subset: list[str] | None = None,
    title: str | None = None,
    cmap: str | matplotlib.colors.Colormap = RELATIVE_CONFUSION_DIAGONAL_CMAP,
    cbar: bool = False,
    figsize: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot a relative row-normalized confusion matrix for two runs.

    The matrix compares run *B* against run *A*. Cell color shows the
    change in row-normalized confusion. The annotation shows the raw
    difference (B - A) in percentage points.
    """
    _validate_confusion_inputs(y_true_a, y_pred_a)
    _validate_confusion_inputs(y_true_b, y_pred_b)

    if y_true_a != y_true_b:
        raise ValueError(
            "y_true_a and y_true_b must be identical - comparing runs "
            "evaluated on different ground-truth samples is not meaningful."
        )

    all_labels: list[str] = sorted(set(y_true_a) | set(y_pred_a) | set(y_true_b) | set(y_pred_b))
    cm_a_norm = confusion_matrix(y_true_a, y_pred_a, labels=all_labels, normalize="true")
    cm_b_norm = confusion_matrix(y_true_b, y_pred_b, labels=all_labels, normalize="true")
    diff_pp = np.round((cm_b_norm - cm_a_norm) * 100).astype(int)
    signed_matrix = diff_pp.astype(float)
    sign_annot = np.where(diff_pp == 0, "", np.vectorize(lambda v: f"{v:+d}")(diff_pp))

    display_labels, display_matrix = _resolve_display_subset(all_labels, signed_matrix, subset)
    _, display_annot = _resolve_display_subset(all_labels, sign_annot, subset)

    if ax is None:
        if figsize is None:
            figsize = _default_figsize()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_labels = len(display_labels)
    diagonal_mask = ~np.eye(n_labels, dtype=bool)
    off_diagonal_mask = np.eye(n_labels, dtype=bool)

    diagonal_values = display_matrix
    diagonal_vmax = float(np.max(np.abs(np.diag(display_matrix)))) if n_labels else 0.0
    diagonal_vmax = diagonal_vmax if diagonal_vmax > 0 else 100.0
    off_diagonal_values = display_matrix
    if n_labels > 1:
        off_diagonal_vmax = float(np.max(np.abs(off_diagonal_values[~off_diagonal_mask])))
    else:
        off_diagonal_vmax = 0.0
    off_diagonal_vmax = off_diagonal_vmax if off_diagonal_vmax > 0 else 100.0

    sns.heatmap(
        diagonal_values,
        mask=diagonal_mask,
        annot=False,
        fmt="",
        cmap=cmap,
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
        norm=matplotlib.colors.TwoSlopeNorm(
            vmin=-diagonal_vmax,
            vcenter=0.0,
            vmax=diagonal_vmax,
        ),
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    diagonal_mesh = ax.collections[-1]

    off_diagonal_mesh = None
    if n_labels > 1:
        sns.heatmap(
            off_diagonal_values,
            mask=off_diagonal_mask,
            annot=False,
            fmt="",
            cmap="RdBu_r",
            xticklabels=display_labels,
            yticklabels=display_labels,
            square=True,
            norm=matplotlib.colors.TwoSlopeNorm(
                vmin=-off_diagonal_vmax,
                vcenter=0.0,
                vmax=off_diagonal_vmax,
            ),
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        off_diagonal_mesh = ax.collections[-1]

    def _annotation_color(value: float, *, is_diagonal: bool) -> str:
        if is_diagonal:
            rgba = diagonal_mesh.cmap(diagonal_mesh.norm(value))
        elif off_diagonal_mesh is not None:
            rgba = off_diagonal_mesh.cmap(off_diagonal_mesh.norm(value))
        else:
            rgba = (1.0, 1.0, 1.0, 1.0)
        return _annotation_text_color_for_rgba(rgba)

    for i in range(n_labels):
        for j in range(n_labels):
            is_diagonal = i == j
            ax.text(
                j + 0.5,
                i + 0.5,
                display_annot[i, j],
                ha="center",
                va="center",
                color=_annotation_color(display_matrix[i, j], is_diagonal=is_diagonal),
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title is not None:
        ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if cbar:
        diagonal_colorbar = fig.colorbar(diagonal_mesh, ax=ax, fraction=0.046, pad=0.04)
        diagonal_colorbar.set_label("Diagonal Δ (pp)")
        diagonal_colorbar.set_ticks([-diagonal_vmax, 0.0, diagonal_vmax])
        diagonal_colorbar.set_ticklabels([f"-{diagonal_vmax:.0f}", "0", f"{diagonal_vmax:.0f}"])

        if off_diagonal_mesh is not None:
            off_diagonal_colorbar = fig.colorbar(off_diagonal_mesh, ax=ax, fraction=0.046, pad=0.12)
            off_diagonal_colorbar.set_label("Off-diagonal Δ (pp)")
            off_diagonal_colorbar.set_ticks([-off_diagonal_vmax, 0.0, off_diagonal_vmax])
            off_diagonal_colorbar.set_ticklabels(
                [f"-{off_diagonal_vmax:.0f}", "0", f"{off_diagonal_vmax:.0f}"]
            )

    fig.tight_layout()
    return fig, ax


__all__ = ["plot_confusion_matrix", "plot_relative_confusion_matrix"]
