"""Metric-comparison plotting helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from falldet.data.video_dataset import idx2label
from falldet.metrics.base import compute_metrics

LabelSequence: TypeAlias = list[str] | list[int] | np.ndarray

COLORS: dict[str, str] = {
    "primary": "#0173B2",
    "secondary": "#DE8F05",
    "tertiary": "#029E73",
    "quaternary": "#CC78BC",
    "warning": "#CA9161",
    "error": "#D55E00",
    "neutral": "#56595c",
    "light_blue": "#6BAED6",
    "light_orange": "#FD8D3C",
}

DEFAULT_METRIC_ORDER: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "fall_f1",
    "fallen_f1",
)

DEFAULT_METRIC_LABELS: dict[str, str] = {
    "macro_f1": "Macro F1",
    "balanced_accuracy": "BAcc",
    "fall_f1": "Fall F1",
    "fallen_f1": "Fallen F1",
}


def _series_colors(num_series: int) -> list[str]:
    base_colors = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["tertiary"],
        COLORS["quaternary"],
        COLORS["light_blue"],
        COLORS["light_orange"],
        COLORS["warning"],
        COLORS["neutral"],
    ]
    if num_series <= len(base_colors):
        return base_colors[:num_series]
    return [base_colors[idx % len(base_colors)] for idx in range(num_series)]


def _normalize_label_value(label: str | int | np.integer) -> str:
    if isinstance(label, (int, np.integer)):
        label_idx = int(label)
        return idx2label.get(label_idx, f"unknown_{label_idx}")
    return str(label)


def _normalize_label_sequence(labels: LabelSequence) -> list[str]:
    return [_normalize_label_value(label) for label in labels]


@dataclass(slots=True)
class PredictionMetricInput:
    """Named label arrays used for metric recomputation and plotting."""

    name: str
    y_true: LabelSequence
    y_pred: LabelSequence


@dataclass(slots=True)
class MetricComparisonPanelSpec:
    """Configuration for one grouped-bar comparison panel."""

    metric_order: Sequence[str]
    metric_labels: Mapping[str, str] | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str = "Score (\\%)"
    scale: float = 100.0
    y_limits: tuple[float, float] | None = None
    xtick_rotation: float = 0.0
    xtick_horizontalalignment: str = "center"


def compute_metric_summary(
    y_true: LabelSequence,
    y_pred: LabelSequence,
    *,
    metric_order: Sequence[str] = DEFAULT_METRIC_ORDER,
    missing_metric_value: float | None = None,
) -> dict[str, float]:
    """Recompute and return a selected subset of evaluation metrics."""
    metrics = compute_metrics(y_pred=y_pred, y_true=y_true)
    missing_metrics = [metric_key for metric_key in metric_order if metric_key not in metrics]
    if missing_metrics and missing_metric_value is None:
        raise ValueError(f"Unknown metric keys requested: {missing_metrics}")

    summary: dict[str, float] = {}
    for metric_key in metric_order:
        if metric_key in metrics:
            summary[metric_key] = float(metrics[metric_key])
        elif missing_metric_value is not None:
            summary[metric_key] = float(missing_metric_value)
    return summary


def compute_metric_summaries(
    runs: Sequence[PredictionMetricInput],
    *,
    metric_order: Sequence[str] = DEFAULT_METRIC_ORDER,
    missing_metric_value: float | None = None,
) -> dict[str, dict[str, float]]:
    """Recompute metrics for multiple named prediction sets."""
    if not runs:
        raise ValueError("At least one prediction set is required.")

    summaries: dict[str, dict[str, float]] = {}
    for run in runs:
        summaries[run.name] = compute_metric_summary(
            y_true=run.y_true,
            y_pred=run.y_pred,
            metric_order=metric_order,
            missing_metric_value=missing_metric_value,
        )
    return summaries


def compute_label_distribution(
    labels: LabelSequence,
    *,
    label_order: Sequence[str] | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Compute label frequencies for a sequence of labels."""
    normalized_labels = _normalize_label_sequence(labels)
    if not normalized_labels:
        raise ValueError("labels must not be empty.")

    ordered_labels = (
        list(label_order) if label_order is not None else sorted(set(normalized_labels))
    )
    counts = {label: 0.0 for label in ordered_labels}
    for label in normalized_labels:
        if label in counts:
            counts[label] += 1.0

    if normalize:
        total_count = float(len(normalized_labels))
        return {label: count / total_count for label, count in counts.items()}
    return counts


def compute_label_distribution_summaries(
    runs: Sequence[PredictionMetricInput],
    *,
    label_order: Sequence[str] | None = None,
    actual_name: str = "Actual",
    normalize: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute actual and predicted label distributions for multiple runs."""
    if not runs:
        raise ValueError("At least one prediction set is required.")

    reference_labels = _normalize_label_sequence(runs[0].y_true)
    for run in runs[1:]:
        if _normalize_label_sequence(run.y_true) != reference_labels:
            raise ValueError("All runs must share the same ground-truth label sequence.")

    distributions = {
        actual_name: compute_label_distribution(
            runs[0].y_true,
            label_order=label_order,
            normalize=normalize,
        )
    }
    for run in runs:
        distributions[run.name] = compute_label_distribution(
            run.y_pred,
            label_order=label_order,
            normalize=normalize,
        )
    return distributions


def plot_metric_comparison(
    run_metrics: Mapping[str, Mapping[str, float]],
    *,
    metric_order: Sequence[str] = DEFAULT_METRIC_ORDER,
    metric_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Score (\\%)",
    scale: float = 100.0,
    figsize: tuple[float, float] | None = None,
    series_colors: Sequence[str] | None = None,
    xtick_rotation: float = 0.0,
    xtick_horizontalalignment: str = "center",
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    apply_tight_layout: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot selected metrics side by side for multiple runs."""
    if not run_metrics:
        raise ValueError("run_metrics must not be empty.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}.")

    resolved_metric_labels = dict(DEFAULT_METRIC_LABELS)
    if metric_labels is not None:
        resolved_metric_labels.update(metric_labels)

    run_names = list(run_metrics.keys())
    metric_names = list(metric_order)
    missing_metrics = {
        run_name: [
            metric_name for metric_name in metric_names if metric_name not in run_metrics[run_name]
        ]
        for run_name in run_names
    }
    missing_metrics = {
        run_name: missing for run_name, missing in missing_metrics.items() if missing
    }
    if missing_metrics:
        raise ValueError(f"Each run must contain all requested metrics: {missing_metrics}")

    values = np.array(
        [
            [float(run_metrics[run_name][metric_name]) * scale for metric_name in metric_names]
            for run_name in run_names
        ],
        dtype=float,
    )

    if ax is None:
        resolved_figsize = figsize if figsize is not None else (10.0, 4.0)
        fig, ax = plt.subplots(figsize=resolved_figsize)
    else:
        fig = ax.get_figure()

    x_positions = np.arange(len(metric_names), dtype=float)
    group_width = 0.8
    bar_width = group_width / max(len(run_names), 1)
    offsets = (np.arange(len(run_names), dtype=float) - (len(run_names) - 1) / 2) * bar_width
    colors = list(series_colors) if series_colors is not None else _series_colors(len(run_names))
    if len(colors) < len(run_names):
        raise ValueError(
            f"series_colors must provide at least {len(run_names)} colors, got {len(colors)}."
        )

    for idx, run_name in enumerate(run_names):
        ax.bar(
            x_positions + offsets[idx],
            values[idx],
            width=bar_width * 0.92,
            label=run_name,
            color=colors[idx],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    x_tick_labels = [
        resolved_metric_labels.get(metric_name, metric_name) for metric_name in metric_names
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels, rotation=xtick_rotation, ha=xtick_horizontalalignment)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25, color=COLORS["neutral"])
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        max_value = float(np.max(values)) if values.size else 0.0
        upper_limit = min(100.0, max(5.0, max_value * 1.15))
        ax.set_ylim(0.0, upper_limit)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        legend = ax.legend(
            loc="best",
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=9,
            edgecolor=COLORS["neutral"],
            framealpha=1.0,
        )
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_facecolor("white")

    if apply_tight_layout:
        fig.tight_layout()
    return fig, ax


def plot_metric_comparison_panels(
    run_metrics: Mapping[str, Mapping[str, float]],
    *,
    panel_specs: Sequence[MetricComparisonPanelSpec],
    figsize: tuple[float, float] | None = None,
    series_colors: Sequence[str] | None = None,
    sharey: bool = False,
    legend_panel_index: int = 0,
    wspace: float = 0.35,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot multiple grouped-bar comparison panels in one figure."""
    if not panel_specs:
        raise ValueError("panel_specs must not be empty.")

    resolved_figsize = figsize if figsize is not None else (12.0, 4.0)
    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=resolved_figsize,
        sharey=sharey,
        squeeze=False,
    )
    axes_array = axes.ravel()

    for idx, panel_spec in enumerate(panel_specs):
        plot_metric_comparison(
            run_metrics,
            metric_order=panel_spec.metric_order,
            metric_labels=panel_spec.metric_labels,
            title=panel_spec.title,
            xlabel=panel_spec.xlabel,
            ylabel=panel_spec.ylabel,
            scale=panel_spec.scale,
            series_colors=series_colors,
            xtick_rotation=panel_spec.xtick_rotation,
            xtick_horizontalalignment=panel_spec.xtick_horizontalalignment,
            y_limits=panel_spec.y_limits,
            show_legend=idx == legend_panel_index,
            apply_tight_layout=False,
            ax=axes_array[idx],
        )

    fig.tight_layout()
    fig.subplots_adjust(wspace=wspace)
    return fig, axes_array


def plot_metric_comparison_from_predictions(
    runs: Sequence[PredictionMetricInput],
    *,
    metric_order: Sequence[str] = DEFAULT_METRIC_ORDER,
    metric_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    ylabel: str = "Score (\\%)",
    scale: float = 100.0,
    figsize: tuple[float, float] | None = None,
    series_colors: Sequence[str] | None = None,
    xtick_rotation: float = 0.0,
    xtick_horizontalalignment: str = "center",
    missing_metric_value: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Recompute metrics from label arrays and plot them side by side."""
    run_metrics = compute_metric_summaries(
        runs,
        metric_order=metric_order,
        missing_metric_value=missing_metric_value,
    )
    return plot_metric_comparison(
        run_metrics,
        metric_order=metric_order,
        metric_labels=metric_labels,
        title=title,
        xlabel=None,
        ylabel=ylabel,
        scale=scale,
        figsize=figsize,
        series_colors=series_colors,
        xtick_rotation=xtick_rotation,
        xtick_horizontalalignment=xtick_horizontalalignment,
        show_legend=True,
        ax=ax,
    )


def plot_metric_comparison_panels_from_predictions(
    runs: Sequence[PredictionMetricInput],
    *,
    panel_specs: Sequence[MetricComparisonPanelSpec],
    figsize: tuple[float, float] | None = None,
    series_colors: Sequence[str] | None = None,
    sharey: bool = False,
    legend_panel_index: int = 0,
    wspace: float = 0.35,
    missing_metric_value: float | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Recompute metrics from label arrays and plot multiple comparison panels."""
    combined_metric_order: list[str] = []
    for panel_spec in panel_specs:
        for metric_name in panel_spec.metric_order:
            if metric_name not in combined_metric_order:
                combined_metric_order.append(metric_name)

    run_metrics = compute_metric_summaries(
        runs,
        metric_order=combined_metric_order,
        missing_metric_value=missing_metric_value,
    )
    return plot_metric_comparison_panels(
        run_metrics,
        panel_specs=panel_specs,
        figsize=figsize,
        series_colors=series_colors,
        sharey=sharey,
        legend_panel_index=legend_panel_index,
        wspace=wspace,
    )


def plot_label_distribution_comparison_from_predictions(
    runs: Sequence[PredictionMetricInput],
    *,
    label_order: Sequence[str] | None = None,
    label_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    ylabel: str = "Frequency (\\%)",
    scale: float = 100.0,
    figsize: tuple[float, float] | None = None,
    actual_name: str = "Actual",
    actual_color: str = COLORS["neutral"],
    xtick_rotation: float = 0.0,
    xtick_horizontalalignment: str = "center",
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot actual and predicted label distributions for multiple runs."""
    ordered_labels = list(label_order) if label_order is not None else None
    distributions = compute_label_distribution_summaries(
        runs,
        label_order=ordered_labels,
        actual_name=actual_name,
        normalize=True,
    )
    series_colors = [actual_color, *_series_colors(len(runs))]
    return plot_metric_comparison(
        distributions,
        metric_order=ordered_labels
        if ordered_labels is not None
        else tuple(distributions[actual_name]),
        metric_labels=label_labels,
        title=title,
        ylabel=ylabel,
        scale=scale,
        figsize=figsize,
        series_colors=series_colors,
        xtick_rotation=xtick_rotation,
        xtick_horizontalalignment=xtick_horizontalalignment,
        ax=ax,
    )


__all__ = [
    "COLORS",
    "DEFAULT_METRIC_LABELS",
    "DEFAULT_METRIC_ORDER",
    "MetricComparisonPanelSpec",
    "PredictionMetricInput",
    "compute_label_distribution",
    "compute_label_distribution_summaries",
    "compute_metric_summary",
    "compute_metric_summaries",
    "plot_label_distribution_comparison_from_predictions",
    "plot_metric_comparison",
    "plot_metric_comparison_panels",
    "plot_metric_comparison_panels_from_predictions",
    "plot_metric_comparison_from_predictions",
]
