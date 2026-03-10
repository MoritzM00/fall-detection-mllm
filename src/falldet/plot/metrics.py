"""Metric-comparison plotting helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


@dataclass(slots=True)
class PredictionMetricInput:
    """Named label arrays used for metric recomputation and plotting."""

    name: str
    y_true: LabelSequence
    y_pred: LabelSequence


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


def plot_metric_comparison(
    run_metrics: Mapping[str, Mapping[str, float]],
    *,
    metric_order: Sequence[str] = DEFAULT_METRIC_ORDER,
    metric_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    ylabel: str = "Score (\\%)",
    scale: float = 100.0,
    figsize: tuple[float, float] | None = None,
    xtick_rotation: float = 0.0,
    xtick_horizontalalignment: str = "center",
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
    colors = _series_colors(len(run_names))

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
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25, color=COLORS["neutral"])
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    max_value = float(np.max(values)) if values.size else 0.0
    upper_limit = min(100.0, max(5.0, max_value * 1.15))
    ax.set_ylim(0.0, upper_limit)

    if title is not None:
        ax.set_title(title)

    if len(run_names) > 1:
        legend = ax.legend(
            loc="best",
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=9,
            edgecolor=COLORS["neutral"],
            framealpha=1.0,
        )
    else:
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

    fig.tight_layout()
    return fig, ax


def plot_metric_comparison_from_predictions(
    runs: Sequence[PredictionMetricInput],
    *,
    metric_order: Sequence[str] = DEFAULT_METRIC_ORDER,
    metric_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    ylabel: str = "Score (\\%)",
    scale: float = 100.0,
    figsize: tuple[float, float] | None = None,
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
        ylabel=ylabel,
        scale=scale,
        figsize=figsize,
        xtick_rotation=xtick_rotation,
        xtick_horizontalalignment=xtick_horizontalalignment,
        ax=ax,
    )


__all__ = [
    "COLORS",
    "DEFAULT_METRIC_LABELS",
    "DEFAULT_METRIC_ORDER",
    "PredictionMetricInput",
    "compute_metric_summary",
    "compute_metric_summaries",
    "plot_metric_comparison",
    "plot_metric_comparison_from_predictions",
]
