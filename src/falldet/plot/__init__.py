"""Public plotting API for confusion matrices and video visualizations."""

from .base import compute_publication_figsize, set_publication_rc_defaults
from .confusion import plot_confusion_matrix, plot_relative_confusion_matrix
from .metrics import (
    DEFAULT_METRIC_LABELS,
    DEFAULT_METRIC_ORDER,
    PredictionMetricInput,
    compute_metric_summaries,
    compute_metric_summary,
    plot_metric_comparison,
    plot_metric_comparison_from_predictions,
)
from .video import video_to_image_grid, visualize_video

__all__ = [
    "DEFAULT_METRIC_LABELS",
    "DEFAULT_METRIC_ORDER",
    "PredictionMetricInput",
    "compute_metric_summaries",
    "compute_metric_summary",
    "compute_publication_figsize",
    "plot_metric_comparison",
    "plot_metric_comparison_from_predictions",
    "plot_confusion_matrix",
    "plot_relative_confusion_matrix",
    "set_publication_rc_defaults",
    "video_to_image_grid",
    "visualize_video",
]
