"""Public plotting API for confusion matrices and video visualizations."""

from .base import compute_publication_figsize, set_publication_rc_defaults
from .confusion import (
    cluster_confusion_labels,
    plot_confusion_matrix,
    plot_relative_confusion_matrix,
)
from .metrics import (
    COLORS,
    DEFAULT_METRIC_LABELS,
    DEFAULT_METRIC_ORDER,
    MetricComparisonPanelSpec,
    PredictionMetricInput,
    compute_label_distribution,
    compute_label_distribution_summaries,
    compute_metric_summaries,
    compute_metric_summary,
    plot_label_distribution_comparison_from_predictions,
    plot_metric_comparison,
    plot_metric_comparison_from_predictions,
    plot_metric_comparison_panels,
    plot_metric_comparison_panels_from_predictions,
)
from .video import video_to_image_grid, visualize_video

__all__ = [
    "COLORS",
    "cluster_confusion_labels",
    "DEFAULT_METRIC_LABELS",
    "DEFAULT_METRIC_ORDER",
    "MetricComparisonPanelSpec",
    "PredictionMetricInput",
    "compute_label_distribution",
    "compute_label_distribution_summaries",
    "compute_metric_summaries",
    "compute_metric_summary",
    "compute_publication_figsize",
    "plot_label_distribution_comparison_from_predictions",
    "plot_metric_comparison",
    "plot_metric_comparison_panels",
    "plot_metric_comparison_panels_from_predictions",
    "plot_metric_comparison_from_predictions",
    "plot_confusion_matrix",
    "plot_relative_confusion_matrix",
    "set_publication_rc_defaults",
    "video_to_image_grid",
    "visualize_video",
]
