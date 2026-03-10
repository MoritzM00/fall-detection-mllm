"""Public plotting API for confusion matrices and video visualizations."""

from .base import compute_publication_figsize, set_publication_rc_defaults
from .confusion import plot_confusion_matrix, plot_relative_confusion_matrix
from .video import video_to_image_grid, visualize_video

__all__ = [
    "compute_publication_figsize",
    "plot_confusion_matrix",
    "plot_relative_confusion_matrix",
    "set_publication_rc_defaults",
    "video_to_image_grid",
    "visualize_video",
]
