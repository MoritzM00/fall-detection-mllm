"""Plot a confusion matrix for a single run, with optional label clustering.

Each run can be specified as either a W&B run ID or a path to a local JSONL
predictions file.

Usage:
    python scripts/plot/confusion_matrix.py RUN [OPTIONS]

Examples:
    # Clustered order (default) from a local file
    python scripts/plot/confusion_matrix.py outputs/predictions/proj/run.jsonl

    # Clustered order with dendrogram sidebar
    python scripts/plot/confusion_matrix.py outputs/predictions/proj/run.jsonl --dendrogram

    # Canonical label order (as defined in label2idx)
    python scripts/plot/confusion_matrix.py run_id --order canonical

    # Row-normalised, specific subset
    python scripts/plot/confusion_matrix.py run_id --normalize true --subset fall fallen

    # Save to a custom path
    python scripts/plot/confusion_matrix.py run_id -o /tmp/cm.pdf
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy

from falldet.data.video_dataset import label2idx
from falldet.plot import (
    cluster_confusion_labels,
    plot_confusion_matrix,
    set_publication_rc_defaults,
)
from falldet.utils.predictions import extract_labels_for_metrics, load_predictions_jsonl
from falldet.utils.wandb import load_run_from_wandb

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/plots")

CANONICAL_ORDER = list(label2idx.keys())

# Set to a list of label strings to restrict the plot to a subset, e.g.:
#   SUBSET = ["fall", "fallen", "walk", "standing"]
# Set to None to show all labels present in the predictions.
SUBSET: list[str] | None = ["jump", "fall", "lie_down", "other", "lying", "fallen"]


def _is_local_file(value: str) -> bool:
    return Path(value).is_file()


def load_predictions(
    run_ref: str,
    *,
    entity: str | None = None,
    project: str | None = None,
    output_root: Path = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    if _is_local_file(run_ref):
        logger.info("Loading local predictions from %s", run_ref)
        _, predictions = load_predictions_jsonl(run_ref)
    else:
        _, predictions = load_run_from_wandb(
            run_ref, project=project, entity=entity, output_root=output_root
        )
    return predictions


def _run_display_name(run_ref: str) -> str:
    if _is_local_file(run_ref):
        return Path(run_ref).stem
    return run_ref


def _resolve_label_order(
    order: str,
    y_true: list[str],
    y_pred: list[str],
    method: str,
    metric: str,
    labels: list[str] | None = None,
) -> tuple[list[str], np.ndarray | None]:
    """Return (ordered_labels, linkage_matrix_or_None) for the requested mode."""
    all_labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    if order == "clustered":
        ordered, linkage_matrix = cluster_confusion_labels(
            y_true, y_pred, method=method, metric=metric, labels=all_labels
        )
        return ordered, linkage_matrix
    elif order == "canonical":
        present = set(all_labels)
        ordered = [label for label in CANONICAL_ORDER if label in present]
        ordered += sorted(present - set(ordered))
        return ordered, None
    else:  # alphabetical
        return all_labels, None


def _plot_with_dendrogram(
    y_true: list[str],
    y_pred: list[str],
    label_order: list[str],
    linkage_matrix: np.ndarray,
    normalize: str | None,
    figsize: tuple[float, float],
    annot_threshold: float | None = None,
    normalization_labels: list[str] | None = None,
) -> plt.Figure:
    """Create a figure with a dendrogram on the left and confusion matrix on the right.

    ``label_order`` must already be in leaf order (as returned by
    ``cluster_confusion_labels``). The dendrogram is drawn with its root on the
    right and leaves on the left, flipped vertically so leaf 0 is at the top -
    matching the seaborn heatmap convention where row 0 is at the top.
    """
    fig, ax_cm = plt.subplots(figsize=figsize)

    # label_order is already in leaf order from cluster_confusion_labels.
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=normalize,
        label_order=label_order,
        normalization_labels=normalization_labels,
        annot_threshold=annot_threshold,
        ax=ax_cm,
    )

    # Compute the right edge and height of the actual heatmap content in figure
    # coords, then add the dendrogram axes flush against it.
    fig.canvas.draw()
    n = len(label_order)
    to_fig = fig.transFigure.inverted()
    x_right, y_bottom = to_fig.transform(ax_cm.transData.transform((n, n)))
    x_left, y_top = to_fig.transform(ax_cm.transData.transform((0, 0)))
    dendro_width = (x_right - x_left) * 0.22
    ax_dendro = fig.add_axes([x_right, y_bottom, dendro_width, y_top - y_bottom])

    hierarchy.dendrogram(
        linkage_matrix,
        orientation="right",
        no_labels=True,
        ax=ax_dendro,
        color_threshold=0,
        above_threshold_color="steelblue",
        link_color_func=lambda _: "steelblue",
    )
    ax_dendro.invert_yaxis()
    ax_dendro.set_axis_off()

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a confusion matrix with optional label clustering.",
    )
    parser.add_argument(
        "run",
        help="W&B run ID or path to a local JSONL predictions file",
    )
    parser.add_argument(
        "--order",
        choices=["clustered", "canonical", "alphabetical"],
        default="clustered",
        help="Label ordering strategy (default: clustered)",
    )
    parser.add_argument(
        "--linkage-method",
        default="average",
        help="Linkage method for clustering (default: average). "
        "Use 'ward' with --metric euclidean.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Distance metric for clustering (default: cosine)",
    )
    parser.add_argument(
        "--normalize",
        choices=["true", "pred", "all"],
        default=None,
        help="Confusion matrix normalisation (default: raw counts)",
    )
    parser.add_argument(
        "--dendrogram",
        action="store_true",
        default=False,
        help="Show a dendrogram sidebar (only with --order clustered)",
    )
    parser.add_argument(
        "--annot-threshold",
        type=float,
        default=0.03,
        help="Hide annotations below this value (default: 0.03)",
    )
    parser.add_argument(
        "--cbar",
        action="store_true",
        default=False,
        help="Show color bar",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Root dir containing predictions/<project>/<run_id>.jsonl (default: outputs/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: outputs/plots/cm_<run>.pdf)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _, (figure_width, figure_height) = set_publication_rc_defaults(
        use_tex=True, height_ratio=0.9, width_fraction=0.5, rc={"font.size": 9}
    )
    figsize = (figure_width, figure_height)

    args = parse_args()
    output_root = Path(args.cache_dir)

    predictions = load_predictions(
        args.run, entity=args.entity, project=args.project, output_root=output_root
    )
    y_true, y_pred = extract_labels_for_metrics(predictions)

    # For clustering/ordering: use only rows whose true label is in SUBSET so that
    # out-of-subset classes don't influence the cluster distances.
    subset_set = set(SUBSET) if SUBSET is not None else None
    if subset_set is not None:
        pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in subset_set]
        y_clust, p_clust = (list(x) for x in zip(*pairs)) if pairs else ([], [])
    else:
        y_clust, p_clust = y_true, y_pred

    subset_labels = list(SUBSET) if SUBSET is not None else None
    # When a subset is displayed, normalization must use all labels so that row
    # sums reflect the full prediction distribution, not just the subset columns.
    normalization_labels = sorted(set(y_true) | set(y_pred)) if subset_labels is not None else None

    # Compute per-class support (number of true samples) for y-axis labels.
    from collections import Counter

    true_counts = Counter(y_true)
    support = {
        label: true_counts.get(label, 0)
        for label in (subset_labels or sorted(set(y_true) | set(y_pred)))
    }

    if args.dendrogram and args.order == "clustered":
        label_order, linkage_matrix = _resolve_label_order(
            args.order,
            y_clust,
            p_clust,
            args.linkage_method,
            args.metric,
            labels=subset_labels,
        )
        assert linkage_matrix is not None
        fig = _plot_with_dendrogram(
            y_true=y_true,
            y_pred=y_pred,
            label_order=label_order,
            linkage_matrix=linkage_matrix,
            normalize=args.normalize,
            figsize=(figure_width * 1.4, figure_height),
            annot_threshold=args.annot_threshold,
            normalization_labels=normalization_labels,
        )
    else:
        label_order, _ = _resolve_label_order(
            args.order,
            y_clust,
            p_clust,
            args.linkage_method,
            args.metric,
            labels=subset_labels,
        )
        fig, _ = plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            normalize=args.normalize,
            label_order=label_order,
            normalization_labels=normalization_labels,
            annot_threshold=args.annot_threshold,
            cbar=args.cbar,
            figsize=figsize,
            support=support,
        )

    if args.output:
        output_path = Path(args.output)
    else:
        run_name = _run_display_name(args.run)
        suffix = f"_{args.order}"
        if args.dendrogram:
            suffix += "_dendro"
        output_path = DEFAULT_OUTPUT_DIR / f"cm_{run_name}{suffix}.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    logger.info("Saved confusion matrix to %s", output_path)


if __name__ == "__main__":
    main()
