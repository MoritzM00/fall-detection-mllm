"""Compare two runs by generating a relative confusion matrix.

Each run can be specified as either a W&B run ID or a path to a local JSONL
predictions file. W&B runs resolve to canonical local files under
``outputs/predictions/{wandb_project}/{run_id}.jsonl``; missing files are
downloaded once and backfilled there.

Usage:
    python scripts/plot/relative_confusion.py RUN_A RUN_B [OPTIONS]

Examples:
    # Two W&B run IDs
    python scripts/plot/relative_confusion.py abc123 def456

    # One local file, one W&B run
    python scripts/plot/relative_confusion.py outputs/preds/baseline.jsonl def456

    # Both local files
    python scripts/plot/relative_confusion.py run_a.jsonl run_b.jsonl

    # With subset, clustered order (default)
    python scripts/plot/relative_confusion.py abc123 def456 \\
        --subset fall fallen lie_down lying

    # Canonical label order
    python scripts/plot/relative_confusion.py abc123 def456 --order canonical
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from falldet.data.video_dataset import label2idx
from falldet.plot import (
    cluster_confusion_labels,
    plot_relative_confusion_matrix,
    set_publication_rc_defaults,
)
from falldet.utils.predictions import extract_labels_for_metrics, load_predictions_jsonl
from falldet.utils.wandb import load_run_from_wandb

CANONICAL_ORDER = list(label2idx.keys())

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/plots")


def _is_local_file(value: str) -> bool:
    """Return True if *value* looks like a path to an existing file."""
    return Path(value).is_file()


def load_predictions(
    run_ref: str,
    *,
    entity: str | None = None,
    project: str | None = None,
    output_root: Path = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    """Load predictions from a local JSONL file or a W&B run ID.

    Args:
        run_ref: Either a path to a local ``.jsonl`` file or a W&B run ID.
        entity: W&B entity (only used when *run_ref* is a run ID).
        project: W&B project (only used when *run_ref* is a run ID).
        output_root: Root directory containing the ``predictions/`` subtree.

    Returns:
        List of prediction dicts.
    """
    if _is_local_file(run_ref):
        logger.info("Loading local predictions from %s", run_ref)
        _, predictions = load_predictions_jsonl(run_ref)
    else:
        _, predictions = load_run_from_wandb(
            run_ref, project=project, entity=entity, output_root=output_root
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two runs via a relative confusion matrix.",
    )
    parser.add_argument(
        "run_a",
        help="Run A (baseline): W&B run ID or path to a local JSONL predictions file",
    )
    parser.add_argument(
        "run_b",
        help="Run B (comparison): W&B run ID or path to a local JSONL predictions file",
    )
    parser.add_argument("--entity", default=None, help="W&B entity (defaults to WANDB_ENTITY)")
    parser.add_argument("--project", default=None, help="W&B project (defaults to WANDB_PROJECT)")
    parser.add_argument(
        "--subset",
        nargs="+",
        default=None,
        help="Subset of class labels to display (e.g. --subset fall fallen lie_down)",
    )
    parser.add_argument(
        "--order",
        choices=["clustered", "canonical", "alphabetical", "fixed"],
        default="clustered",
        help="Label ordering strategy (default: clustered). "
        "'fixed' preserves the --subset order exactly.",
    )
    parser.add_argument(
        "--linkage-method",
        default="average",
        help="Linkage method for clustering (default: average)",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Distance metric for clustering (default: cosine)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: outputs/plots/relative_cm_<run_a>_vs_<run_b>.pdf)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title (default: None)",
    )
    parser.add_argument(
        "--cbar",
        action="store_true",
        default=False,
        help="Show color bar (default: off)",
    )
    parser.add_argument(
        "--no-ylabel",
        action="store_true",
        default=False,
        help="Hide y-axis labels (use for the right plot in a side-by-side figure)",
    )
    parser.add_argument(
        "--width-fraction",
        type=float,
        default=0.5,
        help="Figure width as fraction of text width (default: 0.5)",
    )
    parser.add_argument(
        "--height-ratio",
        type=float,
        default=0.9,
        help="Figure height ratio (default: 0.9)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Output root containing predictions/<project>/<run_id>.jsonl (default: outputs/)",
    )
    return parser.parse_args()


def _run_display_name(run_ref: str) -> str:
    """Derive a short display name from a run reference."""
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
) -> list[str]:
    """Return an ordered list of labels for the requested mode."""
    all_labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    if order == "clustered":
        ordered, _ = cluster_confusion_labels(
            y_true, y_pred, method=method, metric=metric, labels=all_labels
        )
        return ordered
    elif order == "canonical":
        present = set(all_labels)
        ordered = [label for label in CANONICAL_ORDER if label in present]
        ordered += sorted(present - set(ordered))
        return ordered
    elif order == "fixed":
        # Preserve the exact order given by --subset (or the natural label order).
        return all_labels
    else:  # alphabetical
        return sorted(all_labels)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    _, (figure_width, figure_height) = set_publication_rc_defaults(
        use_tex=True,
        height_ratio=args.height_ratio,
        width_fraction=args.width_fraction,
        rc={"font.size": 9},
    )

    output_root = Path(args.cache_dir)

    # Load predictions
    preds_a = load_predictions(
        args.run_a, entity=args.entity, project=args.project, output_root=output_root
    )
    preds_b = load_predictions(
        args.run_b, entity=args.entity, project=args.project, output_root=output_root
    )

    # Extract label lists
    y_true_a, y_pred_a = extract_labels_for_metrics(preds_a)
    y_true_b, y_pred_b = extract_labels_for_metrics(preds_b)

    # Support counts from run A (baseline ground truth).
    from collections import Counter

    true_counts = Counter(y_true_a)

    # For clustering: use combined labels from both runs, filtered to subset if given.
    subset_labels: list[str] | None = args.subset
    all_true = y_true_a + y_true_b
    all_pred = y_pred_a + y_pred_b
    if subset_labels is not None:
        subset_set = set(subset_labels)
        pairs = [(t, p) for t, p in zip(all_true, all_pred) if t in subset_set]
        y_clust, p_clust = (list(x) for x in zip(*pairs)) if pairs else ([], [])
    else:
        y_clust, p_clust = all_true, all_pred

    label_order = _resolve_label_order(
        args.order,
        y_clust,
        p_clust,
        args.linkage_method,
        args.metric,
        labels=subset_labels,
    )

    support = {label: true_counts.get(label, 0) for label in label_order}

    # Build plot kwargs
    plot_kwargs: dict = {}
    if args.title:
        plot_kwargs["title"] = args.title
    if args.cbar:
        plot_kwargs["cbar"] = args.cbar

    # Plot
    fig, _ = plot_relative_confusion_matrix(
        y_true_a=y_true_a,
        y_pred_a=y_pred_a,
        y_true_b=y_true_b,
        y_pred_b=y_pred_b,
        label_order=label_order,
        figsize=(figure_width, figure_height),
        support=support,
        show_ylabel=not args.no_ylabel,
        **plot_kwargs,
    )

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        name_a = _run_display_name(args.run_a)
        name_b = _run_display_name(args.run_b)
        output_path = DEFAULT_OUTPUT_DIR / f"relative_cm_{name_a}_vs_{name_b}_{args.order}.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Saved relative confusion matrix to {output_path}")


if __name__ == "__main__":
    main()
