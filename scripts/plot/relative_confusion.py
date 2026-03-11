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

    # With subset
    python scripts/plot/relative_confusion.py abc123 def456 \\
        --subset fall fallen lie_down lying
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from falldet.plot import plot_relative_confusion_matrix, set_publication_rc_defaults
from falldet.utils.predictions import extract_labels_for_metrics, load_predictions_jsonl
from falldet.utils.wandb import load_run_from_wandb

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _, (figure_width, figure_height) = set_publication_rc_defaults(
        use_tex=True, height_ratio=1, width_fraction=0.5, rc={"font.size": 8}
    )
    args = parse_args()

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

    # Build plot kwargs
    plot_kwargs: dict = {}
    if args.subset:
        plot_kwargs["subset"] = args.subset
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
        figsize=(figure_width, figure_height),
        **plot_kwargs,
    )

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        name_a = _run_display_name(args.run_a)
        name_b = _run_display_name(args.run_b)
        output_path = DEFAULT_OUTPUT_DIR / f"relative_cm_{name_a}_vs_{name_b}.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Saved relative confusion matrix to {output_path}")


if __name__ == "__main__":
    main()
