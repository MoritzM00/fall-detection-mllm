"""Load multiple runs, recompute metrics, and generate comparison plots.

Each run can be specified as either a W&B run ID or a path to a local JSONL
predictions file. W&B runs resolve to canonical local files under
``outputs/predictions/{wandb_project}/{run_id}.jsonl``; missing files are
downloaded once and backfilled there. The script saves one overall metric
comparison plot and one per-class F1 comparison plot.

Usage:
    python scripts/plot/compare_runs.py RUN [RUN ...] [OPTIONS]

Examples:
    python scripts/plot/compare_runs.py abc123 def456 ghi789
    python scripts/plot/compare_runs.py outputs/preds/a.jsonl def456
"""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from falldet.data.video_dataset import idx2label
from falldet.plot import (
    COLORS,
    MetricComparisonPanelSpec,
    PredictionMetricInput,
    plot_label_distribution_comparison_from_predictions,
    plot_metric_comparison_from_predictions,
    plot_metric_comparison_panels_from_predictions,
)
from falldet.plot.base import set_publication_rc_defaults
from falldet.schemas import InferenceConfig, ModelConfig
from falldet.utils.predictions import extract_labels_for_metrics, load_predictions_jsonl
from falldet.utils.wandb import load_run_from_wandb

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/plots")
CLASS_SUBSET: tuple[str, ...] | None = (
    "fall",
    "fallen",
    "lying",
    "lie_down",
    "other",
    "walk",
    "standing",
    "sit_down",
    "sitting",
    "crawl",
)


@dataclass(slots=True)
class LoadedRun:
    run_ref: str
    display_name: str
    dataset_name: str
    config: dict[str, Any]
    predictions: list[dict[str, Any]]


def _is_local_file(value: str) -> bool:
    return Path(value).is_file()


def _model_display_name_from_config(config: dict[str, Any]) -> str | None:
    if not config:
        return None

    try:
        inference_config = InferenceConfig.model_validate(config)
        return inference_config.model.name
    except Exception:
        pass

    model_config = config.get("model")
    if isinstance(model_config, dict):
        try:
            return ModelConfig.model_validate(model_config).name
        except Exception:
            return None

    return None


def _strip_name_after_model_size(model_name: str) -> str:
    match = re.search(r"^(.*?\b\d+(?:_\d+)?B(?:-A\d+B)?)", model_name)
    if match is not None:
        return match.group(1)
    return model_name


def _run_display_name(run_ref: str, config: dict[str, Any] | None = None) -> str:
    if config is not None:
        model_name = _model_display_name_from_config(config)
        if model_name:
            return _strip_name_after_model_size(model_name)

    if _is_local_file(run_ref):
        return Path(run_ref).stem
    return run_ref


def _default_output_name(run_names: list[str]) -> str:
    safe_names = [name.replace("/", "_").replace(" ", "_") for name in run_names]
    return f"metric_comparison_{'_vs_'.join(safe_names)}.pdf"


def _default_class_output_name(dataset_name: str, run_names: list[str]) -> str:
    safe_names = [_sanitize_filename_part(name) for name in run_names]
    return f"{dataset_name}_class_f1_comparison_{'_vs_'.join(safe_names)}.pdf"


def _default_distribution_output_name(dataset_name: str, run_names: list[str]) -> str:
    safe_names = [_sanitize_filename_part(name) for name in run_names]
    return f"{dataset_name}_label_distribution_{'_vs_'.join(safe_names)}.pdf"


def _default_precision_recall_output_name(dataset_name: str, run_names: list[str]) -> str:
    safe_names = [_sanitize_filename_part(name) for name in run_names]
    return f"{dataset_name}_precision_recall_{'_vs_'.join(safe_names)}.pdf"


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def _dataset_name_from_config(config: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    dataset_config = config.get("dataset") if isinstance(config, dict) else None
    if isinstance(dataset_config, dict):
        video_datasets = dataset_config.get("video_datasets")
        if isinstance(video_datasets, list) and video_datasets:
            dataset_names = []
            for item in video_datasets:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    dataset_names.append(item["name"])
            if dataset_names:
                return "_".join(dataset_names)

        dataset_name = dataset_config.get("name")
        if isinstance(dataset_name, str) and dataset_name:
            return dataset_name

    if predictions:
        prediction_dataset_names = sorted(
            {
                prediction["dataset"]
                for prediction in predictions
                if isinstance(prediction.get("dataset"), str) and prediction["dataset"]
            }
        )
        if prediction_dataset_names:
            return "_".join(prediction_dataset_names)

    return "unknown_dataset"


def _per_class_metric_order(class_subset: tuple[str, ...] | None = CLASS_SUBSET) -> tuple[str, ...]:
    labels = (
        class_subset
        if class_subset is not None
        else tuple(idx2label[idx] for idx in sorted(idx2label))
    )
    return tuple(f"{label}_f1" for label in labels)


def _per_class_metric_labels(class_subset: tuple[str, ...] | None = CLASS_SUBSET) -> dict[str, str]:
    labels = (
        class_subset
        if class_subset is not None
        else tuple(idx2label[idx] for idx in sorted(idx2label))
    )
    return {f"{label}_f1": label.replace("_", " ") for label in labels}


def _per_class_precision_metric_order(
    class_subset: tuple[str, ...] | None = CLASS_SUBSET,
) -> tuple[str, ...]:
    labels = (
        class_subset
        if class_subset is not None
        else tuple(idx2label[idx] for idx in sorted(idx2label))
    )
    return tuple(f"{label}_precision" for label in labels)


def _per_class_precision_metric_labels(
    class_subset: tuple[str, ...] | None = CLASS_SUBSET,
) -> dict[str, str]:
    labels = (
        class_subset
        if class_subset is not None
        else tuple(idx2label[idx] for idx in sorted(idx2label))
    )
    return {f"{label}_precision": label.replace("_", " ") for label in labels}


def _per_class_recall_metric_order(
    class_subset: tuple[str, ...] | None = CLASS_SUBSET,
) -> tuple[str, ...]:
    labels = (
        class_subset
        if class_subset is not None
        else tuple(idx2label[idx] for idx in sorted(idx2label))
    )
    return tuple(f"{label}_sensitivity" for label in labels)


def _per_class_recall_metric_labels(
    class_subset: tuple[str, ...] | None = CLASS_SUBSET,
) -> dict[str, str]:
    labels = (
        class_subset
        if class_subset is not None
        else tuple(idx2label[idx] for idx in sorted(idx2label))
    )
    return {f"{label}_sensitivity": label.replace("_", " ") for label in labels}


def load_run_predictions(
    run_ref: str,
    *,
    entity: str | None = None,
    project: str | None = None,
    output_root: Path = DEFAULT_CACHE_DIR,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load config and predictions from a local JSONL file or a W&B run ID."""
    if _is_local_file(run_ref):
        logger.info(f"Loading local predictions from {run_ref}")
        metadata, predictions = load_predictions_jsonl(run_ref)
        return metadata.get("config", {}), predictions

    return load_run_from_wandb(run_ref, project=project, entity=entity, output_root=output_root)


def load_comparison_runs(
    run_refs: list[str],
    *,
    entity: str | None = None,
    project: str | None = None,
    output_root: Path = DEFAULT_CACHE_DIR,
    display_names: list[str] | None = None,
) -> list[LoadedRun]:
    """Load all runs needed for a comparison workflow."""
    loaded_runs: list[LoadedRun] = []

    if display_names is not None and len(display_names) != len(run_refs):
        raise ValueError(
            f"display_names must have exactly one entry per run ({len(run_refs)} runs, "
            f"{len(display_names)} names provided)."
        )

    for idx, run_ref in enumerate(run_refs):
        config, predictions = load_run_predictions(
            run_ref,
            entity=entity,
            project=project,
            output_root=output_root,
        )
        display_name = (
            display_names[idx] if display_names else _run_display_name(run_ref, config=config)
        )
        loaded_runs.append(
            LoadedRun(
                run_ref=run_ref,
                display_name=display_name,
                dataset_name=_dataset_name_from_config(config, predictions),
                config=config,
                predictions=predictions,
            )
        )
        logger.info(f"Prepared {display_name} with {len(predictions)} predictions")

    return loaded_runs


def build_prediction_metric_inputs(runs: list[LoadedRun]) -> list[PredictionMetricInput]:
    """Convert loaded prediction dicts into generic plotting inputs."""
    metric_inputs: list[PredictionMetricInput] = []
    for run in runs:
        y_true, y_pred = extract_labels_for_metrics(run.predictions)
        metric_inputs.append(
            PredictionMetricInput(name=run.display_name, y_true=y_true, y_pred=y_pred)
        )
    return metric_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load multiple runs, recompute metrics, and plot them side by side.",
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="W&B run IDs or local JSONL prediction files to compare",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional display names for the runs, in the same order as the inputs",
    )
    parser.add_argument("--entity", default=None, help="W&B entity (defaults to WANDB_ENTITY)")
    parser.add_argument("--project", default=None, help="W&B project (defaults to WANDB_PROJECT)")
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title (default: none)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: outputs/plots/metric_comparison_<runs>.pdf)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Output root containing predictions/<project>/<run_id>.jsonl (default: outputs/)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if len(args.runs) < 2:
        raise SystemExit("Please provide at least two runs to compare.")

    if args.names is not None and len(args.names) != len(args.runs):
        raise SystemExit("--names must contain exactly one display name per run.")

    set_publication_rc_defaults(
        use_tex=True,
        rc={
            "savefig.pad_inches": 0.1,
        },
    )

    output_root = Path(args.cache_dir)
    loaded_runs = load_comparison_runs(
        args.runs,
        entity=args.entity,
        project=args.project,
        output_root=output_root,
        display_names=args.names,
    )
    metric_inputs = build_prediction_metric_inputs(loaded_runs)

    dataset_names = {_sanitize_filename_part(run.dataset_name) for run in loaded_runs}
    dataset_name = dataset_names.pop() if len(dataset_names) == 1 else "mixed_datasets"
    fig, _ = plot_metric_comparison_from_predictions(
        metric_inputs,
        title=args.title,
        figsize=(10.0, 4.0),
    )

    run_names = [run.display_name for run in loaded_runs]
    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"{dataset_name}_{_default_output_name(run_names)}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {output_path}")

    class_fig, _ = plot_metric_comparison_from_predictions(
        metric_inputs,
        metric_order=_per_class_metric_order(),
        metric_labels=_per_class_metric_labels(),
        title=None,
        figsize=(10.0, 4.8),
        missing_metric_value=0.0,
    )
    class_output_path = output_path.with_name(_default_class_output_name(dataset_name, run_names))
    class_fig.savefig(class_output_path, bbox_inches="tight")
    logger.info(f"Saved class F1 comparison plot to {class_output_path}")

    distribution_labels = (
        CLASS_SUBSET
        if CLASS_SUBSET is not None
        else tuple(idx2label[idx] for idx in sorted(idx2label))
    )
    distribution_fig, _ = plot_label_distribution_comparison_from_predictions(
        metric_inputs,
        label_order=distribution_labels,
        label_labels={label: label.replace("_", " ") for label in distribution_labels},
        title=None,
        figsize=(10.0, 4.8),
        actual_name="Actual",
        actual_color=COLORS["neutral"],
    )
    distribution_output_path = output_path.with_name(
        _default_distribution_output_name(dataset_name, run_names)
    )
    distribution_fig.savefig(distribution_output_path, bbox_inches="tight")
    logger.info(f"Saved label distribution plot to {distribution_output_path}")

    precision_recall_fig, _ = plot_metric_comparison_panels_from_predictions(
        metric_inputs,
        panel_specs=[
            MetricComparisonPanelSpec(
                metric_order=_per_class_precision_metric_order(),
                metric_labels=_per_class_precision_metric_labels(),
                title=None,
                xlabel="(a) Precision",
                ylabel="Score (\\%)",
                xtick_rotation=0.0,
                xtick_horizontalalignment="center",
            ),
            MetricComparisonPanelSpec(
                metric_order=_per_class_recall_metric_order(),
                metric_labels=_per_class_recall_metric_labels(),
                title=None,
                xlabel="(b) Recall",
                ylabel="",
                xtick_rotation=0.0,
                xtick_horizontalalignment="center",
            ),
        ],
        figsize=(12.0, 4.8),
        sharey=True,
        legend_panel_index=1,
        wspace=0.10,
        missing_metric_value=0.0,
    )
    precision_recall_output_path = output_path.with_name(
        _default_precision_recall_output_name(dataset_name, run_names)
    )
    precision_recall_fig.savefig(precision_recall_output_path, bbox_inches="tight")
    logger.info(f"Saved precision/recall comparison plot to {precision_recall_output_path}")


if __name__ == "__main__":
    main()
