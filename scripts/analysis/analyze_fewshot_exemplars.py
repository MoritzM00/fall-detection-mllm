#!/usr/bin/env python
"""Analyze exemplar selection in similarity-based few-shot experiments.

For a given W&B run (similarity shot_selection), this script:
  - Reconstructs the full cosine similarity matrix from stored embeddings
  - For each value of k, recomputes which exemplars would be selected
  - Computes per-sample and aggregate metrics relating exemplar labels to predictions

Metrics computed per k:
  - GT coverage rate: fraction of samples where the ground truth label appears in the exemplars
  - Prediction-in-exemplar rate: fraction where the predicted label appears in the exemplars
  - Novel prediction rate: fraction where the predicted label does NOT appear in the exemplars
  - Mean unique classes among the k exemplars
  - Class coverage: how many of 16 classes appear in at least one exemplar set
  - Majority vote: fraction where prediction matches majority-vote exemplar label,
    majority vote accuracy (majority vote == GT), fraction where GT matches majority vote
  - Error bias: among wrong predictions, fraction biased toward an exemplar label vs novel
  - Per-class exemplar selection frequency vs training frequency (minority class analysis)
  - Per-class F1 conditioned on whether GT was covered by exemplars

Usage:
    python scripts/analysis/analyze_fewshot_exemplars.py \\
        --run-id <ID> --project <PROJECT> \\
        [--entity <ENTITY>] [--output-root outputs] \\
        [--k-values 1 2 3 5 8 16] [--save-json]
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from falldet.data.video_dataset import label2idx
from falldet.data.video_dataset_factory import get_video_datasets
from falldet.embeddings import compute_similarity_scores, get_embedding_filename, load_embeddings
from falldet.metrics.base import compute_metrics
from falldet.schemas import InferenceConfig
from falldet.utils.wandb import load_run_from_wandb

ALL_LABELS = list(label2idx.keys())
NUM_CLASSES = len(ALL_LABELS)


# ---------------------------------------------------------------------------
# Default runs (Qwen3-VL-8B, OOPS cross-subject, similarity selection)
# ---------------------------------------------------------------------------

DEFAULT_PROJECT = "fall-detection-fewshot"
DEFAULT_RUN_IDS = [
    "lz8e33if",  # k=1
    "xisxlsxd",  # k=3
    "iy7xabko",  # k=5
    "zy6qp8ph",  # k=7
    "uei9m4ra",  # k=10
    "iwoxtm2s",  # k=15
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exemplar selection in similarity-based few-shot experiments"
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=DEFAULT_RUN_IDS,
        help="One or more W&B run IDs. k values are inferred from each run's num_shots. "
        f"Defaults to the Qwen3-VL-8B OOPS runs: {DEFAULT_RUN_IDS}",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project name")
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (defaults to WANDB_ENTITY env var)",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output directory containing predictions/ and embeddings/ subdirs",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results to JSON alongside the predictions file",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a paper-ready LaTeX table to stdout",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="PATH",
        help="Save exemplar-frequency line plot to this path (e.g. outputs/plots/exemplar_freq.pdf)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions_and_config(
    run_id: str,
    project: str,
    entity: str | None,
    output_root: str,
) -> tuple[InferenceConfig, list[dict[str, Any]]]:
    config_dict, predictions = load_run_from_wandb(
        run_id, project=project, entity=entity, output_root=output_root
    )
    config = InferenceConfig.model_validate(config_dict)
    print(f"Loaded {len(predictions)} predictions")
    print(f"  shot_selection: {config.prompt.shot_selection}")
    print(f"  original num_shots: {config.prompt.num_shots}")
    print(f"  dataset: {config.dataset.name}")
    return config, predictions


def load_embeddings_and_corpus(
    config: InferenceConfig,
    output_root: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load query + corpus embeddings and extract corpus labels from the train dataset.

    Replicates the embedding-loading logic from setup_fewshot_sampler.
    The dataset key used for embedding filenames (e.g. "OOPS_cs") is resolved
    by loading the train dataset via get_video_datasets, mirroring vllm_inference.py.
    """
    emb_model_name = config.embedding_model_name or config.model.name
    emb_dir = Path(output_root) / "embeddings"
    if not emb_dir.exists():
        emb_dir = Path(config.embeddings_dir)

    # Load train dataset first to resolve the actual dataset key (e.g. "OOPS_cs")
    train_datasets = get_video_datasets(
        config=config,
        mode="train",
        split=config.data.split,
        size=config.data.size,
        seed=config.data.seed,
        return_individual=True,
    )
    individual: dict = train_datasets["individual"]  # type: ignore[index]
    dataset_name, train_ds = next(iter(individual.items()))
    print(f"Resolved dataset name: {dataset_name!r}")

    train_emb_file = get_embedding_filename(
        dataset_name,
        "train",
        config.num_frames,
        config.model_fps,
        model_name=emb_model_name,
        data_size=config.data.size,
    )
    query_emb_file = get_embedding_filename(
        dataset_name,
        config.data.mode,
        config.num_frames,
        config.model_fps,
        model_name=emb_model_name,
        data_size=config.data.size,
    )
    print(f"Loading corpus embeddings: {emb_dir / train_emb_file}")
    corpus_emb, _ = load_embeddings(emb_dir / train_emb_file)
    print(f"Loading query embeddings:  {emb_dir / query_emb_file}")
    query_emb, _ = load_embeddings(emb_dir / query_emb_file)

    if config.num_samples is not None:
        n = min(config.num_samples, query_emb.shape[0])
        query_emb = query_emb[:n]
        print(f"Sliced query embeddings to {n} (num_samples={config.num_samples})")

    corpus_labels: list[str] = [
        train_ds.video_segments[i]["label_str"] for i in range(len(train_ds))
    ]
    print(f"Corpus size: {len(corpus_labels)} | Query size: {query_emb.shape[0]}")
    return query_emb, corpus_emb, corpus_labels


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _majority_vote(labels: list[str]) -> str:
    """Return the most frequent label; tie-break by label2idx order."""
    counts = Counter(labels)
    return max(counts, key=lambda lbl: (counts[lbl], -label2idx.get(lbl, 999)))


def analyze_for_k(
    k: int,
    similarity: torch.Tensor,
    corpus_labels: list[str],
    predictions: list[dict[str, Any]],
    train_class_dist: dict[str, float],
) -> dict[str, Any]:
    """Compute all metrics for a single value of k."""
    similarity.shape[0]
    k_capped = min(k, similarity.shape[1])
    if k_capped < k:
        print(f"  [k={k}] capped at corpus size {k_capped}")

    _, topk_idx = torch.topk(similarity, k=k_capped, dim=1)  # (N, k)
    topk_idx_list = topk_idx.tolist()

    # Per-sample analysis
    gt_labels: list[str] = []
    pred_labels: list[str] = []
    gt_covered: list[bool] = []
    pred_in_exemplars: list[bool] = []
    majority_votes: list[str] = []
    exemplar_label_sets: list[set[str]] = []
    unique_class_counts: list[int] = []

    for i, pred in enumerate(predictions):
        gt = pred["label_str"]
        predicted = pred.get("predicted_label") or ""
        exemplar_idxs = topk_idx_list[i]
        exemplar_lbls = [corpus_labels[j] for j in exemplar_idxs]
        exemplar_set = set(exemplar_lbls)

        gt_labels.append(gt)
        pred_labels.append(predicted)
        gt_covered.append(gt in exemplar_set)
        pred_in_exemplars.append(predicted in exemplar_set)
        majority_votes.append(_majority_vote(exemplar_lbls))
        exemplar_label_sets.append(exemplar_set)
        unique_class_counts.append(len(exemplar_set))

    n = len(predictions)

    # --- Aggregate coverage rates ---
    gt_coverage_rate = sum(gt_covered) / n
    pred_in_exemplar_rate = sum(pred_in_exemplars) / n
    novel_pred_rate = 1.0 - pred_in_exemplar_rate
    mean_unique_classes = sum(unique_class_counts) / n

    # Class coverage: how many of 16 classes appear in at least one exemplar set
    all_exemplar_labels = set().union(*exemplar_label_sets)
    class_coverage = len(all_exemplar_labels)

    # --- Majority vote metrics ---
    mv_matches_pred = sum(mv == p for mv, p in zip(majority_votes, pred_labels)) / n
    mv_matches_gt = sum(mv == gt for mv, gt in zip(majority_votes, gt_labels)) / n

    # --- Per-class exemplar selection frequency ---
    label_slot_counts: Counter[str] = Counter()
    for i in range(n):
        exemplar_lbls = [corpus_labels[j] for j in topk_idx_list[i]]
        label_slot_counts.update(exemplar_lbls)
    total_slots = k_capped * n
    exemplar_freq: dict[str, float] = {
        lbl: label_slot_counts.get(lbl, 0) / total_slots for lbl in ALL_LABELS
    }

    # --- Prediction-dependent metrics ---
    # These use the fixed predictions from the run. For k != actual num_shots they answer:
    # "given these predictions, how do they relate to the exemplar set that *would* have
    # been selected at k=X?" — valid as a retrospective analysis across all k.
    pred_in_exemplar_rate = sum(pred_in_exemplars) / n
    novel_pred_rate = 1.0 - pred_in_exemplar_rate

    n_errors = 0
    errors_in_exemplars = 0
    for i in range(n):
        if pred_labels[i] != gt_labels[i]:
            n_errors += 1
            if pred_in_exemplars[i]:
                errors_in_exemplars += 1
    error_bias_rate = errors_in_exemplars / n_errors if n_errors > 0 else float("nan")

    covered_indices = [i for i, c in enumerate(gt_covered) if c]
    not_covered_indices = [i for i, c in enumerate(gt_covered) if not c]

    def _subset_metrics(indices: list[int]) -> dict[str, float] | None:
        if len(indices) < 2:
            return None
        yt = [gt_labels[i] for i in indices]
        yp = [pred_labels[i] for i in indices]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            return compute_metrics(yp, yt)

    metrics_covered = _subset_metrics(covered_indices)
    metrics_not_covered = _subset_metrics(not_covered_indices)

    return {
        "k": k,
        "k_capped": k_capped,
        "n": n,
        "gt_coverage_rate": gt_coverage_rate,
        "pred_in_exemplar_rate": pred_in_exemplar_rate,
        "novel_pred_rate": novel_pred_rate,
        "mean_unique_classes": mean_unique_classes,
        "class_coverage": class_coverage,
        "classes_in_any_exemplar": sorted(all_exemplar_labels),
        "mv_matches_pred_rate": mv_matches_pred,
        "mv_matches_gt_rate": mv_matches_gt,
        "n_errors": n_errors,
        "error_bias_rate": error_bias_rate,
        "exemplar_freq": exemplar_freq,
        "train_freq": train_class_dist,
        "n_covered": len(covered_indices),
        "n_not_covered": len(not_covered_indices),
        "metrics_covered": metrics_covered,
        "metrics_not_covered": metrics_not_covered,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(val: float | None, pct: bool = True) -> str:
    if val is None or (isinstance(val, float) and val != val):  # nan
        return "   n/a"
    if pct:
        return f"{val * 100:5.1f}%"
    return f"{val:6.3f}"


def print_results(results_by_k: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("OVERVIEW (per k)")
    print("=" * 80)
    header = f"{'k':>4}  {'GT cov':>7}  {'Pred∈ex':>7}  {'Novel':>7}  {'Uniq cls':>8}  {'Cls cov':>7}  {'MV==Pred':>8}  {'MV acc':>7}  {'Err bias':>8}"
    print(header)
    print("-" * len(header))
    for r in results_by_k:
        print(
            f"{r['k']:>4}  "
            f"{_fmt(r['gt_coverage_rate']):>7}  "
            f"{_fmt(r['pred_in_exemplar_rate']):>7}  "
            f"{_fmt(r['novel_pred_rate']):>7}  "
            f"{r['mean_unique_classes']:>8.2f}  "
            f"{r['class_coverage']:>4}/{NUM_CLASSES}    "
            f"{_fmt(r['mv_matches_pred_rate']):>8}  "
            f"{_fmt(r['mv_matches_gt_rate']):>7}  "
            f"{_fmt(r['error_bias_rate']):>8}"
        )

    print("\n" + "=" * 80)
    print("MINORITY CLASS: exemplar selection frequency vs training frequency")
    print("=" * 80)
    # Sort classes by training frequency (ascending) from first k result
    train_freq = results_by_k[0]["train_freq"]
    sorted_labels = sorted(ALL_LABELS, key=lambda lbl: train_freq.get(lbl, 0))
    k_values = [r["k"] for r in results_by_k]
    header2 = f"{'Class':<14}  {'Train%':>6}  " + "  ".join(f"k={k:>2}%" for k in k_values)
    print(header2)
    print("-" * len(header2))
    for lbl in sorted_labels:
        train_pct = train_freq.get(lbl, 0) * 100
        row = f"{lbl:<14}  {train_pct:>5.1f}%  "
        row += "  ".join(f"{r['exemplar_freq'].get(lbl, 0) * 100:>7.1f}%" for r in results_by_k)
        print(row)

    print("\n" + "=" * 80)
    print("PER-CLASS F1: conditioned on exemplar coverage (GT label ∈ exemplars?)")
    print("  'covered' = GT label appeared in the k exemplars for that sample")
    print("=" * 80)
    # Collect all classes that appear across any run
    gt_classes: set[str] = set()
    for r in results_by_k:
        for metrics in (r["metrics_covered"], r["metrics_not_covered"]):
            if metrics:
                gt_classes.update(key.replace("_f1", "") for key in metrics if key.endswith("_f1"))
    sorted_gt_classes = sorted(gt_classes, key=lambda c: label2idx.get(c, 999))

    # Header
    k_header = "  ".join(f"{'k=' + str(r['k']):>14}" for r in results_by_k)
    print(f"  {'Class':<14}  {k_header}")
    print(f"  {'':14}  " + "  ".join(f"{'cov':>6} {'!cov':>6}" for _ in results_by_k))
    print("  " + "-" * (16 + 15 * len(results_by_k)))
    for lbl in sorted_gt_classes:
        key = f"{lbl}_f1"
        cells = []
        for r in results_by_k:
            cov = r["metrics_covered"].get(key) if r["metrics_covered"] else None
            ncov = r["metrics_not_covered"].get(key) if r["metrics_not_covered"] else None
            cells.append(f"{_fmt(cov):>6} {_fmt(ncov):>6}")
        print(f"  {lbl:<14}  " + "  ".join(cells))

    print("\n" + "=" * 80)
    print("SAMPLE COUNTS per k (covered / not covered / total)")
    print("=" * 80)
    for r in results_by_k:
        print(
            f"  k={r['k']:>2} ({r['run_id']}): {r['n_covered']:>5} covered, "
            f"{r['n_not_covered']:>5} not covered, {r['n']:>5} total"
        )


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def _bold(val: str, is_best: bool, is_second: bool) -> str:
    if is_best:
        return f"\\textbf{{{val}}}"
    if is_second:
        return f"\\underline{{{val}}}"
    return val


def generate_latex_table(results_by_k: list[dict[str, Any]]) -> str:
    """Generate a paper-ready LaTeX table from the analysis results."""
    col_keys = [
        ("gt_coverage_rate", "$y \\in \\mathcal{S}_k$", True),
        ("pred_in_exemplar_rate", "$\\hat{y} \\in \\mathcal{S}_k$", True),
        ("mean_unique_classes", "Unique classes", False),
        ("mv_matches_pred_rate", "$\\hat{y} = \\mathrm{mode}(\\mathcal{S}_k)$", True),
    ]

    # siunitx column specs: S for % columns, S[table-format=1.1] for unique classes
    col_specs_map = {"mean_unique_classes": "S[table-format=1.1]"}
    col_spec_inner = " ".join(
        col_specs_map.get(key, "S[table-format=2.1,table-number-alignment=right]")
        for key, _, _ in col_keys
    )
    col_spec = f"@{{}}c {col_spec_inner}@{{}}"

    # Headers must be wrapped in {} to prevent siunitx from parsing them
    header_cells = " & ".join(f"{{{h}}}" for _, h, _ in col_keys)

    rows = []
    for r in results_by_k:
        cells = [str(r["k"])]
        for key, _, _ in col_keys:
            val = r.get(key)
            if val is None or (isinstance(val, float) and val != val):
                cells.append("--")
                continue
            if key == "mean_unique_classes":
                cells.append(f"{val:.1f}")
            else:
                cells.append(f"\\qty{{{val * 100:.1f}}}{{\\percent}}")
        rows.append("        " + " & ".join(cells) + " \\\\")

    rows_str = "\n".join(rows)

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
    \\centering
    \\caption{{\\textbf{{Exemplar Selection Analysis.}} Effect of the number of in-context exemplars $k$ on retrieval and prediction behaviour for similarity-based selection with Qwen3-VL-8B on OF-ItW. $\\mathcal{{S}}_k$ denotes the support set of $k$ exemplars, $y$ the ground-truth label, and $\\hat{{y}}$ the predicted label.}}
    \\label{{tab:exemplar_analysis}}
    \\begin{{tabular}}{{{col_spec}}}
        \\toprule
        $k$ & {header_cells} \\\\
        \\midrule
{rows_str}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
\\endgroup"""
    return table


# ---------------------------------------------------------------------------
# Exemplar frequency plot
# ---------------------------------------------------------------------------

# Classes with train freq > 5% — these are worth plotting
_FREQ_CLASSES_MAIN = ["fall", "other", "fallen", "walk", "stand_up", "standing", "jump"]
_FREQ_CLASSES_RARE = [lbl for lbl in ALL_LABELS if lbl not in _FREQ_CLASSES_MAIN]


def plot_exemplar_frequency(results_by_k: list[dict[str, Any]], output_path: str) -> None:
    """Line plot of exemplar selection frequency vs k for frequent classes."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    k_values = [r["k"] for r in results_by_k]
    train_freq = results_by_k[0]["train_freq"]

    # Colour palette — warm for majority, cool for others
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(_FREQ_CLASSES_MAIN)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # --- Panel (a): frequent classes ---
    ax = axes[0]
    for cls in _FREQ_CLASSES_MAIN:
        freq_vals = [r["exemplar_freq"].get(cls, 0) * 100 for r in results_by_k]
        train_pct = train_freq.get(cls, 0) * 100
        ax.plot(k_values, freq_vals, marker="o", label=cls, color=class_colors[cls])
        ax.axhline(train_pct, color=class_colors[cls], linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("$k$ (number of exemplars)")
    ax.set_ylabel("Exemplar selection frequency (%)")
    ax.xaxis.set_major_locator(mticker.FixedLocator(k_values))
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=1.5, label="Exemplar freq."),
        Line2D(
            [0], [0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Train freq."
        ),
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + legend_elements,
        labels=ax.get_legend_handles_labels()[1] + ["— Exemplar freq.", "-- Train freq."],
        fontsize=7,
        ncol=2,
        loc="upper right",
    )
    ax.text(0.5, -0.18, "(a) Frequent classes", ha="center", transform=ax.transAxes, fontsize=9)

    # --- Panel (b): rare classes (train freq ≤ 5%) with meaningful presence ---
    ax2 = axes[1]
    rare_to_plot = [
        c
        for c in _FREQ_CLASSES_RARE
        if any(r["exemplar_freq"].get(c, 0) > 0.001 for r in results_by_k)
    ]
    rare_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, cls in enumerate(rare_to_plot):
        freq_vals = [r["exemplar_freq"].get(cls, 0) * 100 for r in results_by_k]
        train_pct = train_freq.get(cls, 0) * 100
        color = rare_colors[i % len(rare_colors)]
        ax2.plot(k_values, freq_vals, marker="o", label=cls, color=color)
        ax2.axhline(train_pct, color=color, linestyle="--", linewidth=0.8, alpha=0.5)

    ax2.set_xlabel("$k$ (number of exemplars)")
    ax2.set_ylabel("Exemplar selection frequency (%)")
    ax2.xaxis.set_major_locator(mticker.FixedLocator(k_values))
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.text(0.5, -0.18, "(b) Rare classes", ha="center", transform=ax2.transAxes, fontsize=9)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load all runs; k is inferred from each run's num_shots
    runs: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        print(f"\nLoading run {run_id} from project {args.project}")
        config, predictions = load_predictions_and_config(
            run_id, args.project, args.entity, args.output_root
        )
        if config.prompt.shot_selection != "similarity":
            print(
                f"  WARNING: shot_selection={config.prompt.shot_selection!r}, expected 'similarity'"
            )
        runs.append(
            {
                "run_id": run_id,
                "config": config,
                "predictions": predictions,
                "k": config.prompt.num_shots,
            }
        )

    # Sort by k so the table rows are ordered
    runs.sort(key=lambda r: r["k"])

    # Load embeddings once using the first run's config (all runs must share dataset/embeddings)
    print()
    query_emb, corpus_emb, corpus_labels = load_embeddings_and_corpus(
        runs[0]["config"], args.output_root
    )

    # Verify all runs have the same number of predictions as query embeddings
    for r in runs:
        if len(r["predictions"]) != query_emb.shape[0]:
            raise ValueError(
                f"Run {r['run_id']} has {len(r['predictions'])} predictions but "
                f"query embeddings have {query_emb.shape[0]} entries"
            )

    # Training class distribution (from corpus labels)
    corpus_counter = Counter(corpus_labels)
    total_corpus = len(corpus_labels)
    train_class_dist: dict[str, float] = {
        lbl: corpus_counter.get(lbl, 0) / total_corpus for lbl in ALL_LABELS
    }

    # Compute full similarity matrix once (shared across all runs)
    print("Computing similarity matrix...")
    similarity = compute_similarity_scores(query_emb, corpus_emb)
    print(f"Similarity matrix: {similarity.shape}")

    # Analyze each run at its own k using its own predictions
    results_by_k: list[dict[str, Any]] = []
    for r in runs:
        k = r["k"]
        print(f"Analyzing k={k} (run {r['run_id']})...")
        result = analyze_for_k(k, similarity, corpus_labels, r["predictions"], train_class_dist)
        result["run_id"] = r["run_id"]
        results_by_k.append(result)

    print_results(results_by_k)

    if args.latex:
        print("\n" + "=" * 80)
        print("LATEX TABLE")
        print("=" * 80)
        print(generate_latex_table(results_by_k))

    if args.plot:
        plot_exemplar_frequency(results_by_k, args.plot)

    if args.save_json:

        def _serialize(obj: object) -> object:
            if isinstance(obj, set):
                return sorted(obj)
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            return obj

        run_ids_str = "_".join(r["run_id"] for r in runs)
        out_path = (
            Path(args.output_root)
            / "predictions"
            / args.project
            / f"{run_ids_str}_exemplar_analysis.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_serialize(results_by_k), f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
