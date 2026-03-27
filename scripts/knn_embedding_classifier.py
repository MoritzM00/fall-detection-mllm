"""kNN classifier on precomputed Qwen3-VL embeddings.

Train embeddings serve as the corpus; eval set is classified via majority vote
over K nearest neighbors using cosine similarity.
"""

import argparse
import collections
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from falldet.embeddings import load_embeddings
from falldet.metrics.base import compute_metrics

console = Console()

LATEX_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "fall_sensitivity",
    "fall_specificity",
    "fall_f1",
    "fallen_sensitivity",
    "fallen_specificity",
    "fallen_f1",
]
HEATMAP_COLUMNS = [0, 2, 5, 8]  # BAcc, MacF1, Fall F1, Fallen F1

# Embedding filename parameters (fixed for current Qwen3-VL runs)
_NUM_FRAMES = 16
_FPS_STR = "7_5"
_DATA_SIZE = 448


def _embedding_path(emb_dir: Path, dataset_prefix: str, split: str, model: str) -> Path:
    filename = f"{dataset_prefix}_{split}_{_NUM_FRAMES}@{_FPS_STR}_{model}_{_DATA_SIZE}.pt"
    return emb_dir / filename


def extract_labels(samples: list[dict]) -> list[str]:
    return [s["label_str"] for s in samples]


def knn_classify(
    train_emb: torch.Tensor,
    train_labels: list[str],
    eval_emb: torch.Tensor,
    k_values: list[int],
) -> dict[int, list[str]]:
    """Return predicted labels for each K via majority-vote kNN."""
    dist_matrix = torch.cdist(eval_emb.float(), train_emb.float())  # (n_eval, n_train)
    max_k = max(k_values)
    # topk smallest distances for each eval sample — shape (n_eval, max_k)
    _, top_indices = torch.topk(dist_matrix, k=max_k, dim=1, largest=False)

    predictions: dict[int, list[str]] = {}
    for k in k_values:
        preds = []
        for i in range(len(eval_emb)):
            neighbor_indices = top_indices[i, :k].tolist()
            neighbor_labels = [train_labels[idx] for idx in neighbor_indices]
            vote = collections.Counter(neighbor_labels).most_common(1)[0][0]
            preds.append(vote)
        predictions[k] = preds
    return predictions


def evaluate_model(
    emb_dir: Path,
    dataset_prefix: str,
    model: str,
    eval_split: str,
    k_values: list[int],
) -> dict[int, dict[str, float]]:
    """Load embeddings, run kNN, return metrics per K."""
    train_path = _embedding_path(emb_dir, dataset_prefix, "train", model)
    eval_path = _embedding_path(emb_dir, dataset_prefix, eval_split, model)

    console.print(f"  Loading train: [dim]{train_path.name}[/dim]")
    train_emb, train_samples = load_embeddings(train_path)
    train_labels = extract_labels(train_samples)

    console.print(f"  Loading eval:  [dim]{eval_path.name}[/dim]")
    eval_emb, eval_samples = load_embeddings(eval_path)
    eval_labels = extract_labels(eval_samples)

    console.print(f"  Train: {train_emb.shape[0]} samples | Eval: {eval_emb.shape[0]} samples")

    predictions = knn_classify(train_emb, train_labels, eval_emb, k_values)

    results: dict[int, dict[str, float]] = {}
    for k, preds in predictions.items():
        results[k] = compute_metrics(preds, eval_labels)
    return results


def print_results_table(
    all_results: dict[str, dict[int, dict[str, float]]],
    k_values: list[int],
    eval_split: str,
) -> None:
    """Print a Rich table: rows = K values, columns = metric × model."""
    metrics_to_show = [
        ("accuracy", "Acc"),
        ("balanced_accuracy", "BAcc"),
        ("macro_f1", "Mac F1"),
        ("fall_f1", "Fall F1"),
    ]
    models = list(all_results.keys())

    table = Table(
        title=f"kNN Classification — split: {eval_split}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("K", style="bold", justify="right")
    for model in models:
        short = model.replace("Qwen3-VL-Embedding-", "")
        for _, col_label in metrics_to_show:
            table.add_column(f"{short}\n{col_label}", justify="right")

    for k in k_values:
        row = [str(k)]
        for model in models:
            m = all_results[model][k]
            for metric_key, _ in metrics_to_show:
                row.append(f"{m[metric_key]:.3f}")
        table.add_row(*row)

    console.print(table)

    # Per-class F1 table — one per model, rows = class, columns = K
    all_classes = [
        "walk",
        "fall",
        "fallen",
        "sit_down",
        "sitting",
        "lie_down",
        "lying",
        "stand_up",
        "standing",
        "other",
        "kneel_down",
        "kneeling",
        "squat_down",
        "squatting",
        "crawl",
        "jump",
    ]
    for model in models:
        short = model.replace("Qwen3-VL-Embedding-", "")
        per_class_table = Table(
            title=f"Per-class F1 — {short} — split: {eval_split}",
            show_header=True,
            header_style="bold magenta",
        )
        per_class_table.add_column("Class", style="bold")
        for k in k_values:
            per_class_table.add_column(f"K={k}", justify="right")

        for cls in all_classes:
            row = [cls]
            for k in k_values:
                val = all_results[model][k].get(f"{cls}_f1")
                row.append(f"{val:.3f}" if val is not None else "—")
            per_class_table.add_row(*row)

        console.print(per_class_table)


def _col_stats(values: list[float]) -> dict[str, float]:
    unique = sorted(set(values), reverse=True)
    return {
        "max": unique[0] if unique else -1.0,
        "second": unique[1] if len(unique) > 1 else -1.0,
        "min": unique[-1] if unique else 0.0,
    }


def _format_latex_cell(
    raw: float,
    col_idx: int,
    global_stats: list[dict[str, float]],
    heatmap_stats: list[dict[str, float]],
) -> str:
    """Format a single metric value with bold/underline and optional heatmap."""
    val_rounded = round(raw, 1)
    s = f"{val_rounded:.1f}"

    if raw == global_stats[col_idx]["max"]:
        s = f"\\textbf{{{s}}}"
    elif raw == global_stats[col_idx]["second"]:
        s = f"\\underline{{{s}}}"

    if col_idx in HEATMAP_COLUMNS:
        mn = heatmap_stats[col_idx]["min"]
        mx = heatmap_stats[col_idx]["max"]
        level = int(round(10 + (raw - mn) / (mx - mn) * 90)) if mx != mn else 100
        s = f"\\gc{{{level}}}{{{s}}}"

    return s


def _latex_table_for_model(
    model: str,
    results: dict[int, dict[str, float]],
    k_values: list[int],
    eval_split: str,
) -> str:
    """Build a single LaTeX table for one model (rows = K values)."""
    model.replace("Qwen3-VL-Embedding-", "")

    # Collect values in 0–100 scale per column
    col_vals: list[list[float]] = [[] for _ in range(len(LATEX_METRICS))]
    for k in k_values:
        for ci, key in enumerate(LATEX_METRICS):
            col_vals[ci].append(results[k][key] * 100)

    stats = [_col_stats(vals) for vals in col_vals]

    col_spec = "@{}l rrr rrr rrr@{}"

    subgroup_row = (
        " & "
        "\\multicolumn{3}{c}{16-class} & "
        "\\multicolumn{3}{c}{Fall $\\Delta$} & "
        "\\multicolumn{3}{c}{Fallen $\\Delta$} \\\\"
    )
    sub_header_row = (
        " & "
        "\\multicolumn{1}{c}{BAcc} & \\multicolumn{1}{c}{Acc} & \\multicolumn{1}{c}{F1} & "
        "\\multicolumn{1}{c}{Se} & \\multicolumn{1}{c}{Sp} & \\multicolumn{1}{c}{F1} & "
        "\\multicolumn{1}{c}{Se} & \\multicolumn{1}{c}{Sp} & \\multicolumn{1}{c}{F1} \\\\"
    )

    data_rows = []
    for k in k_values:
        cells = [f"$k={k}$"]
        for ci, key in enumerate(LATEX_METRICS):
            raw = results[k][key] * 100
            cells.append(_format_latex_cell(raw, ci, stats, stats))
        data_rows.append(" & ".join(cells) + " \\\\")
    data_body = "\n".join(data_rows)

    return f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{table}}[htp]
\\centering
\\caption{{$k$-NN classification results on OF-ItW {eval_split} split for task-specific embeddings of {model}.
Rows indicate the number of neighbors $k$. Best value is \\textbf{{bolded}} and second-best \\underline{{underlined}}.
Darker cells indicate better performance. \\textbf{{B}}alanced \\textbf{{Acc}}uracy, \\textbf{{Se}}nsitivity, and \\textbf{{Sp}}ecificity}}
\\label{{tab:knn_results}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{subgroup_row}
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}
{sub_header_row}
\\midrule
{data_body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\endgroup"""


def print_latex_table(
    all_results: dict[str, dict[int, dict[str, float]]],
    k_values: list[int],
    eval_split: str,
) -> None:
    """Print one LaTeX table per model (rows = K values)."""
    for model, results in all_results.items():
        print(_latex_table_for_model(model, results, k_values, eval_split))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kNN classifier on precomputed embeddings")
    parser.add_argument(
        "--embedding-dir",
        default="outputs/embeddings",
        help="Directory containing .pt embedding files (default: outputs/embeddings)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen3-VL-Embedding-2B", "Qwen3-VL-Embedding-8B"],
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 10, 15, 20],
        help="K values for kNN (default: 1 3 5 7 10 15 20)",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        help="Eval split to use (default: test)",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="OOPS_cs",
        help="Dataset prefix (default: OOPS_cs)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output a LaTeX table instead of Rich console tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.embedding_dir)

    all_results: dict[str, dict[int, dict[str, float]]] = {}
    for model in args.models:
        console.print(f"\n[bold green]Model: {model}[/bold green]")
        all_results[model] = evaluate_model(
            emb_dir=emb_dir,
            dataset_prefix=args.dataset_prefix,
            model=model,
            eval_split=args.eval_split,
            k_values=args.k_values,
        )

    if args.latex:
        print_latex_table(all_results, args.k_values, args.eval_split)
    else:
        console.print()
        print_results_table(all_results, args.k_values, args.eval_split)


if __name__ == "__main__":
    main()
