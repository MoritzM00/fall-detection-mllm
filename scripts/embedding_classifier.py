"""Sklearn classifiers on precomputed Qwen3-VL embeddings.

Train embeddings serve as training data; eval set is classified using
various scikit-learn classifiers. Supports KNN, MLP, SVM, Logistic
Regression, and Histogram Gradient Boosting.
"""

import argparse
import copy
import sys
import warnings
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC

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

# ── Classifier registry ──────────────────────────────────────────────

_MLP_BASE: dict = {"max_iter": 1, "warm_start": True, "solver": "adam"}

CLASSIFIER_REGISTRY: dict[str, tuple[str, type, dict]] = {
    "knn": (
        "$k$-NN ($k{=}5$)",
        KNeighborsClassifier,
        {"n_neighbors": 5, "metric": "euclidean"},
    ),
    # ── MLP variants ──
    "mlp": (
        "MLP",
        MLPClassifier,
        {**_MLP_BASE, "hidden_layer_sizes": (512, 256, 128, 64)},
    ),
    "mlp_4l": (
        "MLP (256-128-64-32)",
        MLPClassifier,
        {**_MLP_BASE, "hidden_layer_sizes": (256, 128, 64, 32)},
    ),
    "mlp_2l": (
        "MLP (512-256)",
        MLPClassifier,
        {**_MLP_BASE, "hidden_layer_sizes": (512, 256)},
    ),
    "mlp_narrow": (
        "MLP (128-64)",
        MLPClassifier,
        {**_MLP_BASE, "hidden_layer_sizes": (128, 64)},
    ),
    # ── Other classifiers ──
    "svm": (
        "SVM (RBF)",
        SVC,
        {"kernel": "rbf", "C": 1.0},
    ),
    "lr": (
        "Logistic Regression",
        LogisticRegression,
        {"max_iter": 1000},
    ),
    "gb": (
        "Hist Gradient Boosting",
        HistGradientBoostingClassifier,
        {"max_iter": 1, "warm_start": True},
    ),
}

# Classifiers that support val-based early stopping
_EARLY_STOP_KEYS = {"mlp", "mlp_4l", "mlp_2l", "mlp_narrow", "gb"}


def _embedding_path(emb_dir: Path, dataset_prefix: str, split: str, model: str) -> Path:
    filename = f"{dataset_prefix}_{split}_{_NUM_FRAMES}@{_FPS_STR}_{model}_{_DATA_SIZE}.pt"
    return emb_dir / filename


def extract_labels(samples: list[dict]) -> list[str]:
    return [s["label_str"] for s in samples]


# ── Training & prediction ────────────────────────────────────────────

_MAX_EPOCHS = 500
_PATIENCE = 15
_HGBC_BATCH = 10  # boosting rounds per early-stopping check


def _snapshot_state(clf: MLPClassifier | HistGradientBoostingClassifier) -> object:
    """Capture restorable internal state for early stopping."""
    if isinstance(clf, MLPClassifier):
        return copy.deepcopy(clf.coefs_), copy.deepcopy(clf.intercepts_)
    # HGBC: deep-copy the whole estimator (lightweight — only tree pointers)
    return copy.deepcopy(clf)


def _restore_state(
    clf: MLPClassifier | HistGradientBoostingClassifier, state: object
) -> MLPClassifier | HistGradientBoostingClassifier:
    if isinstance(clf, MLPClassifier):
        clf.coefs_, clf.intercepts_ = state  # type: ignore[assignment]
        return clf
    return state  # type: ignore[return-value]


def _train_with_early_stopping(
    clf_key: str,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    seed: int,
) -> MLPClassifier | HistGradientBoostingClassifier:
    """Train a warm-startable classifier with val-based early stopping."""
    _, clf_class, default_kwargs = CLASSIFIER_REGISTRY[clf_key]
    kwargs = {**default_kwargs, "random_state": seed}

    is_hgbc = issubclass(clf_class, HistGradientBoostingClassifier)
    max_rounds = _MAX_EPOCHS if not is_hgbc else _MAX_EPOCHS // _HGBC_BATCH

    clf = clf_class(**kwargs)
    best_val_score = -1.0
    wait = 0
    best_state: object | None = None

    for step in range(max_rounds):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(train_emb, train_labels)

        if is_hgbc:
            clf.max_iter += _HGBC_BATCH  # schedule next batch

        val_preds = clf.predict(val_emb)
        val_score = balanced_accuracy_score(val_labels, val_preds)

        iteration = (step + 1) * _HGBC_BATCH if is_hgbc else step + 1
        if val_score > best_val_score + 1e-4:
            best_val_score = val_score
            best_state = _snapshot_state(clf)
            wait = 0
        else:
            wait += 1
            if wait >= _PATIENCE:
                console.print(
                    f"    Early stop at iter {iteration} (best val BAcc: {best_val_score:.4f})"
                )
                break

    if best_state is not None:
        clf = _restore_state(clf, best_state)

    return clf


def train_and_predict(
    clf_key: str,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    eval_emb: np.ndarray,
    label_encoder: LabelEncoder,
    seed: int,
    val_emb: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
) -> list[str]:
    """Instantiate a classifier, fit on train, predict on eval."""
    if clf_key in _EARLY_STOP_KEYS and val_emb is not None and val_labels is not None:
        clf = _train_with_early_stopping(
            clf_key, train_emb, train_labels, val_emb, val_labels, seed
        )
        preds_encoded = clf.predict(eval_emb)
        return label_encoder.inverse_transform(preds_encoded).tolist()

    display_name, clf_class, default_kwargs = CLASSIFIER_REGISTRY[clf_key]
    kwargs = {**default_kwargs}

    # Inject random_state for classifiers that support it
    if clf_key != "knn":
        kwargs["random_state"] = seed

    clf = clf_class(**kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf.fit(train_emb, train_labels)

    preds_encoded = clf.predict(eval_emb)
    return label_encoder.inverse_transform(preds_encoded).tolist()


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_model(
    emb_dir: Path,
    dataset_prefix: str,
    model: str,
    eval_split: str,
    classifier_keys: list[str],
    do_normalize: bool,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Load embeddings, run each classifier, return {clf_key: metrics}."""
    train_path = _embedding_path(emb_dir, dataset_prefix, "train", model)
    eval_path = _embedding_path(emb_dir, dataset_prefix, eval_split, model)
    val_path = _embedding_path(emb_dir, dataset_prefix, "val", model)

    console.print(f"  Loading train: [dim]{train_path.name}[/dim]")
    train_emb, train_samples = load_embeddings(train_path)
    train_labels_str = extract_labels(train_samples)

    console.print(f"  Loading eval:  [dim]{eval_path.name}[/dim]")
    eval_emb, eval_samples = load_embeddings(eval_path)
    eval_labels_str = extract_labels(eval_samples)

    # Val embeddings for MLP early stopping
    val_np: np.ndarray | None = None
    val_encoded: np.ndarray | None = None
    val_labels_str: list[str] = []
    if val_path.exists() and _EARLY_STOP_KEYS.intersection(classifier_keys):
        console.print(f"  Loading val:   [dim]{val_path.name}[/dim]")
        val_emb, val_samples = load_embeddings(val_path)
        val_labels_str = extract_labels(val_samples)
        val_np = val_emb.float().numpy()

    console.print(f"  Train: {train_emb.shape[0]} samples | Eval: {eval_emb.shape[0]} samples")

    # Convert to numpy
    train_np = train_emb.float().numpy()
    eval_np = eval_emb.float().numpy()

    if do_normalize:
        console.print("  [dim]L2-normalizing embeddings[/dim]")
        train_np = normalize(train_np, norm="l2")
        eval_np = normalize(eval_np, norm="l2")
        if val_np is not None:
            val_np = normalize(val_np, norm="l2")

    # Encode labels to ints for sklearn — fit on union of all splits so
    # unseen classes in val/eval don't cause errors
    all_labels = train_labels_str + eval_labels_str + val_labels_str
    le = LabelEncoder()
    le.fit(all_labels)
    train_encoded = le.transform(train_labels_str)
    if val_np is not None:
        val_encoded = le.transform(val_labels_str)

    results: dict[str, dict[str, float]] = {}
    for clf_key in classifier_keys:
        display_name = CLASSIFIER_REGISTRY[clf_key][0]
        console.print(f"  Running [bold]{display_name}[/bold] …")
        preds = train_and_predict(
            clf_key,
            train_np,
            train_encoded,
            eval_np,
            le,
            seed,
            val_emb=val_np,
            val_labels=val_encoded,
        )
        results[clf_key] = compute_metrics(preds, eval_labels_str)

    return results


# ── Rich console output ──────────────────────────────────────────────


def print_results_table(
    all_results: dict[str, dict[str, dict[str, float]]],
    classifier_keys: list[str],
    eval_split: str,
) -> None:
    """Rich table: rows = classifiers, columns = metric × model."""
    metrics_to_show = [
        ("accuracy", "Acc"),
        ("balanced_accuracy", "BAcc"),
        ("macro_f1", "Mac F1"),
        ("fall_f1", "Fall F1"),
    ]
    models = list(all_results.keys())

    table = Table(
        title=f"Embedding Classifiers — split: {eval_split}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Classifier", style="bold")
    for model in models:
        short = model.replace("Qwen3-VL-Embedding-", "")
        for _, col_label in metrics_to_show:
            table.add_column(f"{short}\n{col_label}", justify="right")

    for clf_key in classifier_keys:
        display_name = CLASSIFIER_REGISTRY[clf_key][0]
        row = [display_name]
        for model in models:
            m = all_results[model][clf_key]
            for metric_key, _ in metrics_to_show:
                row.append(f"{m[metric_key]:.3f}")
        table.add_row(*row)

    console.print(table)

    # Per-class F1 table — one per model
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
        for clf_key in classifier_keys:
            per_class_table.add_column(CLASSIFIER_REGISTRY[clf_key][0], justify="right")

        for cls in all_classes:
            row = [cls]
            for clf_key in classifier_keys:
                val = all_results[model][clf_key].get(f"{cls}_f1")
                row.append(f"{val:.3f}" if val is not None else "—")
            per_class_table.add_row(*row)

        console.print(per_class_table)


# ── LaTeX output ──────────────────────────────────────────────────────


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
    results: dict[str, dict[str, float]],
    classifier_keys: list[str],
    eval_split: str,
) -> str:
    """Build a single LaTeX table for one model (rows = classifiers)."""
    # Collect values in 0–100 scale per column
    col_vals: list[list[float]] = [[] for _ in range(len(LATEX_METRICS))]
    for clf_key in classifier_keys:
        for ci, key in enumerate(LATEX_METRICS):
            col_vals[ci].append(results[clf_key][key] * 100)

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
    for clf_key in classifier_keys:
        display_name = CLASSIFIER_REGISTRY[clf_key][0]
        cells = [display_name]
        for ci, key in enumerate(LATEX_METRICS):
            raw = results[clf_key][key] * 100
            cells.append(_format_latex_cell(raw, ci, stats, stats))
        data_rows.append(" & ".join(cells) + " \\\\")
    data_body = "\n".join(data_rows)

    return f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{table}}[htp]
\\centering
\\caption{{Classifier comparison on OF-ItW {eval_split} split using task-specific embeddings of {model}.
Best value is \\textbf{{bolded}} and second-best \\underline{{underlined}}.
Darker cells indicate better performance. \\textbf{{B}}alanced \\textbf{{Acc}}uracy, \\textbf{{Se}}nsitivity, and \\textbf{{Sp}}ecificity}}
\\label{{tab:clf_results}}
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
    all_results: dict[str, dict[str, dict[str, float]]],
    classifier_keys: list[str],
    eval_split: str,
) -> None:
    """Print one LaTeX table per model (rows = classifiers)."""
    for model, results in all_results.items():
        print(_latex_table_for_model(model, results, classifier_keys, eval_split))
        print()


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sklearn classifiers on precomputed embeddings")
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
        "--classifiers",
        nargs="+",
        choices=list(CLASSIFIER_REGISTRY.keys()),
        default=list(CLASSIFIER_REGISTRY.keys()),
        help="Classifiers to run (default: all)",
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
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings before classification (default: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for model in args.models:
        console.print(f"\n[bold green]Model: {model}[/bold green]")
        all_results[model] = evaluate_model(
            emb_dir=emb_dir,
            dataset_prefix=args.dataset_prefix,
            model=model,
            eval_split=args.eval_split,
            classifier_keys=args.classifiers,
            do_normalize=args.normalize,
            seed=args.seed,
        )

    if args.latex:
        print_latex_table(all_results, args.classifiers, args.eval_split)
    else:
        console.print()
        print_results_table(all_results, args.classifiers, args.eval_split)


if __name__ == "__main__":
    main()
