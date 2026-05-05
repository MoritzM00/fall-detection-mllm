"""Merged kNN + ML classifier LaTeX table on Qwen3-VL embeddings.

Produces a single table with kNN rows (k=1,3,5,7,10,15,20) above a midrule
and ML classifiers (MLP, SVM, LR, RF) below.  Bold/underline/heatmap are
computed over the full merged row set so comparisons are global.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import embedding_classifier
import knn_embedding_classifier

# Redirect console output from imported modules to stderr so that
# --latex produces clean stdout suitable for shell redirection.
_stderr_console = Console(file=sys.stderr)
embedding_classifier.console = _stderr_console
knn_embedding_classifier.console = _stderr_console

from embedding_classifier import (
    LATEX_METRICS,
    _col_stats,
    _format_latex_cell,
)
from embedding_classifier import (
    evaluate_model as evaluate_ml,
)
from knn_embedding_classifier import evaluate_model as evaluate_knn

_KNN_K_VALUES = [1, 3, 5, 7, 10, 15, 20]
_ML_CLASSIFIER_KEYS = ["mlp", "svm", "lr", "rf"]
_ML_DISPLAY_NAMES = {
    "mlp": "MLP",
    "svm": "SVM (RBF)",
    "lr": "Logistic Regression",
    "rf": "Random Forest",
}

_NUM_FRAMES = 16
_FPS_STR = "7_5"
_DATA_SIZE = 448


def _embedding_path(emb_dir: Path, dataset_prefix: str, split: str, model: str) -> Path:
    filename = f"{dataset_prefix}_{split}_{_NUM_FRAMES}@{_FPS_STR}_{model}_{_DATA_SIZE}.pt"
    return emb_dir / filename


def build_merged_latex(
    knn_results: dict[int, dict[str, float]],
    clf_results: dict[str, dict[str, float]],
    model: str,
    eval_split: str,
) -> str:
    # Ordered row list — None signals \midrule
    rows: list[tuple[str, dict[str, float]] | None] = [
        (f"{k}-NN", knn_results[k]) for k in _KNN_K_VALUES
    ]
    rows.append(None)
    rows += [(name, clf_results[key]) for key, name in _ML_DISPLAY_NAMES.items()]

    numeric_rows = [r for r in rows if r is not None]

    # Compute global stats over all numeric rows
    col_vals: list[list[float]] = [[] for _ in range(len(LATEX_METRICS))]
    for _, metrics in numeric_rows:
        for ci, key in enumerate(LATEX_METRICS):
            col_vals[ci].append(metrics[key] * 100)

    stats = [_col_stats(vals) for vals in col_vals]

    col_spec = "@{}c rrr rrr rrr@{}"
    subgroup_row = (
        "\\multirow{2}{*}{Model} & "
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

    data_lines = []
    for row in rows:
        if row is None:
            data_lines.append("\\midrule")
            continue
        label, metrics = row
        cells = [label]
        for ci, key in enumerate(LATEX_METRICS):
            raw = metrics[key] * 100
            cells.append(_format_latex_cell(raw, ci, stats, stats))
        data_lines.append(" & ".join(cells) + " \\\\")

    data_body = "\n".join(data_lines)

    return f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{table}}[htp]
\\centering
\\caption{{\\textbf{{Embedding-based classifier comparison on OF-ItW}} {eval_split} split using task-specific embeddings of {model}~\\cite{{li_qwen3-vl-embedding_2026}}.
The upper block shows $k$-nearest-neighbour results for varying $k$; the lower block shows other classifiers.
Best value is \\textbf{{bolded}} and second-best \\underline{{underlined}} across all rows.
Darker cells indicate better performance. \\textbf{{B}}alanced \\textbf{{Acc}}uracy, \\textbf{{Se}}nsitivity, and \\textbf{{Sp}}ecificity}}
\\label{{tab:merged_clf_results}}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merged kNN + ML classifier LaTeX table")
    parser.add_argument("--embedding-dir", default="outputs/embeddings")
    parser.add_argument("--model", default="Qwen3-VL-Embedding-2B")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--dataset-prefix", default="OOPS_cs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.embedding_dir)

    knn_results = evaluate_knn(
        emb_dir=emb_dir,
        dataset_prefix=args.dataset_prefix,
        model=args.model,
        eval_split=args.eval_split,
        k_values=_KNN_K_VALUES,
    )

    clf_results = evaluate_ml(
        emb_dir=emb_dir,
        dataset_prefix=args.dataset_prefix,
        model=args.model,
        eval_split=args.eval_split,
        classifier_keys=_ML_CLASSIFIER_KEYS,
        do_normalize=True,
        seed=args.seed,
    )

    if args.latex:
        print(build_merged_latex(knn_results, clf_results, args.model, args.eval_split))
    else:
        print("Run with --latex to get LaTeX output.")


if __name__ == "__main__":
    main()
