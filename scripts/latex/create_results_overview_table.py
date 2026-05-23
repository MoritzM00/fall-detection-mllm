"""Build a results-overview table comparing Qwen3-VL-8B across experimental regimes.

Fetches metrics for one run per regime from W&B and prepends two hard-coded
VMAE-K400 baselines. Output: ~/thesis-overleaf/tables/results_overview.tex.
"""

from pathlib import Path

import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT_ZS = "fall-detection-zeroshot-v4"
PROJECT_FS = "fall-detection-fewshot"

# Output path — set to "" to print to stdout only
OUTPUT_PATH = Path("~/thesis-overleaf/tables/results_overview.tex").expanduser()

DATASET = "OOPS"
SPLIT = "cs"

# Columns to apply heatmap coloring (BAcc, macro_f1, fall_f1, fallen_f1, fall_union_fallen_f1)
HEATMAP_COLUMNS = [0, 2, 5, 8, 11]

# ==========================================
# W&B RUNS — one per regime (Qwen3-VL-8B on OF-ItW)
# ==========================================
MLLM_RUNS: list[tuple[str, str, str]] = [
    # (display_name, project, run_id)
    ("Zero-shot", PROJECT_ZS, "p1r3exbe"),
    ("Zero-shot + CoT", PROJECT_ZS, "fmmrnf5j"),
    # Best few-shot ICL config: similarity-based selection, k=4
    ("Few-shot ICL", PROJECT_FS, "dc2rkb9f"),
    # SFT LoRA rank=16, trained on ItW only
    ("SFT LoRA (rank 16)", PROJECT_ZS, "a5nz8f5v"),
]

# ==========================================
# HARD-CODED BASELINES
# ==========================================
# VMAE-K400 trained on staged + synthetic data, evaluated on OF-ItW (OOD w.r.t. in-the-wild).
# Source: create_latex_table.py SPECIALIZED_MODEL_METRICS_ALL
VMAE_OOD: list[float] = [21.4, 47.6, 21.9, 72.9, 85.4, 65.6, 33.1, 96.3, 41.0, 68.2, 82.4, 67.5]

# VMAE-K400 trained including the OF-ItW training split (in-distribution).
# Source: thesis-overleaf/tables/fewshot_similarity_results.tex, specialized-model row.
VMAE_INDOMAIN: list[float] = [
    25.8,
    95.5,
    26.7,
    79.3,
    91.3,
    76.0,
    57.2,
    92.8,
    53.6,
    80.9,
    84.7,
    76.9,
]

# ==========================================
# METRIC MAPPING
# ==========================================
METRICS_ORDER = [
    f"{DATASET}_{SPLIT}_balanced_accuracy",
    f"{DATASET}_{SPLIT}_accuracy",
    f"{DATASET}_{SPLIT}_macro_f1",
    f"{DATASET}_{SPLIT}_fall_sensitivity",
    f"{DATASET}_{SPLIT}_fall_specificity",
    f"{DATASET}_{SPLIT}_fall_f1",
    f"{DATASET}_{SPLIT}_fallen_sensitivity",
    f"{DATASET}_{SPLIT}_fallen_specificity",
    f"{DATASET}_{SPLIT}_fallen_f1",
    f"{DATASET}_{SPLIT}_fall_union_fallen_sensitivity",
    f"{DATASET}_{SPLIT}_fall_union_fallen_specificity",
    f"{DATASET}_{SPLIT}_fall_union_fallen_f1",
]


def fetch_run_data(api: wandb.Api, project: str, run_id: str) -> list[float | None]:
    try:
        run = api.run(f"{ENTITY}/{project}/{run_id}")
        summary = run.summary
        row_values: list[float | None] = []
        for metric_key in METRICS_ORDER:
            val = summary.get(metric_key)
            row_values.append(float(val) * 100.0 if val is not None else None)
        return row_values
    except Exception as exc:
        print(f"Error fetching {project}/{run_id}: {exc}")
        return [None] * len(METRICS_ORDER)


def compute_col_stats(all_rows: list[dict]) -> list[dict]:
    stats = []
    for i in range(len(METRICS_ORDER)):
        values = [r["metrics"][i] for r in all_rows if r["metrics"][i] is not None]
        unique = sorted(set(values), reverse=True)
        stats.append(
            {
                "max": unique[0] if unique else -1,
                "second": unique[1] if len(unique) > 1 else -1,
                "min": unique[-1] if unique else 0,
            }
        )
    return stats


def format_value(val: float | None, col_index: int, stats: list[dict]) -> str:
    if val is None:
        return "--"

    val_rounded = round(val, 1)
    formatted = f"{val_rounded:.1f}"
    s = stats[col_index]

    if val == s["max"]:
        formatted = f"\\textbf{{{formatted}}}"
    elif val == s["second"]:
        formatted = f"\\underline{{{formatted}}}"

    if col_index in HEATMAP_COLUMNS:
        if s["max"] != s["min"]:
            level = int(round(10 + (val - s["min"]) / (s["max"] - s["min"]) * 90))
        else:
            level = 100
        formatted = f"\\gc{{{level}}}{{{formatted}}}"

    return formatted


def render_row(row: dict, stats: list[dict]) -> str:
    cells = [format_value(v, i, stats) for i, v in enumerate(row["metrics"])]
    return f"{row['name']} & {' & '.join(cells)} \\\\"


def generate_latex() -> str:
    api = wandb.Api()

    all_rows: list[dict] = [
        {"name": r"VMAE-K400 (OOD)", "metrics": VMAE_OOD, "group": "specialized"},
        {"name": r"VMAE-K400 (in-dist.)", "metrics": VMAE_INDOMAIN, "group": "specialized"},
    ]
    for display_name, project, run_id in MLLM_RUNS:
        metrics = fetch_run_data(api, project, run_id)
        all_rows.append({"name": display_name, "metrics": metrics, "group": "mllm"})

    stats = compute_col_stats(all_rows)

    specialized_rows = "\n".join(
        render_row(r, stats) for r in all_rows if r["group"] == "specialized"
    )
    mllm_rows = "\n".join(render_row(r, stats) for r in all_rows if r["group"] == "mllm")

    table = (
        r"""
\begingroup
\renewcommand{\arraystretch}{1.2}
\begin{table}[htp]
\caption{\textbf{Results overview.} Qwen3-VL-8B on OF-ItW across all experimental regimes compared to two VMAE-K400 baselines: one trained on staged and synthetic data only (out-of-distribution with respect to OF-ItW) and one trained on the OF-ItW training split (in-distribution). Best results in \textbf{bold}, second-best \underline{underlined}. Darker cells indicate better performance.}
\label{tab:results_overview}

\resizebox{\columnwidth}{!}{
\begin{tabular}{@{}l rrr rrr rrr rrr@{}}
\toprule
% Top Header Row
\multirow{2}{*}{{Method}} &
\multicolumn{3}{c}{16-class} &
\multicolumn{3}{c}{Fall $\Delta$} &
\multicolumn{3}{c}{Fallen $\Delta$} &
\multicolumn{3}{c}{Fall $\cup$ Fallen} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}

% Sub Header Row
 & \multicolumn{1}{c}{BAcc} & \multicolumn{1}{c}{Acc} & \multicolumn{1}{c}{F1}
 & \multicolumn{1}{c}{Se}   & \multicolumn{1}{c}{Sp}  & \multicolumn{1}{c}{F1}
 & \multicolumn{1}{c}{Se}   & \multicolumn{1}{c}{Sp}  & \multicolumn{1}{c}{F1}
 & \multicolumn{1}{c}{Se}   & \multicolumn{1}{c}{Sp}  & \multicolumn{1}{c}{F1} \\
\midrule

\multicolumn{13}{@{}l}{\textit{Specialized Model}} \\
"""
        + specialized_rows
        + r"""
\midrule

\multicolumn{13}{@{}l}{\textit{Qwen3-VL-8B on OF-ItW}} \\
"""
        + mllm_rows
        + r"""

\bottomrule
\end{tabular}}
\end{table}
\endgroup
"""
    )
    return table


def main() -> None:
    latex = generate_latex()

    if OUTPUT_PATH and str(OUTPUT_PATH).strip():
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(latex)
        print(f"Written to {OUTPUT_PATH}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
