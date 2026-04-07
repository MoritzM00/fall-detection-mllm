import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-fewshot"

# Mapping run IDs to pretty display names (sorted by model size)
# TODO: Fill in W&B run IDs after running fewshot-similarity experiments
MODEL_NAMES: dict[str, str] = {
    "8y5wihub": "InternVL3.5-2B",
    "o6ta2929": "Qwen3-VL-2B",
    "j77j15xy": "InternVL3.5-8B",
    "dc2rkb9f": "Qwen3-VL-8B",
    "lp4yccqf": "Qwen3-VL-32B",
    "TODO_internvl_38b": "InternVL3.5-38B",
}

DATASET = "OOPS"
SPLIT = "cs"

# Whether to include the Fall ∪ Fallen binary metrics column group
INCLUDE_FALL_UNION_FALLEN = True

# Columns to apply heatmap coloring (indices: 0=BAcc, 2=F1, 5=fall_f1, 8=fallen_f1)
HEATMAP_COLUMNS = [0, 2, 5, 8]
if INCLUDE_FALL_UNION_FALLEN:
    HEATMAP_COLUMNS.append(11)  # fall_union_fallen_f1

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
]
if INCLUDE_FALL_UNION_FALLEN:
    METRICS_ORDER += [
        f"{DATASET}_{SPLIT}_fall_union_fallen_sensitivity",
        f"{DATASET}_{SPLIT}_fall_union_fallen_specificity",
        f"{DATASET}_{SPLIT}_fall_union_fallen_f1",
    ]


def fetch_run_data(api: wandb.Api, run_id: str) -> list[float | None]:
    """Fetches summary metrics as raw floats."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary

        row_values = []
        for metric_key in METRICS_ORDER:
            val = summary.get(metric_key)
            if val is not None:
                row_values.append(val * 100)
            else:
                row_values.append(None)
        return row_values

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return [None] * len(METRICS_ORDER)


def format_value(val: float | None, col_index: int, stats: list[dict[str, float]]) -> str:
    """Formats a value with bold/underline based on column stats, and heatmap for specific columns."""
    if val is None:
        return "--"

    val_rounded = round(val, 1)
    formatted_str = f"{val_rounded:.1f}"

    max_val = stats[col_index]["max"]
    second_val = stats[col_index]["second"]

    if val == max_val:
        formatted_str = f"\\textbf{{{formatted_str}}}"
    elif val == second_val:
        formatted_str = f"\\underline{{{formatted_str}}}"

    if col_index in HEATMAP_COLUMNS:
        min_val = stats[col_index]["min"]
        if max_val != min_val:
            level = int(round(10 + (val - min_val) / (max_val - min_val) * 90))
        else:
            level = 100
        formatted_str = f"\\gc{{{level}}}{{{formatted_str}}}"

    return formatted_str


def generate_latex() -> None:
    api = wandb.Api()

    all_rows = []
    for run_id, display_name in MODEL_NAMES.items():
        metrics = fetch_run_data(api, run_id)
        all_rows.append({"name": display_name, "metrics": metrics})

    # Calculate stats per column (max, second max, min)
    num_metrics = len(METRICS_ORDER)
    col_stats = []

    for i in range(num_metrics):
        values = [row["metrics"][i] for row in all_rows if row["metrics"][i] is not None]
        unique_vals = sorted(list(set(values)), reverse=True)

        stats = {
            "max": unique_vals[0] if len(unique_vals) > 0 else -1,
            "second": unique_vals[1] if len(unique_vals) > 1 else -1,
            "min": unique_vals[-1] if len(unique_vals) > 0 else 0,
        }
        col_stats.append(stats)

    # Format rows
    mllm_latex_rows = []
    for row in all_rows:
        formatted_metrics = []
        for i, val in enumerate(row["metrics"]):
            formatted_metrics.append(format_value(val, i, col_stats))

        metrics_str = " & ".join(formatted_metrics)
        mllm_latex_rows.append(f"{row['name']} & {metrics_str} \\\\")

    mllm_body = "\n".join(mllm_latex_rows)

    # Construct table
    if INCLUDE_FALL_UNION_FALLEN:
        col_spec = "@{}l rrr rrr rrr rrr@{}"
        union_header = " &\n\\multicolumn{3}{c}{Fall $\\cup$ Fallen}"
        union_cmidrule = " \\cmidrule(lr){11-13}"
        union_sub_header = (
            "\n & \\multicolumn{1}{c}{Se}   & \\multicolumn{1}{c}{Sp}  & \\multicolumn{1}{c}{F1}"
        )
    else:
        col_spec = "@{}l rrr rrr rrr@{}"
        union_header = ""
        union_cmidrule = ""
        union_sub_header = ""

    full_table = f"""
\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
\caption{{\\textbf{{Few-shot similarity fall detection results}} on the OmniFall-In-the-Wild dataset.
We report classification metrics for the 16-class action recognition task, as well as binary metrics for the Fall, Fallen and combined Fall/Fallen classes. Exemplars are selected via cosine similarity over pre-computed embeddings ($k=3$). The best results are highlighted in \\textbf{{bold}}, and the second-best are \\underline{{underlined}}. Darker cells indicate better performance.}}
\\label{{tab:fewshot_similarity_results}}

\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{{col_spec}}}
\\toprule
% Top Header Row
\\multirow{{2}}{{*}}{{{{Model}}}} &
\\multicolumn{{3}}{{c}}{{16-class}} &
\\multicolumn{{3}}{{c}}{{Fall $\\Delta$}} &
\\multicolumn{{3}}{{c}}{{Fallen $\\Delta$}}{union_header} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}{union_cmidrule}

% Sub Header Row
 & \\multicolumn{{1}}{{c}}{{BAcc}} & \\multicolumn{{1}}{{c}}{{Acc}} & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}}{union_sub_header} \\\\
\\midrule

{mllm_body}

\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
\\endgroup
"""

    print(full_table)


if __name__ == "__main__":
    generate_latex()
