import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-zeroshot-v4"

# Mapping run IDs to pretty display names (sorted by model size)
MODEL_NAMES: dict[str, str] = {
    "pau6imuk": "InternVL3.5-2B",
    "d4e8gwu0": "Qwen3-VL-2B",
    "cn28qd5a": "InternVL3.5-4B",
    "fdb89xu4": "Qwen3-VL-4B",
    "mx12190v": "InternVL3.5-8B",
    "p1r3exbe": "Qwen3-VL-8B",
    "w7jl4ly0": "Keye-VL-1.5-8B",
    "hektv801": "InternVL3.5-14B",
    "3ugpfhso": "InternVL3.5-30B-A3B",
    "f4imsgcv": "Qwen3-VL-30B-A3B",
    "toe74d9a": "Qwen3-VL-32B",
    "pkjbh92w": "InternVL3.5-38B",
}

MODEL_NAMES_COT: dict[str, str] = {
    "dts57kgz": "InternVL3.5-2B",
    "91g7t1y1": "Qwen3-VL-2B",
    "cpe2sto4": "InternVL3.5-8B",
    "fmmrnf5j": "Qwen3-VL-8B",
    "73ivqn3d": "Qwen3-VL-32B",
    "o8i8pojr": "InternVL3.5-38B",
}
USE_COT = True  # Set to True to use COT runs instead

DATASET = "OOPS"
SPLIT = "cs"

# Whether to include the Fall ∪ Fallen binary metrics column group
INCLUDE_FALL_UNION_FALLEN = True

# We define the specialized model data as a raw list of floats here
# so it can be included in the calculation for bold/underline.
SPECIALIZED_MODEL_NAME = "VMAE-K400"
SPECIALIZED_MODEL_METRICS_ALL: list[float | None] = [
    22.9,
    94.5,
    23.3,
    77.4,
    84.8,
    67.5,
    28.0,
    97.6,
    38.2,
    70.9,
    84.1,
    70.3,
]
SPECIALIZED_MODEL_METRICS: list[float | None] = (
    SPECIALIZED_MODEL_METRICS_ALL
    if INCLUDE_FALL_UNION_FALLEN
    else SPECIALIZED_MODEL_METRICS_ALL[:9]
)

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


def fetch_run_data(api, run_id):
    """Fetches summary metrics as raw floats."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary

        row_values = []
        for metric_key in METRICS_ORDER:
            val = summary.get(metric_key)
            if val is not None:
                # Store as float (multiplied by 100)
                row_values.append(val * 100)
            else:
                row_values.append(None)
        return row_values

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return [None] * len(METRICS_ORDER)


def format_value(val, col_index, stats):
    """Formats a value with bold/underline based on column stats, and heatmap for specific columns."""
    if val is None:
        return "--"

    val_rounded = round(val, 1)
    formatted_str = f"{val_rounded:.1f}"

    max_val = stats[col_index]["max"]
    second_val = stats[col_index]["second"]

    # Compare at full precision to break ties between values that display the same
    if val == max_val:
        formatted_str = f"\\textbf{{{formatted_str}}}"
    elif val == second_val:
        formatted_str = f"\\underline{{{formatted_str}}}"

    # Apply heatmap coloring for specific columns
    if col_index in HEATMAP_COLUMNS:
        min_val = stats[col_index]["min"]
        if max_val != min_val:
            level = int(round((val - min_val) / (max_val - min_val) * 100))
        else:
            level = 100
        formatted_str = f"\\gc{{{level}}}{{{formatted_str}}}"

    return formatted_str


def generate_latex():
    api = wandb.Api()

    # 1. Collect all data into a list of dictionaries
    all_rows = []

    # Add Specialized Model first
    all_rows.append(
        {
            "name": SPECIALIZED_MODEL_NAME,
            "metrics": SPECIALIZED_MODEL_METRICS,
            "type": "specialized",
        }
    )

    # Add WandB Models
    model_dict = MODEL_NAMES_COT if USE_COT else MODEL_NAMES
    for run_id, display_name in model_dict.items():
        metrics = fetch_run_data(api, run_id)
        all_rows.append({"name": display_name, "metrics": metrics, "type": "mllm"})

    # 2. Calculate Stats per column (Max and Second Max)
    # We loop through the number of metrics (0 to 8)
    num_metrics = len(METRICS_ORDER)
    col_stats = []

    for i in range(num_metrics):
        # Extract all valid values for this column from all models
        values = [row["metrics"][i] for row in all_rows if row["metrics"][i] is not None]

        # Get unique values sorted descending (full precision for accurate ranking)
        unique_vals = sorted(list(set(values)), reverse=True)

        stats = {
            "max": unique_vals[0] if len(unique_vals) > 0 else -1,
            "second": unique_vals[1] if len(unique_vals) > 1 else -1,
            "min": unique_vals[-1] if len(unique_vals) > 0 else 0,
        }
        col_stats.append(stats)

    # 3. Format Rows with Highlights
    specialized_latex = ""
    mllm_latex_rows = []

    for row in all_rows:
        formatted_metrics = []
        for i, val in enumerate(row["metrics"]):
            formatted_metrics.append(format_value(val, i, col_stats))

        metrics_str = " & ".join(formatted_metrics)
        latex_line = f"{row['name']} & {metrics_str} \\\\"

        if row["type"] == "specialized":
            specialized_latex = latex_line
        else:
            mllm_latex_rows.append(latex_line)

    mllm_body = "\n".join(mllm_latex_rows)

    # 4. Construct Final Table
    # Compute layout dimensions based on INCLUDE_FALL_UNION_FALLEN
    if INCLUDE_FALL_UNION_FALLEN:
        col_spec = "@{}l rrr rrr rrr rrr@{}"
        total_cols = 13
        union_header = " &\n\\multicolumn{3}{c}{Fall $\\cup$ Fallen}"
        union_cmidrule = " \\cmidrule(lr){11-13}"
        union_sub_header = (
            "\n & \\multicolumn{1}{c}{Se}   & \\multicolumn{1}{c}{Sp}  & \\multicolumn{1}{c}{F1}"
        )
    else:
        col_spec = "@{}l rrr rrr rrr@{}"
        total_cols = 10
        union_header = ""
        union_cmidrule = ""
        union_sub_header = ""

    full_table = f"""
\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
\caption{{\\textbf{{Zero-shot fall detection results}} on the OmniFall-In-the-Wild dataset.
We report classification metrics for the 16-class action recognition task, as well as binary metrics for the Fall, Fallen and combined Fall/Fallen classes. Open-source MLLMs are sorted by parameter count. The best results are highlighted in \\textbf{{bold}}, and the second-best are \\underline{{underlined}}. Darker cells indicate better performance.}}
\\label{{tab:zero_shot_fall_detection_results}}

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

% SECTION 1
\\multicolumn{{{total_cols}}}{{@{{}}l}}{{\\textit{{Specialized Model}}}} \\\\
{specialized_latex}
\\midrule

% SECTION 2
\\multicolumn{{{total_cols}}}{{@{{}}l}}{{\\textit{{Open-source MLLMs}}}} \\\\
{mllm_body}

\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
\\endgroup
"""

    print(full_table)


if __name__ == "__main__":
    generate_latex()
