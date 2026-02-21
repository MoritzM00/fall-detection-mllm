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
    "0fyrvcyv": "Qwen3-VL-2B",
    "2yzuuwqm": "Qwen3-VL-4B",
    "j5m9qihw": "Qwen3-VL-8B",
    "opzspcfj": "Qwen3-VL-30B-A3B",
    "ml3489zj": "Qwen3-VL-32B",
}
DATASET = "OOPS"
SPLIT = "cs"

# We define the specialized model data as a raw list of floats here
# so it can be included in the calculation for bold/underline
SPECIALIZED_MODEL_NAME = "VMAE-K400"
SPECIALIZED_MODEL_METRICS: list[float] = [21.4, 47.6, 21.9, 72.9, 85.4, 65.6, 33.1, 96.3, 41.0]

# Columns to apply heatmap coloring (indices: 0=BAcc, 2=F1)
HEATMAP_COLUMNS = [0, 2, 5, 8]

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
    for run_id, display_name in MODEL_NAMES.items():
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
    full_table = f"""
\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
\caption{{\\textbf{{Zero-shot fall detection results}} on the OF-ItW dataset using the cross-subject split.
We report classification metrics on the cross-subject (CS) split for the 16-class action recognition task, as well as binary metrics for the specific Fall and Fallen classes. The best results are highlighted in \\textbf{{bold}}, and the second-best are \\underline{{underlined}}.}}
\\label{{tab:zero_shot_fall_detection_results}}

\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{@{{}}l rrr rrr rrr@{{}}}}
\\toprule
% Top Header Row
\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} &
\\multicolumn{{3}}{{c}}{{16-class}} &
\\multicolumn{{3}}{{c}}{{Fall $\\Delta$}} &
\\multicolumn{{3}}{{c}}{{Fallen $\\Delta$}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}

% Sub Header Row
 & \\multicolumn{{1}}{{c}}{{BAcc}} & \\multicolumn{{1}}{{c}}{{Acc}} & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}} \\\\
\\midrule

% SECTION 1
\\multicolumn{{10}}{{@{{}}l}}{{\\textbf{{Specialized Model}}}} \\\\
{specialized_latex}
\\midrule

% SECTION 2
\\multicolumn{{10}}{{@{{}}l}}{{\\textbf{{Open-source MLLMs}}}} \\\\
{mllm_body}

\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
\\endgroup
"""

    print(full_table)


if __name__ == "__main__":
    generate_latex()
