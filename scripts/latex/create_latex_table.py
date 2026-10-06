"""Build the zero-shot results table on OmniFall-In-the-Wild.

Fetches metrics per run from W&B and prepends the hard-coded VMAE-K400 baseline.
Output: $PROJECT/master-thesis/tables/zeroshot_results.tex.
"""

import os
from pathlib import Path

import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-zeroshot-v4"

# Output path — set to "" to print to stdout only
OUTPUT_PATH = Path(os.path.expandvars("$PROJECT/master-thesis/tables/zeroshot_results.tex"))

# A section is (title, groups); a group is a list of (run_id, display_name) separated by
# \addlinespace. Models are sorted by parameter count, grouped by size.
Section = tuple[str, list[list[tuple[str, str]]]]

SECTIONS: list[Section] = [
    (
        "Open-source MLLMs",
        [
            [("pau6imuk", "InternVL3.5-2B"), ("d4e8gwu0", "Qwen3-VL-2B")],
            [("cn28qd5a", "InternVL3.5-4B"), ("fdb89xu4", "Qwen3-VL-4B")],
            [
                ("mx12190v", "InternVL3.5-8B"),
                ("p1r3exbe", "Qwen3-VL-8B"),
                ("w7jl4ly0", "Keye-VL-1.5-8B"),
            ],
            [("hektv801", "InternVL3.5-14B")],
            [("3ugpfhso", "InternVL3.5-30B-A3B"), ("f4imsgcv", "Qwen3-VL-30B-A3B")],
            [("toe74d9a", "Qwen3-VL-32B"), ("pkjbh92w", "InternVL3.5-38B")],
        ],
    ),
    (
        # New models plus vLLM 0.30 reruns of Qwen3-VL-8B / InternVL3.5-8B for comparison
        "Open-source MLLMs (vLLM 0.30)",
        [
            [("mlsxhlg2", "MiniCPM-V-4.6-1.3B")],
            # Cosmos3-Edge is a reasoner: 4096 max tokens instead of 64
            [("u30llsrc", "Cosmos3-Edge-4B"), ("6b4ewvx8", "Gemma-4-E4B")],
            [
                ("tf99x9u2", "InternVL3.5-8B"),
                ("q8y8l9n4", "Qwen3-VL-8B"),
                ("wfaqajs1", "LLaVA-OV-7B"),
                ("d0lrkne1", "LLaVA-OV-2-8B"),
                # Hybrid thinking models, run with enable_thinking=False
                ("i8754t01", "MiniCPM-V-4.5-8B"),
                ("koettehi", "Qwen3.5-9B"),
            ],
            [("p5x00wd9", "Gemma-4-12B")],
        ],
    ),
]

SECTIONS_COT: list[Section] = [
    (
        "Open-source MLLMs",
        [
            [("dts57kgz", "InternVL3.5-2B"), ("91g7t1y1", "Qwen3-VL-2B")],
            [("cpe2sto4", "InternVL3.5-8B"), ("fmmrnf5j", "Qwen3-VL-8B")],
            [("73ivqn3d", "Qwen3-VL-32B"), ("o8i8pojr", "InternVL3.5-38B")],
        ],
    ),
]
USE_COT = False  # Set to True to use COT runs instead

DATASET = "OOPS"
SPLIT = "cs"

# Whether to include the Fall ∪ Fallen binary metrics column group
INCLUDE_FALL_UNION_FALLEN = True

# We define the specialized model data as a raw list of floats here
# so it can be included in the calculation for bold/underline.
SPECIALIZED_MODEL_NAME = "VMAE-K400"
SPECIALIZED_MODEL_METRICS_ALL: list[float | None] = [
    21.4,
    47.6,
    21.9,
    72.9,
    85.4,
    65.6,
    33.1,
    96.3,
    41.0,
    68.2,
    82.4,
    67.5,
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
            level = int(round(10 + (val - min_val) / (max_val - min_val) * 90))
        else:
            level = 100
        formatted_str = f"\\gc{{{level}}}{{{formatted_str}}}"

    return formatted_str


def format_row(name, metrics, col_stats):
    metrics_str = " & ".join(format_value(val, i, col_stats) for i, val in enumerate(metrics))
    return f"{name} & {metrics_str} \\\\"


def generate_latex():
    api = wandb.Api()
    sections = SECTIONS_COT if USE_COT else SECTIONS

    # 1. Fetch metrics, keeping the section/group structure
    fetched = [
        (title, [[(name, fetch_run_data(api, run_id)) for run_id, name in group] for group in groups])
        for title, groups in sections
    ]
    all_metrics = [SPECIALIZED_MODEL_METRICS] + [
        metrics for _, groups in fetched for group in groups for _, metrics in group
    ]

    # 2. Calculate Stats per column (Max and Second Max)
    num_metrics = len(METRICS_ORDER)
    col_stats = []

    for i in range(num_metrics):
        # Extract all valid values for this column from all models
        values = [metrics[i] for metrics in all_metrics if metrics[i] is not None]

        # Get unique values sorted descending (full precision for accurate ranking)
        unique_vals = sorted(list(set(values)), reverse=True)

        stats = {
            "max": unique_vals[0] if len(unique_vals) > 0 else -1,
            "second": unique_vals[1] if len(unique_vals) > 1 else -1,
            "min": unique_vals[-1] if len(unique_vals) > 0 else 0,
        }
        col_stats.append(stats)

    # 3. Format Rows with Highlights
    specialized_latex = format_row(SPECIALIZED_MODEL_NAME, SPECIALIZED_MODEL_METRICS, col_stats)

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

    section_blocks = []
    for title, groups in fetched:
        group_blocks = [
            "\n".join(format_row(name, metrics, col_stats) for name, metrics in group)
            for group in groups
        ]
        body = "\n\\addlinespace\n".join(group_blocks)
        section_blocks.append(
            f"\\multicolumn{{{total_cols}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n{body}"
        )
    mllm_body = "\n\\midrule\n\n".join(section_blocks)

    # 4. Construct Final Table
    full_table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{table}}[htp]
\\caption[Zero-shot fall detection results on OF-ItW]{{\\textbf{{Zero-shot fall detection results on OmniFall-In-the-Wild.}}
We report classification metrics for the 16-class action recognition task, as well as binary metrics for the \\Fall, \\Fallen, and \\fallfallen subtasks. Open-source MLLMs are sorted by parameter count. The best results are highlighted in \\textbf{{bold}}, and the second-best are \\underline{{underlined}}. Darker cells indicate better performance. \\textbf{{B}}alanced \\textbf{{Acc}}uracy, \\textbf{{Se}}nsitivity, and \\textbf{{Sp}}ecificity}}
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

{mllm_body}

\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
\\endgroup
"""

    if OUTPUT_PATH:
        OUTPUT_PATH.write_text(full_table)
        print(f"Written to {OUTPUT_PATH}")
    else:
        print(full_table)


if __name__ == "__main__":
    generate_latex()
