import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-zeroshot-v4"

METRIC_SUFFIXES = [
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "fall_sensitivity",
    "fall_specificity",
    "fall_f1",
    "fallen_sensitivity",
    "fallen_specificity",
    "fallen_f1",
    "fall_union_fallen_sensitivity",
    "fall_union_fallen_specificity",
    "fall_union_fallen_f1",
]

# Each dataset entry: wb_name (W&B metric prefix), display_name (LaTeX),
# specialized_name (row label), specialized_metrics (12 floats as %, or None if unknown)
DATASETS = [
    {
        "wb_name": "le2i",
        "display_name": "Le2i",
        "cite_key": "le2i",
        "split_type": "cs",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            81.5,
            96.6,
            71.7,
            100.0,
            98.9,
            95.7,
            100.0,
            98.4,
            93.3,
            100.0,
            96.9,
            94.5,
        ],
    },
    {
        "wb_name": "up_fall",
        "display_name": "UP-Fall",
        "cite_key": "upfall",
        "split_type": "cs",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            83.5,
            99.6,
            76.5,
            98.2,
            97.1,
            94.1,
            86.0,
            96.6,
            86.7,
            92.1,
            91.2,
            90.5,
        ],
    },
    {
        "wb_name": "cmdfall",
        "display_name": "CMDFall",
        "cite_key": "cmdfall",
        "split_type": "cs",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            85.6,
            96.1,
            85.6,
            95.0,
            99.2,
            94.8,
            94.2,
            99.4,
            94.8,
            95.7,
            98.7,
            95.9,
        ],
    },
    {
        "wb_name": "gmdcsa24",
        "display_name": "GMDCSA24",
        "cite_key": "gmdcsa24",
        "split_type": "cs",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            80.2,
            87.1,
            74.0,
            100.0,
            92.1,
            85.0,
            76.5,
            100.0,
            86.7,
            88.2,
            89.8,
            85.7,
        ],
    },
    {
        "wb_name": "edf",
        "display_name": "EDF",
        "cite_key": "edf_occu",
        "split_type": "cs",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            64.6,
            93.8,
            68.7,
            87.5,
            95.5,
            80.0,
            95.0,
            96.3,
            88.4,
            97.2,
            92.4,
            89.7,
        ],
    },
    {
        "wb_name": "occu",
        "display_name": "OCCU",
        "cite_key": "edf_occu",
        "split_type": "cs",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            82.5,
            100.0,
            78.1,
            100.0,
            96.5,
            91.4,
            100.0,
            98.8,
            97.0,
            100.0,
            94.2,
            94.1,
        ],
    },
    {
        "wb_name": "caucafall",
        "display_name": "CaucaFall",
        "cite_key": "caucafall",
        "split_type": "cs",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            74.0,
            95.7,
            74.6,
            77.8,
            97.4,
            82.4,
            100.0,
            94.7,
            90.0,
            88.9,
            89.7,
            86.5,
        ],
    },
    {
        "wb_name": "wanfall",
        "display_name": "WanFall",
        "cite_key": "schneider_omnifall_2025",
        "split_type": "random",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [
            58.2,
            99.3,
            59.7,
            93.9,
            95.6,
            81.9,
            65.3,
            97.6,
            67.6,
            88.7,
            94.1,
            82.7,
        ],
    },
]

# Nested dict: model_display_name -> {dataset_wb_name -> run_id}
MLLM_RUN_IDS: dict[str, dict[str, str]] = {
    "InternVL3.5-8B": {
        "le2i": "rxsihd2u",
        "up_fall": "rocdaohb",
        "cmdfall": "5ia833zb",
        "gmdcsa24": "pyn6l2qa",
        "edf": "tnw4d5oq",
        "occu": "g4lgq7hg",
        "caucafall": "q4buift2",
        "wanfall": "v1byk2yi",
    },
    "Qwen3-VL-8B": {
        "le2i": "9uhyvciq",
        "up_fall": "8oj3dfnt",
        "cmdfall": "8fa8ojpb",
        "gmdcsa24": "lqhm4b5n",
        "edf": "0hgpttx8",
        "occu": "ipp0bhcr",
        "caucafall": "gpq1ipeo",
        "wanfall": "j61rz8mk",
    },
}


# ==========================================
# FETCH
# ==========================================
def fetch_run_metrics(
    api, run_id: str, dataset_wb_name: str, split_type: str
) -> list[float | None]:
    """Fetch the 12 metrics for a given run and dataset prefix."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary
        values: list[float | None] = []
        for suffix in METRIC_SUFFIXES:
            key = f"{dataset_wb_name}_{split_type}_{suffix}"
            val = summary.get(key)
            values.append(val * 100 if val is not None else None)
        return values
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return [None] * len(METRIC_SUFFIXES)


# ==========================================
# FORMATTING HELPERS
# ==========================================
def find_best_per_column(rows: list[list[float | None]]) -> list[float | None]:
    """Return the maximum value per column across the given rows."""
    num_cols = len(METRIC_SUFFIXES)
    best: list[float | None] = []
    for col in range(num_cols):
        vals = [row[col] for row in rows if row[col] is not None]
        best.append(max(vals) if vals else None)
    return best


def format_value(val: float | None, best_val: float | None) -> str:
    """Format a metric value, bolding if it equals the best in its group."""
    if val is None:
        return "--"
    val_rounded = round(val, 1)
    formatted = f"{val_rounded:.1f}"
    if best_val is not None and round(best_val, 1) == val_rounded:
        return f"\\textbf{{{formatted}}}"
    return formatted


# ==========================================
# LATEX GENERATION
# ==========================================
def generate_latex() -> None:
    api = wandb.Api()

    # Collect all row data grouped by dataset
    dataset_groups: list[dict] = []
    for ds in DATASETS:
        rows: list[dict] = []

        # Specialized model row (only if baseline metrics are known)
        if ds["specialized_metrics"] is not None:
            rows.append(
                {
                    "name": ds["specialized_name"],
                    "metrics": ds["specialized_metrics"],
                }
            )

        # MLLM rows
        for model_name, run_ids in MLLM_RUN_IDS.items():
            run_id = run_ids[ds["wb_name"]]
            if run_id == "TODO":
                metrics: list[float | None] = [None] * len(METRIC_SUFFIXES)
            else:
                metrics = fetch_run_metrics(api, run_id, ds["wb_name"], ds["split_type"])
            rows.append({"name": model_name, "metrics": metrics})

        dataset_groups.append(
            {
                "display_name": ds["display_name"],
                "cite_key": ds["cite_key"],
                "rows": rows,
            }
        )

    # Build LaTeX body
    body_lines: list[str] = []
    for group_idx, group in enumerate(dataset_groups):
        rows = group["rows"]
        metric_rows = [r["metrics"] for r in rows]
        best = find_best_per_column(metric_rows)

        display_name = group["display_name"]
        cite_key = group["cite_key"]

        for row_idx, row in enumerate(rows):
            # Dataset name cell: multirow on the first row of the group
            if row_idx == 0:
                if cite_key is not None:
                    ds_label = f"{display_name}~\\cite{{{cite_key}}}"
                else:
                    ds_label = display_name
                dataset_cell = f"\\multirow{{{len(rows)}}}{{*}}{{{ds_label}}}"
            else:
                dataset_cell = ""

            formatted = [format_value(v, best[i]) for i, v in enumerate(row["metrics"])]
            metrics_str = " & ".join(formatted)
            body_lines.append(f"        {dataset_cell} & {row['name']} & {metrics_str} \\\\")

        # Separator between groups (but not after the last one)
        if group_idx < len(dataset_groups) - 1:
            body_lines.append("        \\midrule")

    body = "\n".join(body_lines)

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
\\caption{{\\textbf{{Zero-shot fall detection on staged datasets.}}
Results for InternVL3.5-8B and Qwen3-VL-8B compared to a dataset-specific specialized
baseline on several staged fall detection benchmark datasets (cross-subject split) and a synthetic fall dataset WanFall (random split).
Best results per dataset are highlighted in \\textbf{{bold}}.}}
\\label{{tab:cross_dataset_results}}

\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{@{{}}l l rrr rrr rrr rrr@{{}}}}
\\toprule
% Top Header Row
\\multirow{{2}}{{*}}{{\\textbf{{Dataset}}}} &
\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} &
\\multicolumn{{3}}{{c}}{{16-class}} &
\\multicolumn{{3}}{{c}}{{Fall $\\Delta$}} &
\\multicolumn{{3}}{{c}}{{Fallen $\\Delta$}} &
\\multicolumn{{3}}{{c}}{{Fall $\\cup$ Fallen}} \\\\
\\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-8}} \\cmidrule(lr){{9-11}} \\cmidrule(lr){{12-14}}

% Sub Header Row
 & & \\multicolumn{{1}}{{c}}{{BAcc}} & \\multicolumn{{1}}{{c}}{{Acc}} & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}} \\\\
\\midrule

{body}

\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
\\endgroup
"""

    print(table)


def main() -> None:
    generate_latex()


if __name__ == "__main__":
    main()
