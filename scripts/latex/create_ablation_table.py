import logging

import wandb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "prompt-ablations"

DATASET = "OOPS"
SPLIT = "cs"

# Tag-based filtering for ablation runs
ABLATION_TAG = "ablation"
MODEL_TAGS = {
    "internvl": "internvl",
    "qwen": "qwen",
}

# Limit to N most recent runs per model
LATEST_N = 8

# Display names for the table header
MODEL_DISPLAY_NAMES = {
    "internvl": "InternVL3.5",
    "qwen": "Qwen3-VL",
}

# Metrics to extract
METRICS = [
    f"{DATASET}_{SPLIT}_balanced_accuracy",
    f"{DATASET}_{SPLIT}_macro_f1",
]

# Row ordering: (output_format, include_role, include_definitions)
ROW_ORDER = [
    ("text", False, False),
    ("text", True, False),
    ("text", False, True),
    ("text", True, True),
    ("json", False, False),
    ("json", True, False),
    ("json", False, True),
    ("json", True, True),
]


def fetch_ablation_runs(api, model_tag):
    """
    Fetch ablation runs for a specific model from wandb.

    Returns a dict keyed by (output_format, include_role, include_definitions)
    with values being dicts containing 'bacc' and 'f1' metrics.
    """
    query_kwargs = {
        "filters": {
            "$and": [
                {"tags": ABLATION_TAG},
                {"tags": "prompt"},
                {"tags": model_tag},
            ]
        },
        "order": "-created_at",  # newest first
    }

    runs = api.runs(f"{ENTITY}/{PROJECT}", **query_kwargs)

    # latest n runs

    results = {}
    for run in runs:
        logger.info(f"Processing run: {run.name} ({run.id})")
        config = run.config
        prompt_config = config.get("prompt", {})

        output_format = prompt_config.get("output_format", "text")
        include_role = prompt_config.get("role_variant")
        include_definitions = prompt_config.get("definitions_variant")
        if include_role is None:
            include_role = False
        else:
            include_role = True
        if include_definitions is None:
            include_definitions = False
        else:
            include_definitions = True

        key = (output_format, include_role, include_definitions)

        summary = run.summary
        bacc = summary.get(METRICS[0])
        f1 = summary.get(METRICS[1])

        if bacc is not None:
            bacc = bacc * 100
        if f1 is not None:
            f1 = f1 * 100

        results[key] = {"bacc": bacc, "f1": f1}
        if len(results) >= LATEST_N:
            break

    return results


def find_best_in_block(data, block_keys):
    """Find best BAcc and F1 values within a block of rows."""
    bacc_values = [data.get(k, {}).get("bacc") for k in block_keys]
    f1_values = [data.get(k, {}).get("f1") for k in block_keys]

    bacc_values = [v for v in bacc_values if v is not None]
    f1_values = [v for v in f1_values if v is not None]

    best_bacc = max(bacc_values) if bacc_values else None
    best_f1 = max(f1_values) if f1_values else None

    return best_bacc, best_f1


def format_value(val, best_val):
    """Format a metric value, bolding if it's the best."""
    if val is None:
        return "--"

    val_rounded = round(val, 1)
    formatted = f"{val_rounded:.1f}"

    if best_val is not None and round(best_val, 1) == val_rounded:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_latex_table(internvl_data, qwen_data):
    """Generate the LaTeX table comparing both models."""
    # Define blocks
    text_block = ROW_ORDER[:4]
    json_block = ROW_ORDER[4:]

    # Find best values per block per model
    internvl_text_best = find_best_in_block(internvl_data, text_block)
    internvl_json_best = find_best_in_block(internvl_data, json_block)
    qwen_text_best = find_best_in_block(qwen_data, text_block)
    qwen_json_best = find_best_in_block(qwen_data, json_block)

    def get_best_for_key(key):
        """Get the best values for a given config key."""
        if key[0] == "text":
            return internvl_text_best, qwen_text_best
        else:
            return internvl_json_best, qwen_json_best

    # Generate rows
    rows = []
    for i, key in enumerate(ROW_ORDER):
        output_format, include_role, include_definitions = key

        # Multirow for first row of each block
        if i == 0:
            struct_col = "\\multirow{4}{*}{No (Text)}"
        elif i == 4:
            struct_col = "\\multirow{4}{*}{Yes (JSON)}"
        else:
            struct_col = ""

        role_mark = "\\cmark" if include_role else "\\xmark"
        def_mark = "\\cmark" if include_definitions else "\\xmark"

        internvl_best, qwen_best = get_best_for_key(key)

        internvl_metrics = internvl_data.get(key, {})
        qwen_metrics = qwen_data.get(key, {})

        internvl_bacc = format_value(internvl_metrics.get("bacc"), internvl_best[0])
        internvl_f1 = format_value(internvl_metrics.get("f1"), internvl_best[1])
        qwen_bacc = format_value(qwen_metrics.get("bacc"), qwen_best[0])
        qwen_f1 = format_value(qwen_metrics.get("f1"), qwen_best[1])

        row = f"        {struct_col} & {role_mark} & {def_mark} & {internvl_bacc} & {internvl_f1} & {qwen_bacc} & {qwen_f1} \\\\"
        rows.append(row)

        # Add midrule after text block
        if i == 3:
            rows.append("        \\midrule")

    rows_str = "\n".join(rows)

    table = f"""\\begin{{table}}[htp]
    \\centering
    \\begin{{tabular}}{{@{{}}c c c rr rr@{{}}}}
        \\toprule
        \\multirow{{2}}{{*}}{{\\textbf{{Structured Output}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Components}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{{MODEL_DISPLAY_NAMES["internvl"]}}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{{MODEL_DISPLAY_NAMES["qwen"]}}}}} \\\\
        \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
        & \\textbf{{Role}} & \\textbf{{Def.}} & \\textbf{{BAcc}} & \\textbf{{F1}} & \\textbf{{BAcc}} & \\textbf{{F1}} \\\\
        \\midrule

{rows_str}

        \\bottomrule
    \\end{{tabular}}
    \\caption{{\\textbf{{Ablation Study Results.}} We analyze the effect of role prompting, definitions, and structured output constraints. ``Structured Output'' refers to enforcing JSON formatting (Yes) versus free-text generation (No). Best results grouped by structured output are \\textbf{{bolded}}.}}
    \\label{{tab:zeroshot_prompt_ablation}}
\\end{{table}}"""

    return table


def main():
    api = wandb.Api()

    print(f"Fetching ablation runs from {ENTITY}/{PROJECT}...")

    internvl_data = fetch_ablation_runs(api, MODEL_TAGS["internvl"])
    qwen_data = fetch_ablation_runs(api, MODEL_TAGS["qwen"])

    print(f"Found {len(internvl_data)} InternVL runs")
    print(f"Found {len(qwen_data)} Qwen runs")

    if len(internvl_data) != 8 or len(qwen_data) != 8:
        print("Warning: Expected 8 runs per model for full ablation grid")

    latex_table = generate_latex_table(internvl_data, qwen_data)
    print("\n" + latex_table)


if __name__ == "__main__":
    main()
