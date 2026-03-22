"""Generate a LaTeX ablation table for few-shot prompt format.

Rows: Preamble × Response style (system/user × inline/assistant), 4 combinations.
Columns: Acc, BAcc, F1, Fall F1, Fallen F1 for each of two models side by side.
Fixed defaults: num_shots=5.

Usage:
    python scripts/latex/create_fewshot_format_table.py
"""

import wandb

# ============================================================================
# Configuration
# ============================================================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-fewshot"

DATASET = "OOPS"
SPLIT = "cs"

MODELS = ["InternVL3.5-8B", "Qwen3-VL-8B"]

METRICS = [
    f"{DATASET}_{SPLIT}_balanced_accuracy",
    f"{DATASET}_{SPLIT}_accuracy",
    f"{DATASET}_{SPLIT}_macro_f1",
]
METRIC_HEADERS = ["BAcc", "Acc", "Macro F1"]

# Row definitions: (preamble, response) in display order
ROW_KEYS: list[tuple[str, str]] = [
    ("system", "inline"),
    ("system", "assistant"),
    ("user", "inline"),
    ("user", "assistant"),
]

PREAMBLE_DISPLAY: dict[str, str] = {
    "system": "system",
    "user": "user",
}

RESPONSE_DISPLAY: dict[str, str] = {
    "inline": "inline",
    "assistant": "assistant",
}

# How many rows each preamble group spans (for \multirow)
PREAMBLE_SPAN: dict[str, int] = {
    "system": 2,
    "user": 2,
}

# Hardcoded run IDs: {run_id: (model_display_name, preamble, response)}
FORMAT_RUNS: dict[str, tuple[str, str, str]] = {
    "t7lmd4i9": ("InternVL3.5-8B", "system", "inline"),
    "ooexmzo9": ("InternVL3.5-8B", "system", "assistant"),
    "4vcbig26": ("InternVL3.5-8B", "user", "inline"),
    "zhnibwc7": ("InternVL3.5-8B", "user", "assistant"),
    "ctbq17zz": ("Qwen3-VL-8B", "system", "inline"),
    "mrkm94vi": ("Qwen3-VL-8B", "system", "assistant"),
    "vi63fjd0": ("Qwen3-VL-8B", "user", "inline"),
    "x7kkp8ms": ("Qwen3-VL-8B", "user", "assistant"),
}


# ============================================================================
# Data fetching
# ============================================================================


def fetch_run_metrics(api: wandb.Api, run_id: str) -> list[float | None]:
    """Fetch the target metrics from a W&B run summary."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary
        return [val * 100 if (val := summary.get(m)) is not None else None for m in METRICS]
    except Exception as e:
        print(f"Warning: could not fetch run {run_id!r}: {e}")
        return [None] * len(METRICS)


def collect_data(api: wandb.Api) -> dict[tuple[str, str, str], list[float | None]]:
    """Return {(model, preamble, response): [metric_values]} for all run IDs."""
    data: dict[tuple[str, str, str], list[float | None]] = {}
    for run_id, (model, preamble, response) in FORMAT_RUNS.items():
        metrics = fetch_run_metrics(api, run_id)
        data[(model, preamble, response)] = metrics
    return data


# ============================================================================
# Formatting
# ============================================================================


def format_value(val: float | None, best_val: float | None) -> str:
    """Format metric value, bolding if it equals the column best."""
    if val is None:
        return "--"
    rounded = round(val, 1)
    formatted = f"{rounded:.1f}"
    if best_val is not None and round(best_val, 1) == rounded:
        return f"\\textbf{{{formatted}}}"
    return formatted


def compute_col_bests(
    data: dict[tuple[str, str, str], list[float | None]],
) -> list[list[float | None]]:
    """Compute per-column best values across all rows per model."""
    bests: list[list[float | None]] = []
    for model in MODELS:
        model_bests: list[float | None] = []
        for col_idx in range(len(METRICS)):
            values = [
                data[(model, preamble, response)][col_idx]
                for (preamble, response) in ROW_KEYS
                if (model, preamble, response) in data
                and data[(model, preamble, response)][col_idx] is not None
            ]
            model_bests.append(max(values) if values else None)
        bests.append(model_bests)
    return bests


# ============================================================================
# LaTeX generation
# ============================================================================


def generate_latex(data: dict[tuple[str, str, str], list[float | None]]) -> str:
    col_bests = compute_col_bests(data)

    rows: list[str] = []
    prev_preamble: str | None = None

    for preamble, response in ROW_KEYS:
        if prev_preamble is not None and preamble != prev_preamble:
            rows.append("        \\addlinespace[4pt]")

        span = PREAMBLE_SPAN[preamble]
        response_str = RESPONSE_DISPLAY[response]

        if preamble != prev_preamble:
            preamble_cell = f"\\multirow{{{span}}}{{*}}{{{PREAMBLE_DISPLAY[preamble]}}}"
        else:
            preamble_cell = ""

        cells: list[str] = [preamble_cell, response_str]
        for model_idx, model in enumerate(MODELS):
            metrics = data.get((model, preamble, response), [None] * len(METRICS))
            for col_idx, val in enumerate(metrics):
                cells.append(format_value(val, col_bests[model_idx][col_idx]))

        rows.append("        " + " & ".join(cells) + " \\\\")
        prev_preamble = preamble

    rows_str = "\n".join(rows)

    n_metric_cols = len(METRICS)
    # Column layout: Introduction | Response | 3×InternVL | 3×Qwen → cols 3..5 and 6..8
    internvl_start = 3
    internvl_end = internvl_start + n_metric_cols - 1
    qwen_start = internvl_end + 1
    qwen_end = qwen_start + n_metric_cols - 1

    sub_header = " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in METRIC_HEADERS)

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{table}}[htp]
    \\centering
    \\caption{{\\textbf{{Few-Shot Prompt Format Ablation.}} Effect of preamble placement
      and response style on fall detection performance with $k=5$. Best results per
      model are \\textbf{{bolded}}.}}
    \\label{{tab:fewshot_format_ablation}}
    \\resizebox{{\\linewidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}ll rrr rrr@{{}}}}
        \\toprule
        \\multirow{{2}}{{*}}{{\\textbf{{Introduction}}}} &
        \\multirow{{2}}{{*}}{{\\textbf{{Response}}}} &
        \\multicolumn{{{n_metric_cols}}}{{c}}{{\\textbf{{{MODELS[0]}}}}} &
        \\multicolumn{{{n_metric_cols}}}{{c}}{{\\textbf{{{MODELS[1]}}}}} \\\\
        \\cmidrule(lr){{{internvl_start}-{internvl_end}}} \\cmidrule(lr){{{qwen_start}-{qwen_end}}}
        & & {sub_header} & {sub_header} \\\\
        \\midrule

{rows_str}

        \\bottomrule
    \\end{{tabular}}}}
\\end{{table}}
\\endgroup"""

    return table


def main() -> None:
    api = wandb.Api()
    print(f"Fetching few-shot format ablation runs from {ENTITY}/{PROJECT}...")
    data = collect_data(api)
    print(f"Collected data for {len(data)} (model, preamble, response) combinations.")
    print()
    print(generate_latex(data))


if __name__ == "__main__":
    main()
