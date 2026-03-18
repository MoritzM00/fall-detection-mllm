"""Generate a LaTeX ablation table for few-shot exemplar selection + ordering.

Rows: Random (—), Balanced × {asc, desc}, Similarity × {asc, desc}.
Columns: BAcc, F1, Fall F1, Fallen F1 for each of two models side by side.
Fixed defaults: num_shots=5, delimiters=true.

Usage:
    python scripts/latex/create_fewshot_selection_table.py
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
    f"{DATASET}_{SPLIT}_macro_f1",
    f"{DATASET}_{SPLIT}_fall_f1",
    f"{DATASET}_{SPLIT}_fallen_f1",
]
METRIC_HEADERS = ["BAcc", "F1", "Fall F1", "Fallen F1"]

# Row definitions: (selection, ordering) — ordering is None for random
# This determines display order in the table.
ROW_KEYS: list[tuple[str, str | None]] = [
    ("random", None),
    ("balanced", "ascending"),
    ("balanced", "descending"),
    ("similarity", "ascending"),
    ("similarity", "descending"),
]

SELECTION_DISPLAY: dict[str, str] = {
    "random": "Random",
    "balanced": "Balanced",
    "similarity": "Similarity",
}

ORDER_DISPLAY: dict[str | None, str] = {
    None: "---",
    "ascending": "asc",
    "descending": "desc",
}

# How many rows each selection group spans (for \multirow)
SELECTION_SPAN: dict[str, int] = {
    "random": 1,
    "balanced": 2,
    "similarity": 2,
}

# Hardcoded run IDs: {run_id: (model_display_name, selection_strategy, ordering)}
# ordering is None for random (ordering irrelevant), "ascending"/"descending" otherwise.
# Fill in actual W&B run IDs before running.
SELECTION_RUNS: dict[str, tuple[str, str, str | None]] = {
    "TODO_internvl_random": ("InternVL3.5-8B", "random", None),
    "TODO_internvl_balanced_asc": ("InternVL3.5-8B", "balanced", "ascending"),
    "TODO_internvl_balanced_desc": ("InternVL3.5-8B", "balanced", "descending"),
    "szm3htq5": ("InternVL3.5-8B", "similarity", "ascending"),
    "TODO_internvl_similarity_desc": ("InternVL3.5-8B", "similarity", "descending"),
    "qezzftr0": ("Qwen3-VL-8B", "random", None),
    "0kl6yqw0": ("Qwen3-VL-8B", "balanced", "ascending"),
    "lsgd8lyo": ("Qwen3-VL-8B", "balanced", "descending"),
    "9ek95yp9": ("Qwen3-VL-8B", "similarity", "ascending"),
    "3xk84wk1": ("Qwen3-VL-8B", "similarity", "descending"),
}


# ============================================================================
# Data fetching
# ============================================================================


def fetch_run_metrics(api: wandb.Api, run_id: str) -> list[float | None]:
    """Fetch the 4 target metrics from a W&B run summary."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary
        return [val * 100 if (val := summary.get(m)) is not None else None for m in METRICS]
    except Exception as e:
        print(f"Warning: could not fetch run {run_id!r}: {e}")
        return [None] * len(METRICS)


def collect_data(
    api: wandb.Api,
) -> dict[tuple[str, str, str | None], list[float | None]]:
    """Return {(model, selection, ordering): [metric_values]} for all run IDs."""
    data: dict[tuple[str, str, str | None], list[float | None]] = {}
    for run_id, (model, selection, ordering) in SELECTION_RUNS.items():
        metrics = fetch_run_metrics(api, run_id)
        data[(model, selection, ordering)] = metrics
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
    data: dict[tuple[str, str, str | None], list[float | None]],
) -> list[list[float | None]]:
    """Compute per-column best values across all rows per model."""
    bests: list[list[float | None]] = []
    for model in MODELS:
        model_bests: list[float | None] = []
        for col_idx in range(len(METRICS)):
            values = [
                data[(model, sel, order)][col_idx]
                for (sel, order) in ROW_KEYS
                if (model, sel, order) in data and data[(model, sel, order)][col_idx] is not None
            ]
            model_bests.append(max(values) if values else None)
        bests.append(model_bests)
    return bests


# ============================================================================
# LaTeX generation
# ============================================================================


def generate_latex(data: dict[tuple[str, str, str | None], list[float | None]]) -> str:
    col_bests = compute_col_bests(data)

    rows: list[str] = []
    prev_selection: str | None = None

    for row_idx, (selection, ordering) in enumerate(ROW_KEYS):
        # Separate selection groups with whitespace instead of extra rules.
        if prev_selection is not None and selection != prev_selection:
            rows.append("        \\addlinespace[4pt]")

        span = SELECTION_SPAN[selection]
        order_str = ORDER_DISPLAY[ordering]

        # First row of a selection group: emit \multirow for the selection name
        if selection != prev_selection:
            sel_cell = f"\\multirow{{{span}}}{{*}}{{{SELECTION_DISPLAY[selection]}}}"
        else:
            sel_cell = ""

        cells: list[str] = [sel_cell, order_str]
        for model_idx, model in enumerate(MODELS):
            metrics = data.get((model, selection, ordering), [None] * len(METRICS))
            for col_idx, val in enumerate(metrics):
                cells.append(format_value(val, col_bests[model_idx][col_idx]))

        rows.append("        " + " & ".join(cells) + " \\\\")
        prev_selection = selection

    rows_str = "\n".join(rows)

    n_metric_cols = len(METRICS)
    # Column layout: Selection | Order | 4×InternVL | 4×Qwen → columns 3..6 and 7..10
    internvl_start = 3
    internvl_end = internvl_start + n_metric_cols - 1
    qwen_start = internvl_end + 1
    qwen_end = qwen_start + n_metric_cols - 1

    sub_header = " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in METRIC_HEADERS)

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{table}}[htp]
    \\centering
    \\caption{{\\textbf{{Few-Shot Selection \\& Ordering Ablation.}} Effect of exemplar selection strategy and ordering on fall detection performance with $k=5$. Best results per model are \\textbf{{bolded}}.}}
    \\label{{tab:fewshot_selection_ablation}}
    \\begin{{tabular}}{{@{{}}ll rrrr rrrr@{{}}}}
        \\toprule
        \\multirow{{2}}{{*}}{{\\textbf{{Method}}}} &
        \\multirow{{2}}{{*}}{{\\textbf{{Order}}}} &
        \\multicolumn{{{n_metric_cols}}}{{c}}{{\\textbf{{{MODELS[0]}}}}} &
        \\multicolumn{{{n_metric_cols}}}{{c}}{{\\textbf{{{MODELS[1]}}}}} \\\\
        \\cmidrule(lr){{{internvl_start}-{internvl_end}}} \\cmidrule(lr){{{qwen_start}-{qwen_end}}}
        & & {sub_header} & {sub_header} \\\\
        \\midrule

{rows_str}

        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
\\endgroup"""

    return table


def main() -> None:
    api = wandb.Api()
    print(f"Fetching few-shot selection ablation runs from {ENTITY}/{PROJECT}...")
    data = collect_data(api)
    print(f"Collected data for {len(data)} (model, selection, ordering) combinations.")
    print()
    print(generate_latex(data))


if __name__ == "__main__":
    main()
