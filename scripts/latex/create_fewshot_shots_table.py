"""Generate a LaTeX ablation table for few-shot number-of-shots sweep.

Rows: num_shots (1, 2, 3, 5, 7, 10) with similarity-based selection.
Columns: BAcc, F1, Fall F1, Fallen F1 for each of two models side by side.
Fixed defaults: delimiters=true, ordering=ascending.

Usage:
    python scripts/latex/create_fewshot_shots_table.py
"""

import wandb

# ============================================================================
# Configuration
# ============================================================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-fewshot"  # TODO: update to actual W&B project name

DATASET = "OOPS"
SPLIT = "cs"

MODELS = ["InternVL3.5-8B", "Qwen3-VL-8B"]
NUM_SHOTS = [1, 2, 3, 5, 7, 10]

METRICS = [
    f"{DATASET}_{SPLIT}_balanced_accuracy",
    f"{DATASET}_{SPLIT}_macro_f1",
    f"{DATASET}_{SPLIT}_fall_f1",
    f"{DATASET}_{SPLIT}_fallen_f1",
]
METRIC_HEADERS = ["BAcc", "F1", "Fall F1", "Fallen F1"]

# Hardcoded run IDs: {run_id: (model_display_name, num_shots)}
# Fill in actual W&B run IDs before running.
SHOTS_RUNS: dict[str, tuple[str, int]] = {
    "TODO_internvl_1": ("InternVL3.5-8B", 1),
    "TODO_internvl_2": ("InternVL3.5-8B", 2),
    "TODO_internvl_3": ("InternVL3.5-8B", 3),
    "TODO_internvl_5": ("InternVL3.5-8B", 5),
    "TODO_internvl_7": ("InternVL3.5-8B", 7),
    "TODO_internvl_10": ("InternVL3.5-8B", 10),
    "33okjfx5": ("Qwen3-VL-8B", 1),
    "nw3tonsj": ("Qwen3-VL-8B", 2),
    "xhwi2y8g": ("Qwen3-VL-8B", 3),
    "8x2iouxt": ("Qwen3-VL-8B", 5),
    "14v4fcb0": ("Qwen3-VL-8B", 7),
    "h60e7l5k": ("Qwen3-VL-8B", 10),
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
) -> dict[tuple[str, int], list[float | None]]:
    """Return {(model, num_shots): [metric_values]} for all run IDs."""
    data: dict[tuple[str, int], list[float | None]] = {}
    for run_id, (model, shots) in SHOTS_RUNS.items():
        metrics = fetch_run_metrics(api, run_id)
        data[(model, shots)] = metrics
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
    data: dict[tuple[str, int], list[float | None]],
) -> list[list[float | None]]:
    """Compute per-column best values.

    Returns a list of length len(MODELS), each containing len(METRICS) best values
    (best across all num_shots rows for that model/metric combination).
    """
    bests: list[list[float | None]] = []
    for model in MODELS:
        model_bests: list[float | None] = []
        for col_idx in range(len(METRICS)):
            values = [
                data[(model, shots)][col_idx]
                for shots in NUM_SHOTS
                if (model, shots) in data and data[(model, shots)][col_idx] is not None
            ]
            model_bests.append(max(values) if values else None)
        bests.append(model_bests)
    return bests


# ============================================================================
# LaTeX generation
# ============================================================================


def generate_latex(data: dict[tuple[str, int], list[float | None]]) -> str:
    col_bests = compute_col_bests(data)

    # Build table rows
    rows: list[str] = []
    for shots in NUM_SHOTS:
        cells: list[str] = [str(shots)]
        for model_idx, model in enumerate(MODELS):
            metrics = data.get((model, shots), [None] * len(METRICS))
            for col_idx, val in enumerate(metrics):
                cells.append(format_value(val, col_bests[model_idx][col_idx]))
        rows.append(" & ".join(cells) + " \\\\")

    rows_str = "\n".join(rows)

    n_metric_cols = len(METRICS)
    internvl_start = 2
    internvl_end = internvl_start + n_metric_cols - 1
    qwen_start = internvl_end + 1
    qwen_end = qwen_start + n_metric_cols - 1

    sub_header = " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in METRIC_HEADERS)

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
    \\centering
    \\begin{{tabular}}{{@{{}}l rrrr rrrr@{{}}}}
        \\toprule
        \\multirow{{2}}{{*}}{{\\textbf{{Shots}}}} &
        \\multicolumn{{{n_metric_cols}}}{{c}}{{\\textbf{{{MODELS[0]}}}}} &
        \\multicolumn{{{n_metric_cols}}}{{c}}{{\\textbf{{{MODELS[1]}}}}} \\\\
        \\cmidrule(lr){{{internvl_start}-{internvl_end}}} \\cmidrule(lr){{{qwen_start}-{qwen_end}}}
        & {sub_header} & {sub_header} \\\\
        \\midrule

{rows_str}

        \\bottomrule
    \\end{{tabular}}
    \\caption{{\\textbf{{Few-Shot Shots Ablation.}} Effect of the number of in-context exemplars on fall detection performance. Exemplars are selected by similarity (ascending order, with delimiters). Best results per model are \\textbf{{bolded}}.}}
    \\label{{tab:fewshot_shots_ablation}}
\\end{{table}}
\\endgroup"""

    return table


def main() -> None:
    api = wandb.Api()
    print(f"Fetching few-shot shots ablation runs from {ENTITY}/{PROJECT}...")
    data = collect_data(api)
    print(f"Collected data for {len(data)} (model, shots) combinations.")
    print()
    print(generate_latex(data))


if __name__ == "__main__":
    main()
