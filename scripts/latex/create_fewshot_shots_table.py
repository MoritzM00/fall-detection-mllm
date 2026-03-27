"""Generate a LaTeX ablation table for few-shot number-of-shots sweep.

Rows: num_shots (1, 2, 3, 5, 7, 10) with similarity-based selection, stacked per model.
Columns: Avg Input Tokens, Inference Time, BAcc, Acc, F1, Fall F1, Fallen F1.
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
NUM_SHOTS = [1, 2, 3, 5, 7, 10, 15]

# InternVL3.5-8B max model length — rows with this combination are omitted from the table.
# The caption notes this exclusion.
INTERNVL_MAX_MODEL_LEN = 40960

# (model, num_shots) combinations that exceed the model's max context length.
# These rows are skipped entirely; the caption explains the exclusion.
EXCEEDS_CONTEXT: set[tuple[str, int]] = {
    ("InternVL3.5-8B", shots) for shots in NUM_SHOTS if shots >= 10
}

METRICS = [
    f"{DATASET}_{SPLIT}_balanced_accuracy",
    f"{DATASET}_{SPLIT}_accuracy",
    f"{DATASET}_{SPLIT}_macro_f1",
    f"{DATASET}_{SPLIT}_fall_f1",
    f"{DATASET}_{SPLIT}_fallen_f1",
]
METRIC_HEADERS = ["BAcc", "Acc", "F1", "Fall F1", "Fallen F1"]

# Hardcoded run IDs: {run_id: (model_display_name, num_shots)}
# Fill in actual W&B run IDs before running.
SHOTS_RUNS: dict[str, tuple[str, int]] = {
    "TODO_internvl_1": ("InternVL3.5-8B", 1),
    "TODO_internvl_2": ("InternVL3.5-8B", 2),
    "TODO_internvl_3": ("InternVL3.5-8B", 3),
    "TODO_internvl_5": ("InternVL3.5-8B", 5),
    "TODO_internvl_7": ("InternVL3.5-8B", 7),
    "lz8e33if": ("Qwen3-VL-8B", 1),
    "h405omjo": ("Qwen3-VL-8B", 2),
    "xisxlsxd": ("Qwen3-VL-8B", 3),
    "iy7xabko": ("Qwen3-VL-8B", 5),
    "zy6qp8ph": ("Qwen3-VL-8B", 7),
    "TODO_qwen_10": ("Qwen3-VL-8B", 10),
    "TODO_qwen_15": ("Qwen3-VL-8B", 15),
}


# ============================================================================
# Data fetching
# ============================================================================


def fetch_run_metrics(
    api: wandb.Api, run_id: str
) -> tuple[list[float | None], int | None, float | None]:
    """Fetch target metrics, avg input tokens, and inference time from a W&B run summary.

    Returns:
        (metric_values, avg_input_tokens, inference_time_seconds)
    """
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary
        metrics = [val * 100 if (val := summary.get(m)) is not None else None for m in METRICS]
        avg_tokens_raw = summary.get("avg_input_tokens_per_request")
        avg_tokens = int(avg_tokens_raw) if avg_tokens_raw is not None else None
        inference_time_raw = summary.get("inference_time_seconds")
        inference_time = float(inference_time_raw) if inference_time_raw is not None else None
        return metrics, avg_tokens, inference_time
    except Exception as e:
        print(f"Warning: could not fetch run {run_id!r}: {e}")
        return [None] * len(METRICS), None, None


def collect_data(
    api: wandb.Api,
) -> dict[tuple[str, int], tuple[list[float | None], int | None, float | None]]:
    """Return {(model, num_shots): (metric_values, avg_tokens, inference_time)} for all runs."""
    data: dict[tuple[str, int], tuple[list[float | None], int | None, float | None]] = {}
    for run_id, (model, shots) in SHOTS_RUNS.items():
        metrics, avg_tokens, inference_time = fetch_run_metrics(api, run_id)
        data[(model, shots)] = (metrics, avg_tokens, inference_time)
    return data


# ============================================================================
# Formatting
# ============================================================================


def format_metric(val: float | None, best_val: float | None, second_val: float | None) -> str:
    """Format metric value, bolding the column best and underlining the second-best."""
    if val is None:
        return "--"
    rounded = round(val, 1)
    formatted = f"{rounded:.1f}"
    if best_val is not None and round(best_val, 1) == rounded:
        return f"\\textbf{{{formatted}}}"
    if second_val is not None and round(second_val, 1) == rounded:
        return f"\\underline{{{formatted}}}"
    return formatted


def format_tokens(tokens: int | None) -> str:
    """Format avg input token count rounded to nearest 100."""
    if tokens is None:
        return "--"
    rounded = round(tokens / 100) * 100
    return f"{rounded:,}"


def format_time(seconds: float | None) -> str:
    """Format inference time rounded to the nearest minute."""
    if seconds is None:
        return "--"
    minutes = round(seconds / 60)
    if minutes == 0:
        minutes = 1  # show at least 1m
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h~{mins}m"
    return f"{minutes}m"


def compute_col_bests(
    data: dict[tuple[str, int], tuple[list[float | None], int | None, float | None]],
) -> dict[str, tuple[list[float | None], list[float | None]]]:
    """Compute per-column best and second-best metric values per model.

    Returns {model: (bests, seconds)} where each is a list indexed by metric column.
    """
    result: dict[str, tuple[list[float | None], list[float | None]]] = {}
    for model in MODELS:
        bests: list[float | None] = []
        seconds: list[float | None] = []
        for col_idx in range(len(METRICS)):
            values = sorted(
                (
                    data[(model, shots)][0][col_idx]
                    for shots in NUM_SHOTS
                    if (model, shots) in data
                    and data[(model, shots)][0][col_idx] is not None
                    and (model, shots) not in EXCEEDS_CONTEXT
                ),
                reverse=True,
            )
            bests.append(values[0] if values else None)
            seconds.append(values[1] if len(values) > 1 else None)
        result[model] = (bests, seconds)
    return result


# ============================================================================
# LaTeX generation
# ============================================================================


def generate_latex(
    data: dict[tuple[str, int], tuple[list[float | None], int | None, float | None]],
) -> str:
    col_bests = compute_col_bests(data)
    n_cols_total = 1 + len(METRICS) + 2  # shots + metrics + tokens + time

    sub_header = "\\multicolumn{1}{c}{Tokens} & \\multicolumn{1}{c}{Time} & " + " & ".join(
        f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in METRIC_HEADERS
    )

    rows: list[str] = []

    for i, model in enumerate(MODELS):
        midrule = "" if i == 0 else "        \\midrule\n"
        rows.append(
            f"{midrule}"
            f"        \\addlinespace[3pt]\n"
            f"        \\multicolumn{{{n_cols_total}}}{{l}}{{\\textit{{{model}}}}} \\\\"
        )
        bests, seconds = col_bests[model]
        for shots in NUM_SHOTS:
            if (model, shots) in EXCEEDS_CONTEXT:
                # Skip rows that exceed context — the caption explains the exclusion.
                continue
            metrics, avg_tokens, inference_time = data.get(
                (model, shots), ([None] * len(METRICS), None, None)
            )
            metric_cells = [
                format_metric(metrics[i], bests[i], seconds[i]) for i in range(len(METRICS))
            ]
            token_cell = format_tokens(avg_tokens)
            time_cell = format_time(inference_time)
            cells = [str(shots), token_cell, time_cell] + metric_cells
            rows.append("        " + " & ".join(cells) + " \\\\")

    rows_str = "\n".join(rows)

    col_spec = "@{}l r r " + "r" * len(METRICS) + "@{}"

    context_note = (
        f" InternVL3.5-8B is excluded for $k \\geq 10$ as it exceeds"
        f" the model's maximum context length ({INTERNVL_MAX_MODEL_LEN:,}~tokens)."
    )

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
    \\centering
    \\caption{{\\textbf{{Few-Shot Shots Ablation.}} Effect of the number of in-context exemplars on fall detection performance. Exemplars are selected by similarity (ascending order, with delimiters). Tokens denotes the number of input tokens per request. Best metric results per model are \\textbf{{bolded}}; second-best are \\underline{{underlined}}.{context_note}}}
    \\label{{tab:fewshot_shots_ablation}}
    \\begin{{tabular}}{{{col_spec}}}
        \\toprule
        $k$ & {sub_header} \\\\
        \\midrule

{rows_str}

        \\bottomrule
    \\end{{tabular}}
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
