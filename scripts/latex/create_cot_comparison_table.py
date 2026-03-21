import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-zeroshot-v4"
DATASET = "OOPS"
SPLIT = "cs"

# Paired mapping: display name -> (zeroshot_run_id, cot_run_id)
# Sorted by model size (ascending)
MODEL_PAIRS: dict[str, tuple[str, str]] = {
    "Qwen3-VL-2B": ("d4e8gwu0", "91g7t1y1"),
    "InternVL3.5-2B": ("pau6imuk", "dts57kgz"),
    "Qwen3-VL-8B": ("p1r3exbe", "fmmrnf5j"),
    "InternVL3.5-8B": ("mx12190v", "cpe2sto4"),
    "Qwen3-VL-32B": ("toe74d9a", "73ivqn3d"),
    "InternVL3.5-38B": ("pkjbh92w", "o8i8pojr"),
}

# Global metrics — always shown (left section)
GLOBAL_METRICS: list[tuple[str, str]] = [
    (f"{DATASET}_{SPLIT}_balanced_accuracy", "BAcc"),
    (f"{DATASET}_{SPLIT}_accuracy", "Acc"),
    (f"{DATASET}_{SPLIT}_macro_f1", "Macro F1"),
]

# Per-class F1 — configurable (right multicolumn section)
# Keys follow pattern: {DATASET}_{SPLIT}_{class}_f1
PERCLASS_METRICS: list[tuple[str, str]] = [
    (f"{DATASET}_{SPLIT}_fall_f1", "Fall"),
    (f"{DATASET}_{SPLIT}_fallen_f1", "Fallen"),
    (f"{DATASET}_{SPLIT}_sit_down_f1", "Sit down"),
    # (f"{DATASET}_{SPLIT}_other_f1", "Other"),
    # (f"{DATASET}_{SPLIT}_walk_f1", "Walk"),
    # (f"{DATASET}_{SPLIT}_standing_f1", "Standing"),
    # (f"{DATASET}_{SPLIT}_stand_up_f1", "Stand up"),
    (f"{DATASET}_{SPLIT}_jump_f1", "Jump"),
]

ALL_METRICS = GLOBAL_METRICS + PERCLASS_METRICS


# ==========================================
# DATA FETCHING
# ==========================================
def fetch_run_data(api: wandb.Api, run_id: str) -> list[float | None]:
    """Fetches summary metrics as floats ×100."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary
        return [
            summary[key] * 100 if summary.get(key) is not None else None for key, _ in ALL_METRICS
        ]
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return [None] * len(ALL_METRICS)


# ==========================================
# FORMATTING
# ==========================================
def format_delta(delta: float) -> str:
    delta_r = round(delta, 1)
    if delta_r > 0:
        return r"\textcolor{ForestGreen}{(+" + f"{delta_r:.1f}" + r")}"
    elif delta_r < 0:
        return r"\textcolor{BrickRed}{(" + f"{delta_r:.1f}" + r")}"
    else:
        return r"\textcolor{gray}{(0.0)}"


def format_cot_val(cot_val: float | None, is_bold: bool, is_underline: bool) -> str:
    """Right-aligned CoT value cell."""
    if cot_val is None:
        return "--"
    s = f"{round(cot_val, 1):.1f}"
    if is_bold:
        return r"\textbf{" + s + r"}"
    if is_underline:
        return r"\underline{" + s + r"}"
    return s


def format_delta_col(cot_val: float | None, zs_val: float | None) -> str:
    """Left-aligned delta cell."""
    if cot_val is None or zs_val is None:
        return ""
    return format_delta(cot_val - zs_val)


# ==========================================
# TABLE GENERATION
# ==========================================
def generate_latex() -> None:
    api = wandb.Api()
    num_metrics = len(ALL_METRICS)
    num_global = len(GLOBAL_METRICS)
    num_perclass = len(PERCLASS_METRICS)

    # Fetch all data
    rows: list[dict] = []
    for name, (zs_id, cot_id) in MODEL_PAIRS.items():
        zs_vals = fetch_run_data(api, zs_id)
        cot_vals = fetch_run_data(api, cot_id)
        rows.append({"name": name, "zs": zs_vals, "cot": cot_vals})

    # Compute per-column stats on CoT values for bold/underline
    col_stats: list[dict] = []
    for i in range(num_metrics):
        vals = [r["cot"][i] for r in rows if r["cot"][i] is not None]
        unique = sorted(set(vals), reverse=True)
        col_stats.append(
            {
                "max": unique[0] if len(unique) > 0 else -1,
                "second": unique[1] if len(unique) > 1 else -1,
            }
        )

    # Format rows — each metric produces 2 cells: value (r) and delta (l)
    latex_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for i in range(num_metrics):
            cot_val = row["cot"][i]
            zs_val = row["zs"][i]
            stats = col_stats[i]
            is_bold = cot_val is not None and cot_val == stats["max"]
            is_underline = (
                cot_val is not None and cot_val != stats["max"] and cot_val == stats["second"]
            )
            cells.append(format_cot_val(cot_val, is_bold, is_underline))
            cells.append(format_delta_col(cot_val, zs_val))
        latex_rows.append(f"{row['name']} & " + " & ".join(cells) + r" \\")

    body = "\n".join(latex_rows)

    # Each metric occupies 2 columns (r for value, l for delta).
    # Column spec: model | global section (rl pairs) | perclass section (rl pairs)
    pair = r"r@{\,}l"
    col_spec = (
        "@{}l " + " ".join([pair] * num_global) + " | " + " ".join([pair] * num_perclass) + "@{}"
    )

    # cmidrule spans (1-indexed; model=col1, each metric takes 2 cols)
    global_start = 2
    global_end = 1 + 2 * num_global
    perclass_start = global_end + 1
    perclass_end = 1 + 2 * num_metrics

    # Header labels span 2 cols each via \multicolumn{2}{c}{...}
    global_labels = " & ".join(r"\multicolumn{2}{c}{" + lbl + "}" for _, lbl in GLOBAL_METRICS)
    perclass_labels = " & ".join(r"\multicolumn{2}{c}{" + lbl + "}" for _, lbl in PERCLASS_METRICS)

    table = f"""% Requires \\usepackage[dvipsnames]{{xcolor}} in preamble
\\begingroup
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{table}}[htp]
\\caption{{\\textbf{{Chain-of-Thought vs.\\ zero-shot results}} on the OF-ItW dataset.
Values show CoT performance with the difference to zero-shot in parentheses
(\\textcolor{{ForestGreen}}{{green}} = CoT better, \\textcolor{{BrickRed}}{{red}} = zero-shot better).
Best CoT results per column in \\textbf{{bold}}, second-best \\underline{{underlined}}.}}
\\label{{tab:cot_comparison}}

\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{{col_spec}}}
\\toprule
\\multirow{{2}}{{*}}{{Model}}
  & \\multicolumn{{{2 * num_global}}}{{c}}{{16-class}}
  & \\multicolumn{{{2 * num_perclass}}}{{c}}{{Per-class F1}} \\\\
\\cmidrule(lr){{{global_start}-{global_end}}} \\cmidrule(lr){{{perclass_start}-{perclass_end}}}
 & {global_labels} & {perclass_labels} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
\\endgroup
"""
    print(table)


if __name__ == "__main__":
    generate_latex()
