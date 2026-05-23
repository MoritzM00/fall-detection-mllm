"""Generate the SFT data-mix main results LaTeX table and write it to the thesis tables directory.

Fetches evaluation metrics from 12 W&B inference runs (3 eval blocks × 4 training mixes)
and produces a table with eval blocks CMDFall / Syn / ItW and training mixes
ItW / Sta / Syn / All (Sta+Syn+ItW).

Each inference run is cross-checked against the declared training run id (parsed from the
lora.path config field) to catch ID mixups early.

Usage:
    python scripts/latex/create_sft_data_mix_table.py
    python scripts/latex/create_sft_data_mix_table.py --output path/to/sft_lora_data_mix.tex
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)

ENTITY = "moritzm00"
INFERENCE_PROJECT = "fall-detection-zeroshot-v4"
TRAIN_PROJECT = "falldet-mllm-finetune"

# Ordered: how mix rows appear top-to-bottom in each eval block
MIXES: list[str] = ["Sta", "Syn", "ItW", "All"]

MIX_DISPLAY: dict[str, str] = {
    "ItW": "ItW",
    "Sta": "Sta",
    "Syn": "Syn",
    "All": "Sta{+}Syn{+}ItW",
}

# Declared training run IDs per mix — fill these in before running
TRAIN_RUN_IDS: dict[str, str] = {
    "ItW": "pjmzzx8i",
    "Sta": "ya7dqqyu",
    "Syn": "sletu100",
    "All": "77r47xiw",
}

# Eval blocks: display label, W&B metric dataset prefix, split key
EVAL_BLOCKS: list[dict] = [
    {"key": "Sta", "label": "CMDFall", "wb_name": "cmdfall", "split": "cs"},
    {"key": "Syn", "label": "Syn", "wb_name": "wanfall", "split": "random"},
    {"key": "ItW", "label": "ItW", "wb_name": "OOPS", "split": "cs"},
]

# Inference run IDs: eval_block_key -> mix -> run_id (12 total — fill these in)
EVAL_RUN_IDS: dict[str, dict[str, str]] = {
    "Sta": {"ItW": "siyw9wfh", "Sta": "0t0khm9o", "Syn": "8i92so3u", "All": "z0uo30e7"},
    "Syn": {"ItW": "t3w0dwg5", "Sta": "ezvmeln3", "Syn": "7p6l371w", "All": "9tio4kk2"},
    "ItW": {"ItW": "a5nz8f5v", "Sta": "94ehgtjb", "Syn": "zu1yjotn", "All": "2965he1k"},
}

METRIC_SUFFIXES: list[tuple[str, str]] = [
    ("bacc", "balanced_accuracy"),
    ("acc", "accuracy"),
    ("macro_f1", "macro_f1"),
    ("fall_sensitivity", "fall_sensitivity"),
    ("fall_specificity", "fall_specificity"),
    ("fall_f1", "fall_f1"),
    ("fallen_sensitivity", "fallen_sensitivity"),
    ("fallen_specificity", "fallen_specificity"),
    ("fallen_f1", "fallen_f1"),
]

METRIC_KEYS: list[str] = [name for name, _ in METRIC_SUFFIXES]

DEFAULT_OUTPUT = Path("~/thesis-overleaf/tables/sft_lora_data_mix.tex").expanduser()


def _resolve_train_run_id(run: wandb.apis.public.Run) -> str | None:
    lora_path = (run.config.get("lora") or {}).get("path", "")
    train_run_name = Path(lora_path).parent.name
    if not train_run_name:
        return None
    return train_run_name.rsplit("_", 1)[-1]


def fetch_block(
    api: wandb.Api,
    block: dict,
    mix: str,
    run_id: str,
) -> dict:
    wb_prefix = f"{block['wb_name']}_{block['split']}_"
    record: dict = {"block": block["key"], "mix": mix}

    if run_id == "TODO":
        logger.warning("  run_id is TODO for block=%s mix=%s — skipping", block["key"], mix)
        for name in METRIC_KEYS:
            record[name] = None
        return record

    logger.info("Fetching run %s (block=%s mix=%s)", run_id, block["key"], mix)
    run = api.run(f"{ENTITY}/{INFERENCE_PROJECT}/{run_id}")

    resolved_train_id = _resolve_train_run_id(run)
    declared_train_id = TRAIN_RUN_IDS[mix]

    if declared_train_id != "TODO":
        if resolved_train_id != declared_train_id:
            raise ValueError(
                f"Train run ID mismatch for block={block['key']} mix={mix}: "
                f"declared={declared_train_id!r} but lora.path resolved to {resolved_train_id!r}. "
                f"Check TRAIN_RUN_IDS and EVAL_RUN_IDS."
            )
        logger.info("  Train ID verified: %s", resolved_train_id)
    else:
        logger.info(
            "  Train ID (from lora.path): %s (TRAIN_RUN_IDS not set, skipping check)",
            resolved_train_id,
        )

    for name, suffix in METRIC_SUFFIXES:
        key = f"{wb_prefix}{suffix}"
        val = run.summary.get(key)
        record[name] = float(val) * 100.0 if val is not None else None

    return record


def fetch_all(api: wandb.Api) -> dict[str, list[dict]]:
    """Returns {block_key: [record_per_mix, ...]} in MIXES order."""
    results: dict[str, list[dict]] = {}
    for block in EVAL_BLOCKS:
        block_records: list[dict] = []
        for mix in MIXES:
            run_id = EVAL_RUN_IDS[block["key"]][mix]
            record = fetch_block(api, block, mix, run_id)
            block_records.append(record)
        results[block["key"]] = block_records
    return results


GRADIENT_METRICS: frozenset[str] = frozenset({"bacc", "macro_f1", "fall_f1", "fallen_f1"})


def _block_bests(records: list[dict]) -> dict[str, float | None]:
    bests: dict[str, float | None] = {}
    for name in METRIC_KEYS:
        vals = [r[name] for r in records if r[name] is not None]
        bests[name] = max(vals) if vals else None
    return bests


def _block_ranges(records: list[dict]) -> dict[str, tuple[float, float] | None]:
    ranges: dict[str, tuple[float, float] | None] = {}
    for name in METRIC_KEYS:
        vals = [r[name] for r in records if r[name] is not None]
        ranges[name] = (min(vals), max(vals)) if len(vals) >= 2 else None
    return ranges


def _opacity(val: float, min_val: float, max_val: float) -> int:
    if max_val == min_val:
        return 55
    return max(10, min(100, round((val - min_val) / (max_val - min_val) * 90 + 10)))


def _fmt(
    val: float | None,
    best: float | None,
    val_range: tuple[float, float] | None = None,
    gradient: bool = False,
) -> str:
    if val is None:
        return "--"
    s = f"{val:.1f}"
    if best is not None and round(val, 1) == round(best, 1):
        s = f"\\textbf{{{s}}}"
    if gradient and val_range is not None:
        op = _opacity(val, val_range[0], val_range[1])
        return f"\\gc{{{op}}}{{{s}}}"
    return s


def _build_block_rows(block: dict, records: list[dict]) -> str:
    bests = _block_bests(records)
    ranges = _block_ranges(records)
    label = block["label"]
    lines: list[str] = []
    for i, (mix, rec) in enumerate(zip(MIXES, records)):
        mix_label = MIX_DISPLAY[mix]
        metric_cells = [
            _fmt(rec[name], bests[name], ranges[name], name in GRADIENT_METRICS)
            for name in METRIC_KEYS
        ]
        if i == 0:
            row_label = (
                f"\\multirow{{4}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{label}}}}}}}"
            )
        else:
            row_label = ""
        cells = [row_label, mix_label] + metric_cells
        lines.append("    " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def generate_latex(block_data: dict[str, list[dict]]) -> str:
    block_rows: list[str] = []
    for i, block in enumerate(EVAL_BLOCKS):
        rows = _build_block_rows(block, block_data[block["key"]])
        block_rows.append(rows)
        if i < len(EVAL_BLOCKS) - 1:
            block_rows.append("\\midrule")

    body = "\n".join(block_rows)

    return (
        "\\begingroup\n"
        "\\renewcommand{\\arraystretch}{1.1}\n"
        "\\begin{table}[htp]\n"
        "\\centering\n"
        "\\caption[SFT+LoRA data-mix results]{\\textbf{Fine-tuning results across training data mixes.}\n"
        "Qwen3-VL-8B fine-tuned with LoRA on the in-the-wild (ItW), staged (Sta), synthetic (Syn),\n"
        "and combined (Sta{+}Syn{+}ItW) training mixes. As a representative of the staged partition\n"
        "we report results on CMDFall, the largest of the eight staged datasets.\n"
        "The best result per column within each evaluation block is highlighted in \\textbf{bold}.\n"
        "Darker blue cells indicate better performance for BAcc and F1 metrics.\n"
        "Metrics are denoted as \\textbf{B}alanced \\textbf{Acc}uracy, \\textbf{Acc}uracy,\n"
        "\\textbf{Se}nsitivity, and \\textbf{Sp}ecificity.}\n"
        "\\label{tab:sft_lora_data_mix}\n"
        "\n"
        "\\begin{tabular}{@{}cl rrr rrr rrr@{}}\n"
        "\\toprule\n"
        "\\multirow{2}{*}{Eval} & \\multirow{2}{*}{Train mix} &\n"
        "\\multicolumn{3}{c}{16-class} &\n"
        "\\multicolumn{3}{c}{Fall $\\Delta$} &\n"
        "\\multicolumn{3}{c}{Fallen $\\Delta$} \\\\\n"
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\n"
        " & & \\multicolumn{1}{c}{BAcc} & \\multicolumn{1}{c}{Acc} & \\multicolumn{1}{c}{F1}\n"
        "   & \\multicolumn{1}{c}{Se}   & \\multicolumn{1}{c}{Sp}  & \\multicolumn{1}{c}{F1}\n"
        "   & \\multicolumn{1}{c}{Se}   & \\multicolumn{1}{c}{Sp}  & \\multicolumn{1}{c}{F1} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "\\endgroup\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output .tex file path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    api = wandb.Api()
    block_data = fetch_all(api)
    latex = generate_latex(block_data)
    print(latex)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex)
    logger.info("Written to %s", output)


if __name__ == "__main__":
    main()
