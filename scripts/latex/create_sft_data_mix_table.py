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

# Zero-shot Qwen3-VL-8B inference runs, one per eval block (training-free baseline)
ZEROSHOT_RUN_IDS: dict[str, str] = {
    "Sta": "8fa8ojpb",  # cmdfall_cs_*     (cross_dataset_table.py Qwen3-VL-8B/cmdfall)
    "Syn": "j61rz8mk",  # wanfall_random_* (cross_dataset_table.py Qwen3-VL-8B/wanfall)
    "ItW": "p1r3exbe",  # OOPS_cs_*        (results_overview_table.py "Zero-shot")
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


def fetch_zeroshot(api: wandb.Api, block: dict, run_id: str) -> dict:
    """Fetch the training-free zero-shot record for an eval block (no LoRA check)."""
    wb_prefix = f"{block['wb_name']}_{block['split']}_"
    record: dict = {"block": block["key"], "mix": "zeroshot"}

    if run_id == "TODO":
        logger.warning("  zero-shot run_id is TODO for block=%s — skipping", block["key"])
        for name in METRIC_KEYS:
            record[name] = None
        return record

    logger.info("Fetching zero-shot run %s (block=%s)", run_id, block["key"])
    run = api.run(f"{ENTITY}/{INFERENCE_PROJECT}/{run_id}")
    for name, suffix in METRIC_SUFFIXES:
        val = run.summary.get(f"{wb_prefix}{suffix}")
        record[name] = float(val) * 100.0 if val is not None else None
    return record


def fetch_all(api: wandb.Api) -> dict[str, dict]:
    """Returns {block_key: {"zeroshot": record, "mixes": [record_per_mix, ...]}}."""
    results: dict[str, dict] = {}
    for block in EVAL_BLOCKS:
        block_key = block["key"]
        mix_records: list[dict] = []
        for mix in MIXES:
            run_id = EVAL_RUN_IDS[block_key][mix]
            mix_records.append(fetch_block(api, block, mix, run_id))
        zeroshot_record = fetch_zeroshot(api, block, ZEROSHOT_RUN_IDS[block_key])
        results[block_key] = {"zeroshot": zeroshot_record, "mixes": mix_records}
    return results


def _block_bests(records: list[dict]) -> dict[str, float | None]:
    bests: dict[str, float | None] = {}
    for name in METRIC_KEYS:
        vals = [r[name] for r in records if r[name] is not None]
        bests[name] = max(vals) if vals else None
    return bests


def _fmt(val: float | None, best: float | None) -> str:
    if val is None:
        return "--"
    s = f"{val:.1f}"
    if best is not None and round(val, 1) == round(best, 1):
        s = f"\\textbf{{{s}}}"
    return s


def _is_ood(block_key: str, mix: str) -> bool:
    return mix != "All" and mix != block_key


def _oodcell(content: str, is_ood: bool) -> str:
    return f"\\oodcell{{{content}}}" if is_ood else content


def _build_block_rows(block: dict, block_records: dict) -> str:
    zeroshot = block_records["zeroshot"]
    mix_records = block_records["mixes"]
    # Bold competition is among the fine-tuned mixes only — zero-shot is excluded.
    bests = _block_bests(mix_records)
    label = block["label"]
    n_rows = 1 + len(MIXES)

    rotated_label = (
        f"\\multirow{{{n_rows}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{label}}}}}}}"
    )

    lines: list[str] = []

    # Zero-shot row: train mix shown as an em dash, whole row shaded, no bold.
    zs_cells = [_oodcell(_fmt(zeroshot[name], None), True) for name in METRIC_KEYS]
    lines.append("    " + " & ".join([rotated_label, _oodcell("--", True)] + zs_cells) + " \\\\")

    # Fine-tuned mix rows.
    for mix, rec in zip(MIXES, mix_records):
        is_ood = _is_ood(block["key"], mix)
        mix_label = _oodcell(MIX_DISPLAY[mix], is_ood)
        metric_cells = [_oodcell(_fmt(rec[name], bests[name]), is_ood) for name in METRIC_KEYS]
        lines.append("    " + " & ".join(["", mix_label] + metric_cells) + " \\\\")

    return "\n".join(lines)


def generate_latex(block_data: dict[str, dict]) -> str:
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
        "\\newcommand{\\oodcell}[1]{\\cellcolor{blue!6}#1}\n"
        "\\begin{table}[htp]\n"
        "\\centering\n"
        "\\begin{threeparttable}\n"
        "\\caption[SFT+LoRA data-mix results]{\\textbf{Fine-tuning results across training data mixes.}\n"
        "Qwen3-VL-8B zero-shot and fine-tuned with LoRA on the in-the-wild (ItW), staged (Sta), synthetic\n"
        "(Syn), and combined (Sta{+}Syn{+}ItW) training mixes. As a representative of the staged partition\n"
        "we report results on CMDFall, the largest of the eight staged datasets.\n"
        "The best fine-tuned result per column within each evaluation block is highlighted in \\textbf{bold}.\n"
        "Metrics are denoted as \\textbf{B}alanced \\textbf{Acc}uracy, \\textbf{Acc}uracy,\n"
        "\\textbf{F1} score, \\textbf{Se}nsitivity, and \\textbf{Sp}ecificity.}\n"
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
        "\\begin{tablenotes}\n"
        "\\scriptsize\n"
        "\\item[] ``--'' denotes the training-free zero-shot baseline.\n"
        "\\item[] \\colorbox{blue!6}{Shaded} cells mark the zero-shot row and cross-domain rows that use no in-domain training data.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
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
