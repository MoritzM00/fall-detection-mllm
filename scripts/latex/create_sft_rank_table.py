"""Generate the LoRA rank ablation LaTeX table and write it to the thesis tables directory.

Fetches evaluation metrics from the four fine-tuned Qwen3-VL-8B inference runs
(fall-detection-zeroshot-v4) that correspond to the rank sweep r ∈ {4, 8, 16, 32},
and produces a table matching the structure of tables/sft_lora_rank.tex.

Usage:
    python scripts/latex/create_sft_rank_table.py
    python scripts/latex/create_sft_rank_table.py --output path/to/sft_lora_rank.tex
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)

ENTITY = "moritzm00"
PROJECT = "fall-detection-zeroshot-v4"
TRAIN_PROJECT = "falldet-mllm-finetune"
DATASET = "OOPS"
SPLIT = "cs"

# Explicit mapping: run_id -> LoRA rank r (inference run tags are unreliable)
RUN_RANKS: dict[str, int] = {
    "1hi0by2a": 4,
    "mdfkl0jd": 8,
    "a5nz8f5v": 16,
    "wo7nud57": 32,
}


METRICS: list[tuple[str, str]] = [
    ("bacc", f"{DATASET}_{SPLIT}_balanced_accuracy"),
    ("acc", f"{DATASET}_{SPLIT}_accuracy"),
    ("macro_f1", f"{DATASET}_{SPLIT}_macro_f1"),
    ("fall_sensitivity", f"{DATASET}_{SPLIT}_fall_sensitivity"),
    ("fall_specificity", f"{DATASET}_{SPLIT}_fall_specificity"),
    ("fall_f1", f"{DATASET}_{SPLIT}_fall_f1"),
    ("fallen_sensitivity", f"{DATASET}_{SPLIT}_fallen_sensitivity"),
    ("fallen_specificity", f"{DATASET}_{SPLIT}_fallen_specificity"),
    ("fallen_f1", f"{DATASET}_{SPLIT}_fallen_f1"),
]

DEFAULT_OUTPUT = Path("~/thesis-overleaf/tables/sft_lora_rank.tex").expanduser()

METRIC_KEYS = [name for name, _ in METRICS]


def fetch_data(api: wandb.Api) -> list[dict]:
    records: list[dict] = []
    for rid, rank in RUN_RANKS.items():
        logger.info(f"Fetching inference run {rid} (rank {rank})")
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        record: dict = {"rank": rank}
        for name, key in METRICS:
            val = run.summary.get(key)
            record[name] = float(val) * 100.0 if val is not None else None

        lora_path = (run.config.get("lora") or {}).get("path", "")
        train_run_name = Path(lora_path).parent.name  # e.g. Qwen3-VL-8B-Instruct-F16at7.5_pjmzzx8i
        train_run_id = train_run_name.rsplit("_", 1)[-1] if train_run_name else None
        record["trainable_M"] = None
        if train_run_id:
            logger.info(f"  Fetching training run {train_run_id} from lora path")
            train_run = api.run(f"{ENTITY}/{TRAIN_PROJECT}/{train_run_id}")
            params = train_run.summary.get("trainable_params")
            record["trainable_M"] = float(params) / 1e6 if params is not None else None

        records.append(record)
    return sorted(records, key=lambda r: r["rank"])


def _col_bests(records: list[dict]) -> dict[str, float | None]:
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
        return f"\\textbf{{{s}}}"
    return s


def _fmt_M(val: float | None) -> str:
    if val is None:
        return "--"
    return f"{val:.1f}M"


def generate_latex(records: list[dict]) -> str:
    bests = _col_bests(records)

    rows: list[str] = []
    for rec in records:
        pct_cell = _fmt_M(rec.get("trainable_M"))
        metric_cells = [_fmt(rec[name], bests[name]) for name in METRIC_KEYS]
        cells = [str(rec["rank"]), pct_cell] + metric_cells
        rows.append("    " + " & ".join(cells) + " \\\\")

    rows_str = "\n".join(rows)

    return (
        "\\begingroup\n"
        "\\renewcommand{\\arraystretch}{1.1}\n"
        "\\begin{table}[htp]\n"
        "\\centering\n"
        "\\caption[LoRA rank ablation]{\\textbf{LoRA rank ablation.}\n"
        "Qwen3-VL-8B fine-tuned on the in-the-wild partition (ItW) only, evaluated on ItW.\n"
        "All runs use $\\alpha = 2r$ and AdamW optimizer.\n"
        "The best result per column is highlighted in \\textbf{bold}. Metrics are denoted as\n"
        "\\textbf{B}alanced \\textbf{Acc}uracy, \\textbf{Acc}uracy, \\textbf{Se}nsitivity,"
        " and \\textbf{Sp}ecificity.}\n"
        "\\label{tab:sft_lora_rank}\n"
        "\n"
        "\\begin{tabular}{@{}l r rrr rrr rrr@{}}\n"
        "\\toprule\n"
        "\\multirow{2}{*}{Rank $r$} &\n"
        "\\multirow{2}{*}{Params} &\n"
        "\\multicolumn{3}{c}{16-class} &\n"
        "\\multicolumn{3}{c}{Fall $\\Delta$} &\n"
        "\\multicolumn{3}{c}{Fallen $\\Delta$} \\\\\n"
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\n"
        " & & \\multicolumn{1}{c}{BAcc} & \\multicolumn{1}{c}{Acc} & \\multicolumn{1}{c}{F1}\n"
        " &  \\multicolumn{1}{c}{Se}   & \\multicolumn{1}{c}{Sp}  & \\multicolumn{1}{c}{F1}\n"
        " &  \\multicolumn{1}{c}{Se}   & \\multicolumn{1}{c}{Sp}  & \\multicolumn{1}{c}{F1} \\\\\n"
        "\\midrule\n"
        f"{rows_str}\n"
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
    records = fetch_data(api)
    latex = generate_latex(records)
    print(latex)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex)
    logger.info(f"Written to {output}")


if __name__ == "__main__":
    main()
