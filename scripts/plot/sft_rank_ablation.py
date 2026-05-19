"""Plot SFT LoRA rank ablation results from W&B.

Fetches evaluation metrics from four fine-tuned Qwen3-VL-8B runs that were
evaluated in the fall-detection-zeroshot-v4 project, each trained with a
different LoRA rank (4, 8, 16, 32).  Produces a single line plot with rank
on the X axis and balanced accuracy, macro F1, and fall F1 on the Y axis.

Usage:
    python scripts/plot/sft_rank_ablation.py
    python scripts/plot/sft_rank_ablation.py --output outputs/plots/rank_ablation.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import wandb
from falldet.plot.base import compute_publication_figsize, set_publication_rc_defaults
from falldet.plot.metrics import COLORS

logger = logging.getLogger(__name__)

WANDB_ENTITY = "moritzm00"
WANDB_PROJECT = "fall-detection-zeroshot-v4"

# Explicit mapping: run_id -> LoRA rank r (inference run tags are unreliable)
RUN_RANKS: dict[str, int] = {
    "1hi0by2a": 4,
    "mdfkl0jd": 8,
    "a5nz8f5v": 16,
    "wo7nud57": 32,
}

METRIC_KEYS = {
    "Balanced Acc.": "OOPS_cs_balanced_accuracy",
    "Macro F1": "OOPS_cs_macro_f1",
    "Fall F1": "OOPS_cs_fall_f1",
    "Fallen F1": "OOPS_cs_fallen_f1",
}

_METRIC_STYLES: list[dict] = [
    {"color": COLORS["primary"], "marker": "o", "linestyle": "-"},
    {"color": COLORS["secondary"], "marker": "s", "linestyle": "-"},
    {"color": COLORS["tertiary"], "marker": "^", "linestyle": "-"},
    {"color": COLORS["quaternary"], "marker": "D", "linestyle": "-"},
]

DEFAULT_OUTPUT = Path("outputs/plots/sft_rank_ablation.pdf")


def fetch_runs() -> list[dict]:
    api = wandb.Api()
    records: list[dict] = []
    for rid, rank in RUN_RANKS.items():
        logger.info(f"Fetching run {rid} (rank {rank})")
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{rid}")
        record: dict = {"run_id": rid, "rank": rank}
        for label, key in METRIC_KEYS.items():
            value = run.summary.get(key)
            if value is None:
                raise ValueError(f"Metric {key!r} not found in run {rid} summary")
            record[label] = float(value) * 100.0
        records.append(record)
    return sorted(records, key=lambda r: r["rank"])


def plot_rank_ablation(records: list[dict], output: Path) -> None:
    set_publication_rc_defaults(use_tex=True, target="thesis")
    w, h = compute_publication_figsize(target="thesis", width_fraction=0.65, height_ratio=0.75)
    fig, ax = plt.subplots(figsize=(w, h))

    ranks = [r["rank"] for r in records]
    for (label, _), style in zip(METRIC_KEYS.items(), _METRIC_STYLES):
        values = [r[label] for r in records]
        ax.plot(
            ranks,
            values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=1.2,
            markersize=4,
            label=label,
        )

    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("Score (\\%)")
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    sns.despine(ax=ax)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    logger.info(f"Saved to {output}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    records = fetch_runs()
    for rec in records:
        logger.info(
            f"rank={rec['rank']:2d}  "
            + "  ".join(f"{label}={rec[label]:.1f}" for label in METRIC_KEYS)
        )
    plot_rank_ablation(records, Path(args.output))


if __name__ == "__main__":
    main()
