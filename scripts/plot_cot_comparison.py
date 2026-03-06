"""Generate comparison plots for Zero-shot (Instruct) vs Chain-of-Thought (Thinking) experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-zeroshot-v4"
DATASET = "OOPS"
SPLIT = "cs"

MODEL_SIZES = ["2B", "8B", "32B"]

INSTRUCT_RUN_IDS = {
    "2B": "d4e8gwu0",
    "8B": "p1r3exbe",
    "32B": "toe74d9a",
}

# WandB Run IDs for Thinking models
THINKING_RUN_IDS = {
    "2B": "70fkhp71",
    "8B": "gzjkhtvu",
    "32B": "wpnjcvjm",
}

# Metrics to compare
METRICS = {
    "16-class BAcc": f"{DATASET}_{SPLIT}_balanced_accuracy",
    "16-class F1": f"{DATASET}_{SPLIT}_macro_f1",
    "Fall F1": f"{DATASET}_{SPLIT}_fall_f1",
    "Fallen F1": f"{DATASET}_{SPLIT}_fallen_f1",
}
METRIC_NAMES = list(METRICS.keys())

# ==========================================
# STYLE CONFIGURATION
# ==========================================
color_map = {
    "Thinking": "#1f78b4",
    "Instruct": "#a8cee3",
}

# Matplotlib config (paper-ready)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

# Output directory
OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# DATA FETCHING
# ==========================================
def fetch_metrics(api: wandb.Api, run_id: str) -> dict[str, float]:
    """Fetch metrics from a WandB run."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary
        metrics = {}
        for name, key in METRICS.items():
            val = summary.get(key)
            if val is not None:
                metrics[name] = val * 100  # Convert to percentage
            else:
                metrics[name] = None
        return metrics
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return {name: None for name in METRIC_NAMES}


def fetch_all_data(api: wandb.Api) -> dict:
    """Fetch all data for Instruct and Thinking models."""
    data = {"Instruct": {}, "Thinking": {}}

    for size in MODEL_SIZES:
        # Fetch Instruct
        instruct_metrics = fetch_metrics(api, INSTRUCT_RUN_IDS[size])
        data["Instruct"][size] = instruct_metrics

        # Fetch Thinking
        thinking_metrics = fetch_metrics(api, THINKING_RUN_IDS[size])
        data["Thinking"][size] = thinking_metrics

    return data


# ==========================================
# PLOTTING
# ==========================================
def create_comparison_plot(
    size: str,
    instruct_scores: dict[str, float],
    thinking_scores: dict[str, float],
) -> plt.Figure:
    """Create a grouped bar chart comparing Instruct vs Thinking for a single model size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(METRIC_NAMES))
    width = 0.3

    # Get scores as lists
    instruct_vals = [instruct_scores.get(m, 0) or 0 for m in METRIC_NAMES]
    thinking_vals = [thinking_scores.get(m, 0) or 0 for m in METRIC_NAMES]

    # Create bars (instruct, thinking)
    ax.bar(
        x - width / 2,
        instruct_vals,
        width,
        label="Zero-Shot",
        color=color_map["Instruct"],
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        thinking_vals,
        width,
        label="Zero-Shot CoT",
        color=color_map["Thinking"],
        edgecolor="black",
    )

    # Add improvement annotations (percentage points)
    for i, (inst, think) in enumerate(zip(instruct_vals, thinking_vals)):
        if inst > 0 and think is not None:
            diff = think - inst
            sign = "+" if diff >= 0 else ""
            ax.annotate(
                f"{sign}{diff:.1f} pp",
                xy=(x[i] + width / 2, think),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black" if diff >= 0 else "red",
            )

    # Axes & grid
    ax.set_ylabel("Score (\\%)")
    ax.set_xlabel("Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)

    # Set y-axis limits
    all_vals = instruct_vals + thinking_vals
    ax.set_ylim(0, max(v for v in all_vals if v is not None) * 1.2)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.6)

    # Legend
    legend_handles = [
        Patch(facecolor=color_map["Instruct"], edgecolor="black", label="Zero-Shot"),
        Patch(facecolor=color_map["Thinking"], edgecolor="black", label="Zero-Shot CoT"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
    )

    plt.tight_layout()
    return fig


def generate_all_plots(data: dict) -> list[Path]:
    """Generate comparison plots for all model sizes."""
    output_paths = []

    for size in MODEL_SIZES:
        instruct_scores = data["Instruct"][size]
        thinking_scores = data["Thinking"][size]

        fig = create_comparison_plot(size, instruct_scores, thinking_scores)

        # Save
        filename = f"cot_comparison_{size.lower().replace('-', '_')}.pdf"
        output_path = OUTPUT_DIR / filename
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")
        output_paths.append(output_path)

    return output_paths


def generate_latex_code(output_paths: list[Path]) -> str:
    """Generate LaTeX subfigure code for the plots."""
    latex_parts = []

    latex_parts.append(r"\begin{figure}[t]")
    latex_parts.append(r"    \centering")

    for i, path in enumerate(output_paths):
        size = MODEL_SIZES[i]
        latex_parts.append("    \\begin{subfigure}[b]{0.48\\textwidth}")
        latex_parts.append(r"        \centering")
        latex_parts.append(f"        \\includegraphics[width=\\textwidth]{{figures/{path.name}}}")
        latex_parts.append(f"        \\caption{{Qwen3-VL-{size}}}")
        latex_parts.append(
            f"        \\label{{fig:cot_comparison_{size.lower().replace('-', '_')}}}"
        )
        latex_parts.append(r"    \end{subfigure}")

        # Add spacing between rows
        if i % 2 == 1 and i < len(output_paths) - 1:
            latex_parts.append(r"    \vspace{1em}")
        elif i % 2 == 0 and i < len(output_paths) - 1:
            latex_parts.append(r"    \hfill")

    latex_parts.append(r"")
    latex_parts.append(
        r"    \caption{Comparison of Qwen3-VL models with Instruct vs Thinking (Chain-of-Thought) prompting. "
        r"Annotations show improvement in percentage points from Instruct to Thinking.}"
    )
    latex_parts.append(r"    \label{fig:cot_comparison}")
    latex_parts.append(r"\end{figure}")

    return "\n".join(latex_parts)


# ==========================================
# MAIN
# ==========================================
def main():
    print("Fetching data from WandB...")
    api = wandb.Api()
    data = fetch_all_data(api)

    print("\nGenerating plots...")
    output_paths = generate_all_plots(data)

    print("\n" + "=" * 50)
    print("LaTeX code for subfigure:")
    print("=" * 50)
    latex_code = generate_latex_code(output_paths)
    print(latex_code)

    print("\nDone!")


if __name__ == "__main__":
    main()
