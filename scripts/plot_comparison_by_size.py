from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -------------------------
# Data
# -------------------------
data = {
    "Model": [
        "VMAE-K400",
        "InternVL3.5-2B",
        "Qwen3-VL-2B",
        "InternVL3.5-4B",
        "Qwen3-VL-4B",
        "InternVL3.5-8B",
        "Qwen3-VL-8B",
        "InternVL3.5-38B",
        "Qwen3-VL-32B",
    ],
    "16-class": [21.9, 20.3, 13.4, 17.6, 21.2, 21.6, 21.8, 23.7, 21.6],
    "Fall": [65.6, 52.5, 53.7, 42.0, 52.9, 57.6, 53.1, 57.1, 46.7],
    "Fallen": [41.0, 28.8, 8.9, 10.2, 14.4, 27.8, 28.8, 30.5, 28.2],
}

# Create lookup dict for model -> scores
model_scores = {
    model: {"16-class": data["16-class"][i], "Fall": data["Fall"][i], "Fallen": data["Fallen"][i]}
    for i, model in enumerate(data["Model"])
}

# -------------------------
# Size groupings
# -------------------------
size_groups = {
    "Tiny": {"InternVL": "InternVL3.5-2B", "QwenVL": "Qwen3-VL-2B"},
    "Small": {"InternVL": "InternVL3.5-4B", "QwenVL": "Qwen3-VL-4B"},
    "Medium": {"InternVL": "InternVL3.5-8B", "QwenVL": "Qwen3-VL-8B"},
    "Large": {"InternVL": "InternVL3.5-38B", "QwenVL": "Qwen3-VL-32B"},
}

sizes = list(size_groups.keys())
families = ["InternVL", "QwenVL"]

# Baseline model
baseline_model = "VMAE-K400"

# -------------------------
# Style configuration
# -------------------------
USE_HATCHING = False  # Set to True to enable hatching patterns

hatch_map = {
    "InternVL": "//" if USE_HATCHING else "",
    "QwenVL": "" if USE_HATCHING else "",
}

color_map = {
    "QwenVL": "#1f78b4",  # dark blue
    "InternVL": "#a6cee3",
    # "QwenVL": "#33a02c",  # dark green
}

# -------------------------
# Tasks to plot
# -------------------------
tasks = ["16-class", "Fall", "Fallen"]
task_filenames = {
    "16-class": "f1_by_size_16class.pdf",
    "Fall": "f1_by_size_fall.pdf",
    "Fallen": "f1_by_size_fallen.pdf",
}

# -------------------------
# Matplotlib config (paper-ready)
# -------------------------
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

# -------------------------
# Output directory
# -------------------------
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Generate one plot per task
# -------------------------
for task in tasks:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sizes))
    width = 0.35

    # Get scores for each family at each size
    for i, family in enumerate(families):
        scores = []
        for size in sizes:
            model_name = size_groups[size][family]
            scores.append(model_scores[model_name][task])

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=family,
            color=color_map[family],
            hatch=hatch_map[family],
            edgecolor="black",
        )

    # Baseline horizontal line
    baseline_score = model_scores[baseline_model][task]
    ax.axhline(
        baseline_score,
        linestyle="--",
        linewidth=1.5,
        color="gray",
        label=f"VMAE-K400 ({baseline_score:.1f})",
    )

    # Axes & grid
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Model Size")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, max(model_scores[baseline_model][task], max(data[task])) * 1.15)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.6)

    # Legend (only for 16-class plot)
    if task == "16-class":
        family_legend = [
            Patch(
                facecolor=color_map["InternVL"],
                edgecolor="black",
                hatch=hatch_map["InternVL"],
                label="InternVL",
            ),
            Patch(
                facecolor=color_map["QwenVL"],
                edgecolor="black",
                hatch=hatch_map["QwenVL"],
                label="QwenVL",
            ),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="VMAE-K400"),
        ]

        ax.legend(
            handles=family_legend,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
        )

    plt.tight_layout()

    # Export
    output_path = output_dir / task_filenames[task]
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

print("Done!")

# -------------------------
# Generate LaTeX code for subfigure
# -------------------------
latex_code = r"""
\begin{figure}[t]
    \centering
    % Top image now matches the total width of the bottom row (~0.98)
    \begin{subfigure}[b]{0.98\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/f1_by_size_16class.pdf}
        \caption{16-class}
        \label{fig:f1_by_size_16class}
    \end{subfigure}

    \vspace{1em} % Slightly more breathing room

    % Bottom row: 0.48 + 0.48 + hfill = 0.96 total width
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/f1_by_size_fall.pdf}
        \caption{Fall}
        \label{fig:f1_by_size_fall}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/f1_by_size_fallen.pdf}
        \caption{Fallen}
        \label{fig:f1_by_size_fallen}
    \end{subfigure}

    \caption{F1 scores by model size across different tasks. Size categories: Tiny (2B), Small (4B), Medium (8B), Large (32B/38B). The dashed line indicates the VMAE-K400 baseline.}
    \label{fig:f1_by_size_comparison}
\end{figure}
""".strip()

print("\n" + "=" * 50)
print("LaTeX code for subfigure:")
print("=" * 50)
print(latex_code)
