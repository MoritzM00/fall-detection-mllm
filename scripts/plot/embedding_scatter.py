"""Visualise embedding spaces as 2-D scatter plots via t-SNE, UMAP, HNNE, or PCA.

Each input is a .pt file produced by the embed task (keys: ``embeddings``,
``samples``).  All files are projected into the *same* coordinate space by
fitting the reduction on the concatenated embeddings, then splitting back for
optional per-panel rendering.

Usage examples
--------------
Single file (t-SNE default)::

    python scripts/plot/embedding_scatter.py \
        outputs/embeddings/OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt

Two files overlaid::

    python scripts/plot/embedding_scatter.py \
        outputs/embeddings/OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt \
        outputs/embeddings/OOPS_cs_val_16@7_5_Qwen3-VL-Embedding-2B_448.pt \
        --names train val

Split panels::

    python scripts/plot/embedding_scatter.py \
        outputs/embeddings/OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt \
        outputs/embeddings/OOPS_cs_val_16@7_5_Qwen3-VL-Embedding-2B_448.pt \
        --split --names train val

PCA variant::

    python scripts/plot/embedding_scatter.py \
        outputs/embeddings/OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt \
        --method pca

UMAP variant::

    python scripts/plot/embedding_scatter.py \
        outputs/embeddings/OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt \
        --method umap --n-neighbors 15 --min-dist 0.1

HNNE variant::

    python scripts/plot/embedding_scatter.py \
        outputs/embeddings/OOPS_cs_train_16@7_5_Qwen3-VL-Embedding-2B_448.pt \
        --method hnne
"""

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hnne import HNNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from umap import UMAP

from falldet.embeddings import load_embeddings
from falldet.plot import COLORS, compute_publication_figsize, set_publication_rc_defaults

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs/plots")
MARKERS = ["o", "^", "s", "D", "v", "P"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_label_strs(samples: list[dict]) -> list[str]:
    """Return string labels from a list of sample metadata dicts."""
    return [s.get("label_str") or str(s.get("label", "unknown")) for s in samples]


def build_label_colormap(labels: list[str]) -> dict[str, tuple]:
    """Assign colours to sorted unique labels.

    ``"fall"`` is anchored to ``COLORS["error"]`` (red-orange) and
    ``"fallen"`` to ``COLORS["secondary"]`` (orange) for semantic emphasis.
    All other labels receive ``tab20`` colours in alphabetical order.
    """
    unique_labels = sorted(set(labels))
    anchored = {"fall": COLORS["error"], "fallen": COLORS["secondary"]}

    cmap = plt.get_cmap("tab20")
    free_labels = [lb for lb in unique_labels if lb not in anchored]
    colormap: dict[str, tuple] = {}
    tab20_idx = 0
    for lb in unique_labels:
        if lb in anchored:
            colormap[lb] = anchored[lb]
        else:
            colormap[lb] = cmap(tab20_idx / max(len(free_labels), 1))
            tab20_idx += 1
    return colormap


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    method: str,
    perplexity: int,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    """Project *embeddings* to 2-D using t-SNE, UMAP, HNNE, or PCA.

    Args:
        embeddings: Float array of shape ``(n, dim)``.
        method: ``"tsne"``, ``"umap"``, ``"hnne"``, or ``"pca"``.
        perplexity: t-SNE perplexity (ignored for UMAP/HNNE/PCA).
        random_state: Random seed.
        n_neighbors: UMAP number of neighbours (ignored for t-SNE/HNNE/PCA).
        min_dist: UMAP minimum distance (ignored for t-SNE/HNNE/PCA).

    Returns:
        Array of shape ``(n, 2)``.
    """
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_jobs=-1,
        )
    elif method == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
    elif method == "hnne":
        reducer = HNNE(n_components=2, metric="cosine", random_state=random_state)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError(
            f"Unknown reduction method: {method!r}. Choose 'tsne', 'umap', 'hnne', or 'pca'."
        )
    pipeline = Pipeline([("normalizer", Normalizer(norm="l2")), ("reducer", reducer)])
    return pipeline.fit_transform(embeddings)


def plot_embedding_scatter(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: list[str],
    colormap: dict[str, tuple],
    method_label: str,
    *,
    alpha: float,
    point_size: float,
    marker: str = "o",
) -> None:
    """Scatter embeddings on *ax* coloured by class label.

    Args:
        ax: Target axes.
        coords: 2-D coordinates, shape ``(n, 2)``.
        labels: Per-point class label strings.
        colormap: Mapping from label string to colour.
        method_label: Axis label prefix (e.g. ``"t-SNE"`` or ``"PC"``).
        alpha: Point opacity.
        point_size: Scatter point size in points².
        marker: Matplotlib marker style.
    """
    unique_labels = sorted(set(labels))
    label_array = np.asarray(labels)

    for lb in unique_labels:
        mask = label_array == lb
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colormap.get(lb, (0.5, 0.5, 0.5))],
            label=lb,
            alpha=alpha,
            s=point_size,
            marker=marker,
            linewidths=0.4,
            edgecolors="white",
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_shared_legend(
    fig: plt.Figure,
    axes,
    colormap: dict[str, tuple],
    n_panels: int = 1,
    file_markers: dict[str, str] | None = None,
) -> None:
    """Attach class-colour legend (and optional per-file marker legend) below the figure."""
    sorted_labels = sorted(colormap)
    class_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colormap[lb],
            markersize=5,
            label=lb,
        )
        for lb in sorted_labels
    ]
    if file_markers:
        marker_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=mk,
                color="none",
                markerfacecolor="0.3",
                markeredgecolor="none",
                markersize=5,
                label=name,
            )
            for name, mk in file_markers.items()
        ]
        all_handles = class_handles + marker_handles
    else:
        all_handles = class_handles
    n_rows = 2 if n_panels > 1 else 3
    ncols = math.ceil(len(all_handles) / n_rows)
    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncols,
        frameon=False,
        fontsize=8,
    )


def _method_axis_label(method: str) -> str:
    return {"tsne": "t-SNE", "umap": "UMAP", "hnne": "HNNE", "pca": "PC"}.get(
        method, method.upper()
    )


def _default_output_stem(input_paths: list[Path], methods: list[str]) -> str:
    method_str = "_".join(methods)
    if len(input_paths) == 1:
        return f"embedding_scatter_{method_str}_{input_paths[0].stem}"
    stems = "_vs_".join(p.stem for p in input_paths[:3])
    if len(input_paths) > 3:
        stems += f"_and_{len(input_paths) - 3}_more"
    return f"embedding_scatter_{method_str}_{stems}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2-D scatter plot of video embeddings via t-SNE, UMAP, or PCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="One or more .pt embedding files",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["tsne", "umap", "hnne", "pca"],
        default=["tsne"],
        metavar="METHOD",
        help="Dimensionality reduction method(s). Multiple values produce side-by-side panels "
        "(single input file only). Choices: tsne, umap, hnne, pca.",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity (ignored for UMAP/PCA)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP number of neighbours (ignored for t-SNE/PCA)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance between embedded points (ignored for t-SNE/PCA)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Create separate panels per input file instead of overlaying on one plot",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Display names for each input file (in order)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Overall figure title",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Point opacity",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=12.0,
        help="Scatter point size in points²",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (without extension; PDF and PNG saved automatically)",
    )
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX rendering (useful when LaTeX is not installed)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    input_paths = [Path(p) for p in args.input]
    methods: list[str] = args.method
    multi_method = len(methods) > 1

    if multi_method:
        if len(input_paths) != 1:
            raise SystemExit("Multiple --method values require exactly one input file.")
        if args.split:
            raise SystemExit("--split is not compatible with multiple --method values.")

    if args.names is not None and len(args.names) != len(input_paths):
        raise SystemExit(
            f"--names must provide exactly one name per input file "
            f"({len(input_paths)} files, {len(args.names)} names given)."
        )

    display_names = args.names if args.names else [p.stem for p in input_paths]

    set_publication_rc_defaults(use_tex=not args.no_tex, rc={"savefig.pad_inches": 0.1})

    # ------------------------------------------------------------------
    # Load embeddings
    # ------------------------------------------------------------------
    all_embeddings: list[np.ndarray] = []
    all_labels: list[list[str]] = []

    for path in input_paths:
        emb_tensor, samples = load_embeddings(path)
        all_embeddings.append(emb_tensor.float().numpy())
        all_labels.append(_extract_label_strs(samples))
        logger.info(f"Loaded {len(samples)} samples from {path.name}")

    reduce_kwargs = dict(
        perplexity=args.perplexity,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    # ------------------------------------------------------------------
    # Multi-method mode: one panel per method, single file
    # ------------------------------------------------------------------
    if multi_method:
        embeddings = all_embeddings[0]
        labels = all_labels[0]
        colormap = build_label_colormap(labels)

        n_panels = len(methods)
        panel_w, panel_h = compute_publication_figsize(width_fraction=0.5, height_ratio=1)
        fig, axes_raw = plt.subplots(
            1,
            n_panels,
            figsize=(panel_w * n_panels, panel_h),
            squeeze=False,
            sharey=False,
            sharex=False,
        )
        axes = axes_raw.ravel()
        panel_letters = [chr(ord("a") + i) for i in range(n_panels)]

        for idx, method in enumerate(methods):
            logger.info(
                f"Reducing {len(embeddings)} embeddings "
                f"(dim={embeddings.shape[1]}) with {method.upper()} ..."
            )
            coords = reduce_embeddings(embeddings, method=method, **reduce_kwargs)
            logger.info(f"{method.upper()} complete.")
            plot_embedding_scatter(
                axes[idx],
                coords,
                labels,
                colormap,
                _method_axis_label(method),
                alpha=args.alpha,
                point_size=args.point_size,
            )
            axes[idx].text(
                0.5,
                -0.18,
                f"({panel_letters[idx]}) {_method_axis_label(method)}",
                transform=axes[idx].transAxes,
                ha="center",
                va="top",
                fontweight="bold",
                fontsize=11,
            )

        if args.title:
            fig.suptitle(args.title, fontweight="bold", fontsize=12)

        _add_shared_legend(fig, axes, colormap, n_panels=n_panels)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)

    # ------------------------------------------------------------------
    # Single-method mode: panels per file (split) or overlay
    # ------------------------------------------------------------------
    else:
        method = methods[0]
        concat_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(
            f"Reducing {len(concat_embeddings)} embeddings "
            f"(dim={concat_embeddings.shape[1]}) with {method.upper()} ..."
        )
        coords_2d = reduce_embeddings(concat_embeddings, method=method, **reduce_kwargs)
        logger.info("Reduction complete.")

        sizes = [len(e) for e in all_embeddings]
        split_coords: list[np.ndarray] = []
        offset = 0
        for sz in sizes:
            split_coords.append(coords_2d[offset : offset + sz])
            offset += sz

        combined_labels = [lb for lbls in all_labels for lb in lbls]
        colormap = build_label_colormap(combined_labels)
        method_label = _method_axis_label(method)

        n_panels = len(input_paths) if args.split else 1
        width_fraction = 0.5 if n_panels > 1 else 1.0
        panel_w, panel_h = compute_publication_figsize(
            width_fraction=width_fraction, height_ratio=1
        )
        fig, axes_raw = plt.subplots(
            1,
            n_panels,
            figsize=(panel_w * n_panels, panel_h),
            squeeze=False,
            sharey=args.split,
            sharex=args.split,
        )
        axes = axes_raw.ravel()
        panel_letters = [chr(ord("a") + i) for i in range(len(input_paths))]

        if args.split:
            for idx, (coords, labels, name, letter) in enumerate(
                zip(split_coords, all_labels, display_names, panel_letters)
            ):
                plot_embedding_scatter(
                    axes[idx],
                    coords,
                    labels,
                    colormap,
                    method_label,
                    alpha=args.alpha,
                    point_size=args.point_size,
                    marker=MARKERS[idx % len(MARKERS)],
                )
                axes[idx].text(
                    0.5,
                    -0.18,
                    f"({letter}) {name}",
                    transform=axes[idx].transAxes,
                    ha="center",
                    va="top",
                    fontweight="bold",
                    fontsize=11,
                )
            file_markers = None
        else:
            multi_file = len(input_paths) > 1
            for idx, (coords, labels) in enumerate(zip(split_coords, all_labels)):
                plot_embedding_scatter(
                    axes[0],
                    coords,
                    labels,
                    colormap,
                    method_label,
                    alpha=args.alpha,
                    point_size=args.point_size,
                    marker=MARKERS[idx % len(MARKERS)] if multi_file else "o",
                )
            if multi_file:
                file_markers = {
                    name: MARKERS[i % len(MARKERS)] for i, name in enumerate(display_names)
                }
            else:
                file_markers = None

        if args.title:
            fig.suptitle(args.title, fontweight="bold", fontsize=12)

        _add_shared_legend(fig, axes, colormap, n_panels=n_panels, file_markers=file_markers)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.output:
        out_stem = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_stem = DEFAULT_OUTPUT_DIR / _default_output_stem(input_paths, methods)

    out_stem.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        out_path = out_stem.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
