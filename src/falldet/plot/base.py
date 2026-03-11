"""Shared plotting defaults and publication-oriented matplotlib configuration."""

from collections.abc import Mapping
from typing import Literal, TypeAlias, cast

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

SeabornContext: TypeAlias = Literal["paper", "notebook", "talk", "poster"]
SeabornStyle: TypeAlias = Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]

PUBLICATION_TARGET_DEFAULTS: dict[str, dict[str, float | int]] = {
    "paper": {
        "text_width_pt": 246.0,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    },
    "thesis": {
        "text_width_pt": 427.43153,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    },
}

NORMAL_CONFUSION_OFFDIAGONAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "normal_confusion_offdiagonal",
    ["#ffffff", "#fddbc7", "#b2182b"],
)
RELATIVE_CONFUSION_DIAGONAL_CMAP = matplotlib.colormaps["RdBu"]


def _latex_pt_to_inches(value_pt: float) -> float:
    """Convert TeX points to inches."""
    return value_pt / 72.27


def _default_figsize() -> tuple[float, float]:
    """Return the current matplotlib default figure size as a tuple."""
    width, height = plt.rcParams["figure.figsize"]
    return float(width), float(height)


def compute_publication_figsize(
    *,
    target: str = "thesis",
    text_width_pt: float | None = None,
    width_fraction: float = 1.0,
    height_ratio: float = 0.66,
) -> tuple[float, float]:
    """Compute a manuscript-oriented matplotlib figure size in inches."""
    if target not in PUBLICATION_TARGET_DEFAULTS:
        raise ValueError(
            f"target must be one of {set(PUBLICATION_TARGET_DEFAULTS)}, got {target!r}."
        )

    default_text_width_pt = float(PUBLICATION_TARGET_DEFAULTS[target]["text_width_pt"])
    resolved_text_width_pt = default_text_width_pt if text_width_pt is None else text_width_pt
    if resolved_text_width_pt <= 0:
        raise ValueError(f"text_width_pt must be positive, got {resolved_text_width_pt!r}.")
    if width_fraction <= 0:
        raise ValueError(f"width_fraction must be positive, got {width_fraction!r}.")
    if height_ratio <= 0:
        raise ValueError(f"height_ratio must be positive, got {height_ratio!r}.")

    figure_width_in = _latex_pt_to_inches(resolved_text_width_pt) * width_fraction
    figure_height_in = figure_width_in * height_ratio
    return figure_width_in, figure_height_in


def set_publication_rc_defaults(
    *,
    target: str = "thesis",
    context: SeabornContext = "paper",
    style: SeabornStyle = "whitegrid",
    palette: str = "colorblind",
    font_scale: float = 1.0,
    use_tex: bool = False,
    text_width_pt: float | None = None,
    width_fraction: float = 1.0,
    height_ratio: float = 0.66,
    rc: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], tuple[float, float]]:
    """Set seaborn and matplotlib defaults for paper-ready plots.

    The defaults favor manuscript-friendly figures, serif typography, vector-friendly
    output, and restrained grid styling suitable for LaTeX manuscripts.

    Args:
        target: Preset sizing target. Supported values are ``"paper"``
            and ``"thesis"``.
        context: Seaborn plotting context, usually ``"paper"``.
        style: Seaborn style preset.
        palette: Seaborn color palette name.
        font_scale: Seaborn font scaling factor.
        use_tex: Enable matplotlib's LaTeX text rendering. This requires
            a working LaTeX installation.
        text_width_pt: LaTeX text width in TeX points. When omitted,
            a target-specific default is used.
        width_fraction: Fraction of the LaTeX text width used for the
            default matplotlib figure width.
        height_ratio: Figure height divided by figure width.
        rc: Optional rcParams overrides applied after the defaults.

    Returns:
        The resolved rcParams dictionary applied to matplotlib and
        seaborn, along with the computed figure size tuple.
    """
    if target not in PUBLICATION_TARGET_DEFAULTS:
        raise ValueError(
            f"target must be one of {set(PUBLICATION_TARGET_DEFAULTS)}, got {target!r}."
        )

    preset = PUBLICATION_TARGET_DEFAULTS[target]
    figure_width_in, figure_height_in = compute_publication_figsize(
        target=target,
        text_width_pt=text_width_pt,
        width_fraction=width_fraction,
        height_ratio=height_ratio,
    )

    resolved_rc: dict[str, object] = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.figsize": (figure_width_in, figure_height_in),
        "figure.constrained_layout.use": False,
        "font.size": preset["font.size"],
        "axes.titlesize": preset["axes.titlesize"],
        "axes.labelsize": preset["axes.labelsize"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": preset["legend.fontsize"],
        "legend.title_fontsize": preset["legend.title_fontsize"],
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "patch.linewidth": 0.8,
        "xtick.labelsize": preset["xtick.labelsize"],
        "ytick.labelsize": preset["ytick.labelsize"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": use_tex,
    }
    if use_tex:
        resolved_rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    if rc is not None:
        resolved_rc.update(dict(rc))

    sns.set_theme(
        context=cast(SeabornContext, context),
        style=cast(SeabornStyle, style),
        palette=palette,
        font="serif",
        font_scale=font_scale,
        rc=resolved_rc,
    )
    plt.rcParams.update(resolved_rc)

    return resolved_rc, (figure_width_in, figure_height_in)


__all__ = [
    "PUBLICATION_TARGET_DEFAULTS",
    "NORMAL_CONFUSION_OFFDIAGONAL_CMAP",
    "RELATIVE_CONFUSION_DIAGONAL_CMAP",
    "compute_publication_figsize",
    "set_publication_rc_defaults",
]
