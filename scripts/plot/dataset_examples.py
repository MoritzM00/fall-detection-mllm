"""Generate example frame strips for several fall-detection datasets.

For each dataset group, this loads the Hydra inference config (composed per dataset),
instantiates the dataset via the standard factory, and writes one horizontal strip of
``num_frames`` frames for (a) a ``fall`` segment and (b) a random non-fall segment.

Requirements (same as ``vllm_inference.py``): ``OMNIFALL_ROOT`` must be set and HF access
is needed for the ``hf://simplexsigil2/...`` annotation/split CSVs.

Usage examples
--------------
Default list of omnifall datasets::

    python scripts/plot/dataset_examples.py

Single dataset::

    python scripts/plot/dataset_examples.py --datasets omnifall/video/oops

Override the segment indices for a group (skips auto fall/non-fall picking)::

    python scripts/plot/dataset_examples.py --datasets omnifall/video/oops \
        --indices oops=12,45
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from hydra import compose, initialize_config_dir

from falldet.data.video_dataset_factory import get_video_datasets
from falldet.plot import compute_publication_figsize, set_publication_rc_defaults
from falldet.schemas import from_dictconfig

logger = logging.getLogger(__name__)

CONFIG_DIR = (Path(__file__).resolve().parents[2] / "config").resolve()
DEFAULT_OUTPUT_DIR = Path("outputs/plots/dataset_examples")
DEFAULT_DATASETS = [
    "omnifall/video/oops",
    "omnifall/video/cmdfall",
    "omnifall/video/edf",
    "omnifall/video/occu",
    "omnifall/video/le2i",
    "omnifall/video/mcfd",
    "omnifall/video/up-fall",
    "omnifall/video/caucafall",
    "omnifall/video/gmdcsa24",
]


def parse_indices(raw: list[str] | None) -> dict[str, list[int]]:
    """Parse ``--indices group=12,45`` overrides into a ``{basename: [idx, ...]}`` map."""
    overrides: dict[str, list[int]] = {}
    for item in raw or []:
        key, _, values = item.partition("=")
        overrides[key.strip()] = [int(v) for v in values.split(",") if v.strip()]
    return overrides


def pick_example_indices(
    dataset, rng: np.random.Generator, n_examples: int, labels: list[str] | None = None
) -> list[int]:
    """Pick ``n_examples`` diverse segment indices without decoding video.

    With ``labels`` given, picks ``n_examples`` random segments per requested label instead.
    Otherwise prefers ``fall`` first, then one random segment per distinct action label, and
    finally random additional segments if more are still needed. Returns unique indices.
    """
    by_label: dict[str, list[int]] = {}
    for i, s in enumerate(dataset.video_segments):
        by_label.setdefault(s["label_str"], []).append(i)

    if not by_label:
        logger.warning("Dataset has no segments; falling back to index 0")
        return [0]

    if labels:
        chosen: list[int] = []
        for lbl in labels:
            pool = by_label.get(lbl, [])
            if not pool:
                logger.warning("No segments with label '%s'; available: %s", lbl, sorted(by_label))
                continue
            picks = rng.choice(pool, size=min(n_examples, len(pool)), replace=False)
            chosen.extend(int(i) for i in picks)
        return chosen

    chosen: list[int] = []
    # 1) fall first, 2) one per remaining distinct label, 3) random extras.
    labels = (["fall"] if "fall" in by_label else []) + [lbl for lbl in by_label if lbl != "fall"]
    for lbl in labels:
        if len(chosen) >= n_examples:
            break
        chosen.append(int(rng.choice(by_label[lbl])))

    if len(chosen) < n_examples:
        pool = [i for i in range(len(dataset.video_segments)) if i not in set(chosen)]
        rng.shuffle(pool)
        chosen.extend(pool[: n_examples - len(chosen)])

    return chosen[:n_examples]


def compose_config(group: str, num_frames: int, size: int, mode: str, fps: float):
    """Compose and validate the inference config for a single dataset group."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="inference_config",
            overrides=[
                f"dataset={group}",
                f"num_frames={num_frames}",
                f"data.size={size}",
                f"data.mode={mode}",
                f"model_fps={fps}",
            ],
        )
    return from_dictconfig(cfg)


def save_frame_strip(
    video,
    num_frames: int,
    figsize: tuple[float, float],
    out_path: Path,
    padding: int,
    formats: list[str],
    grid_cols: int | None = None,
) -> None:
    """Render a ``(T, C, H, W)`` uint8 tensor as a frame grid and save in ``formats``.

    ``padding`` is the gap (in pixels) placed between and around frames; the gaps and the
    figure background are transparent so the strip drops cleanly onto any slide colour.
    """
    nrow = grid_cols or num_frames
    rgb = vutils.make_grid(video.float() / 255.0, nrow=nrow, padding=padding, pad_value=0.0)
    # Alpha = 1 inside frames, 0 in the padding gaps (built with the same grid geometry).
    mask = vutils.make_grid(
        torch.ones_like(video).float(), nrow=nrow, padding=padding, pad_value=0.0
    )
    rgba = torch.cat([rgb, mask[:1]], dim=0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba.permute(1, 2, 0))
    ax.axis("off")
    for fmt in formats:
        fig.savefig(
            out_path.with_suffix(f".{fmt}"), bbox_inches="tight", pad_inches=0, transparent=True
        )
    plt.close(fig)


def render_dataset(
    group: str,
    args: argparse.Namespace,
    figsize: tuple[float, float],
    output_dir: Path,
    index_overrides: dict[str, list[int]],
) -> None:
    """Load one dataset group and write its example frame strips."""
    basename = group.rsplit("/", 1)[-1]
    config = compose_config(group, args.num_frames, args.size, args.mode, args.fps)

    seed = config.data.seed if args.seed is None else args.seed
    multi = get_video_datasets(
        config=config,
        mode=config.data.mode,
        return_individual=True,
        split=config.data.split,
        size=config.data.size,
        max_size=config.data.max_size,
        seed=config.data.seed,
    )
    dataset = next(iter(multi["individual"].values()))

    if basename in index_overrides:
        indices = index_overrides[basename]
    else:
        indices = pick_example_indices(
            dataset, np.random.default_rng(seed), args.num_examples, args.labels
        )

    for idx in indices:
        sample = dataset[idx]
        label = sample["label_str"]
        out_path = output_dir / f"{basename}_{label}_{idx}"
        save_frame_strip(
            sample["video"],
            args.num_frames,
            figsize,
            out_path,
            args.padding,
            args.formats,
            args.grid_cols,
        )
        logger.info("Wrote %s (label=%s, idx=%d)", out_path, label, idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS, help="Hydra dataset groups to render."
    )
    parser.add_argument(
        "--num-frames", type=int, default=4, help="Frames sampled per segment (default: 4)."
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=None,
        help="Frame grid columns per segment. Default renders all frames in one row.",
    )
    parser.add_argument(
        "--size", type=int, default=768, help="Frame resize/crop size (default: 768)."
    )
    parser.add_argument(
        "--span",
        type=float,
        default=2.0,
        help="Clip duration in seconds to span; fps is derived to hold this (default: 2.0).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Sampling fps (model_fps); overrides --span. Default: derived from --span.",
    )
    parser.add_argument("--mode", default="test", help="Dataset split mode (default: test).")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=2,
        help="Diverse segments to render per dataset when --indices is not given (default: 2).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=16,
        help="White gap in px between/around frames (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random segment selection (default: config seed).",
    )
    parser.add_argument(
        "--indices", nargs="+", default=None, help="Per-group index overrides, e.g. 'oops=12,45'."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Pick --num-examples segments per given label (e.g. 'fall fallen') instead of the "
        "default diverse selection.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png"],
        help="Output formats to save (default: pdf png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Raster dpi for saved figures; overrides the publication default (300).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.fps is None:
        # Hold a fixed temporal span: clip spans (num_frames - 1) / fps seconds.
        args.fps = (args.num_frames - 1) / args.span if args.num_frames > 1 else 1.0
    logger.info(
        "Sampling %d frames at %.3f fps (~%.2fs span)",
        args.num_frames,
        args.fps,
        (args.num_frames - 1) / args.fps if args.num_frames > 1 else 0.0,
    )

    set_publication_rc_defaults(use_tex=True, target="thesis")
    if args.dpi is not None:
        plt.rcParams["savefig.dpi"] = args.dpi
    width, _ = compute_publication_figsize(target="thesis", width_fraction=1.0)
    grid_cols = args.grid_cols or args.num_frames
    grid_rows = int(np.ceil(args.num_frames / grid_cols))
    figsize = (width, width * grid_rows / grid_cols)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    index_overrides = parse_indices(args.indices)

    for group in args.datasets:
        try:
            render_dataset(group, args, figsize, output_dir, index_overrides)
        except Exception:
            logger.exception("Failed to render dataset group '%s'; skipping", group)


if __name__ == "__main__":
    main()
