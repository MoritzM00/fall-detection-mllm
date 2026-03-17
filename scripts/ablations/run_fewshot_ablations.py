#!/usr/bin/env python
"""Run few-shot ablation experiments sweeping over num_shots and optionally shot_selection.

Sweeps:
    - prompt.num_shots: [1, 2, 3, 5] (configurable via --shot-values)
    - prompt.shot_selection: [random, balanced, similarity] (optional, via --sweep-selection)

Base experiment config: experiment=fewshot
For similarity selection: experiment=fewshot_similarity (sets up embeddings dir/model).

W&B tags: [ablation, fewshot, shots-<N>, selection-<strategy>]
"""

import argparse
import subprocess

DEFAULT_SHOT_VALUES = [1, 2, 3, 5]
SELECTION_STRATEGIES = ["random", "balanced", "similarity"]
DELIMITER_VALUES = ["true", "false"]
ORDERING_VALUES = ["ascending", "descending", "random"]


# ---------------------------------------------------------------------------
# Experiment generation
# ---------------------------------------------------------------------------


def _config_key(config: dict) -> tuple:
    """Return a hashable key for deduplication."""
    return tuple(sorted(config.items()))


def generate_experiment_configs(
    shot_values: list[int],
    selection_values: list[str],
    delimiter_values: list[str],
    ordering_values: list[str],
) -> list[dict]:
    """Generate deduplicated experiment configs.

    Produces the cross-product of shot_values x selection_values x delimiter_values x ordering_values.
    """
    seen: set[tuple] = set()
    configs: list[dict] = []

    for num_shots in shot_values:
        for selection in selection_values:
            for use_delimiters in delimiter_values:
                for ordering in ordering_values:
                    config = {
                        "prompt.num_shots": num_shots,
                        "prompt.shot_selection": selection,
                        "prompt.use_delimiters": use_delimiters,
                        "prompt.exemplar_ordering": ordering,
                    }
                    key = _config_key(config)
                    if key not in seen:
                        seen.add(key)
                        configs.append(config)

    return configs


# ---------------------------------------------------------------------------
# Tag & command building
# ---------------------------------------------------------------------------


def build_tags(config: dict) -> list[str]:
    """Build descriptive W&B tags from a config dict."""
    num_shots = config["prompt.num_shots"]
    selection = config["prompt.shot_selection"]
    use_delimiters = config["prompt.use_delimiters"]
    ordering = config["prompt.exemplar_ordering"]
    return [
        "ablation",
        "fewshot",
        f"shots-{num_shots}",
        f"selection-{selection}",
        f"delimiters-{use_delimiters}",
        f"ordering-{ordering}",
    ]


def _experiment_name(config: dict) -> str:
    """Return the Hydra experiment name for this config."""
    if config["prompt.shot_selection"] == "similarity":
        return "fewshot_similarity"
    return "fewshot"


def build_command(config: dict, model: str = "qwenvl", params: str = "8B") -> list[str]:
    """Build the Hydra CLI command for a single experiment."""
    experiment = _experiment_name(config)
    cmd = [
        "python",
        "scripts/vllm_inference.py",
        f"experiment={experiment}",
        f"prompt.num_shots={config['prompt.num_shots']}",
        f"prompt.shot_selection={config['prompt.shot_selection']}",
        f"prompt.use_delimiters={config['prompt.use_delimiters']}",
        f"prompt.exemplar_ordering={config['prompt.exemplar_ordering']}",
        f"model={model}",
        f"model.params={params}",
    ]

    tags = build_tags(config)
    cmd.append(f"wandb.tags={tags}")

    return cmd


def format_command_for_display(cmd: list[str]) -> str:
    """Format a command list for human-readable display."""
    parts = []
    for part in cmd:
        if part.startswith("wandb.tags="):
            parts.append(f'"{part}"')
        else:
            parts.append(part)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_experiment(config: dict, dry_run: bool = False, model: str = "qwenvl", params: str = "8B"):
    """Run a single experiment (or print the command in dry-run mode)."""
    cmd = build_command(config, model=model, params=params)

    if dry_run:
        print(f"[DRY-RUN] {format_command_for_display(cmd)}\n")
        return

    print(f"Running: {format_command_for_display(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with exit code {e.returncode}")
        raise


def _describe_config(config: dict) -> str:
    """Return a short human-readable description of the config."""
    num_shots = config["prompt.num_shots"]
    selection = config["prompt.shot_selection"]
    use_delimiters = config["prompt.use_delimiters"]
    ordering = config["prompt.exemplar_ordering"]
    return f"shots={num_shots}, selection={selection}, delimiters={use_delimiters}, ordering={ordering}"


def main():
    parser = argparse.ArgumentParser(
        description="Run few-shot ablation experiments sweeping over num_shots and shot_selection"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip first N experiments (for resuming, 0-indexed)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwenvl",
        help="Model name for experiments (default: qwenvl)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="8B",
        help="Model parameters for experiments (default: 8B)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=DEFAULT_SHOT_VALUES,
        help=f"num_shots values to sweep (default: {DEFAULT_SHOT_VALUES})",
    )
    parser.add_argument(
        "--selection",
        type=str,
        nargs="+",
        default=["balanced"],
        choices=SELECTION_STRATEGIES,
        help="shot_selection strategies to sweep (default: balanced)",
    )
    parser.add_argument(
        "--delim",
        type=str,
        nargs="+",
        default=["true"],
        choices=DELIMITER_VALUES,
        help="use_delimiters values to sweep (default: true)",
    )
    parser.add_argument(
        "--order",
        type=str,
        nargs="+",
        default=["ascending"],
        choices=ORDERING_VALUES,
        help="exemplar_ordering values to sweep (default: ascending)",
    )
    args = parser.parse_args()

    configs = generate_experiment_configs(
        shot_values=args.shots,
        selection_values=args.selection,
        delimiter_values=args.delim,
        ordering_values=args.order,
    )
    print(f"Total unique experiments: {len(configs)}")
    print()

    print("Experiment plan:")
    for i, config in enumerate(configs):
        marker = "(skip)" if i < args.start_from else ""
        print(f"  {i + 1:>3}. {_describe_config(config):40s} {marker}")
    print()

    if args.start_from > 0:
        print(f"Resuming from experiment {args.start_from + 1}")

    for i, config in enumerate(configs[args.start_from :], start=args.start_from):
        print(f"\n{'=' * 60}")
        print(f"Experiment {i + 1}/{len(configs)}: {_describe_config(config)}")
        print(f"Config: {config}")
        print(f"{'=' * 60}")
        run_experiment(config, args.dry_run, model=args.model, params=args.params)

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("All experiments completed!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
