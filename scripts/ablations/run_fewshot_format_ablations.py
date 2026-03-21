#!/usr/bin/env python
"""Run few-shot format ablation experiments.

Sweeps over:
    - prompt.fewshot_preamble: [system, user]
    - prompt.fewshot_response: [inline, assistant]

Produces 4 combinations per model by default (2 preambles × 2 responses).

Base experiment config: experiment=fewshot

W&B tags: [ablation, fewshot_format, preamble-<p>, response-<r>]

Usage examples:
    # Dry-run to preview all commands
    python scripts/ablations/run_fewshot_format_ablations.py --dry-run

    # Run with specific model
    python scripts/ablations/run_fewshot_format_ablations.py --model qwenvl --params 7B

    # Sweep only a subset
    python scripts/ablations/run_fewshot_format_ablations.py --preambles system --responses inline assistant

    # Resume from experiment N (0-indexed)
    python scripts/ablations/run_fewshot_format_ablations.py --start-from 2
"""

import argparse
import subprocess

FEWSHOT_PREAMBLES = ["system", "user"]
FEWSHOT_RESPONSES = ["inline", "assistant"]


# ---------------------------------------------------------------------------
# Experiment generation
# ---------------------------------------------------------------------------


def generate_experiment_configs(
    preambles: list[str],
    responses: list[str],
) -> list[dict]:
    """Generate the cross-product of preambles × responses."""
    configs: list[dict] = []
    for preamble in preambles:
        for response in responses:
            configs.append(
                {
                    "prompt.fewshot_preamble": preamble,
                    "prompt.fewshot_response": response,
                }
            )
    return configs


# ---------------------------------------------------------------------------
# Tag & command building
# ---------------------------------------------------------------------------


def build_tags(config: dict) -> list[str]:
    """Build descriptive W&B tags from a config dict."""
    preamble = config["prompt.fewshot_preamble"]
    response = config["prompt.fewshot_response"]
    return [
        "ablation",
        "fewshot_format",
        f"preamble-{preamble}",
        f"response-{response}",
    ]


def build_command(config: dict, model: str = "qwenvl", params: str = "8B") -> list[str]:
    """Build the Hydra CLI command for a single experiment."""
    cmd = [
        "python",
        "scripts/vllm_inference.py",
        "experiment=fewshot",
        f"prompt.fewshot_preamble={config['prompt.fewshot_preamble']}",
        f"prompt.fewshot_response={config['prompt.fewshot_response']}",
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


def _describe_config(config: dict, model: str) -> str:
    """Return a short human-readable description of the config."""
    preamble = config["prompt.fewshot_preamble"]
    response = config["prompt.fewshot_response"]
    return f"model={model}, preamble={preamble}, response={response}"


def main():
    parser = argparse.ArgumentParser(
        description="Run few-shot format ablation experiments (preamble × response sweep)"
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
        nargs="+",
        default=["qwenvl"],
        help="Model name(s) (default: qwenvl)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="8B",
        help="Model parameter size applied to all models (default: 8B)",
    )
    parser.add_argument(
        "--preambles",
        type=str,
        nargs="+",
        default=FEWSHOT_PREAMBLES,
        choices=FEWSHOT_PREAMBLES,
        help=f"fewshot_preamble values to sweep (default: all {len(FEWSHOT_PREAMBLES)})",
    )
    parser.add_argument(
        "--responses",
        type=str,
        nargs="+",
        default=FEWSHOT_RESPONSES,
        choices=FEWSHOT_RESPONSES,
        help=f"fewshot_response values to sweep (default: all {len(FEWSHOT_RESPONSES)})",
    )
    args = parser.parse_args()

    configs = generate_experiment_configs(
        preambles=args.preambles,
        responses=args.responses,
    )

    # Flat list: models vary slowest
    experiments: list[tuple[str, dict]] = [
        (model, config) for model in args.model for config in configs
    ]

    print(
        f"Total experiments: {len(experiments)} "
        f"({len(args.model)} model(s) × {len(configs)} config(s))"
    )
    print()

    print("Experiment plan:")
    for i, (model, config) in enumerate(experiments):
        marker = "(skip)" if i < args.start_from else ""
        print(f"  {i + 1:>3}. {_describe_config(config, model):65s} {marker}")
    print()

    if args.start_from > 0:
        print(f"Resuming from experiment {args.start_from + 1}")

    for i, (model, config) in enumerate(experiments[args.start_from :], start=args.start_from):
        print(f"\n{'=' * 60}")
        print(f"Experiment {i + 1}/{len(experiments)}: {_describe_config(config, model)}")
        print(f"Config: {config}")
        print(f"{'=' * 60}")
        run_experiment(config, args.dry_run, model=model, params=args.params)

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("All experiments completed!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
