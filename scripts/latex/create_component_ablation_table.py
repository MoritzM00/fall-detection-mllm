#!/usr/bin/env python
"""Generate a LaTeX table for one-at-a-time prompt component ablation results.

Fetches runs from W&B that were produced by ``run_component_ablations.py``
(tagged with ``ablation`` + ``component``) and renders a single-model table
with one section per prompt component (Role, Task, Labels, Definitions).

Usage:
    python scripts/latex/create_component_ablation_table.py
    python scripts/latex/create_component_ablation_table.py --model-tag qwen --dataset OOPS --split cs
"""

import argparse
import logging
from typing import Any

import wandb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Default configuration
# ============================================================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-ablations"

# Baseline config values (must match run_component_ablations.py BASELINE)
BASELINE = {
    "output_format": "text",
    "role_variant": None,
    "task_variant": "standard",
    "labels_variant": "bulleted",
    "definitions_variant": None,
}

PROMPT_CONFIG_FIELDS = tuple(BASELINE.keys())

# Component sweeps — defines table sections and row ordering.
# Each entry: (section_label, config_key, [variant_values])
# The variant list order determines the row order within each section.
COMPONENT_SECTIONS = [
    ("Role", "role_variant", [None, "standard", "specialized", "video_specialized"]),
    ("Task", "task_variant", ["standard", "extended"]),
    ("Labels", "labels_variant", ["bulleted", "numbered", "grouped", "comma"]),
    ("Definitions", "definitions_variant", [None, "standard", "extended"]),
]

# Display names for variant values in the table
VARIANT_DISPLAY_NAMES: dict[str | None, str] = {
    None: "none",
    "standard": "standard",
    "specialized": "specialized",
    "video_specialized": "video spec.",
    "extended": "extended",
    "bulleted": "bulleted",
    "numbered": "numbered",
    "grouped": "grouped",
    "comma": "comma",
}

# Metric groups (suffix after ``{dataset}_{split}_``)
METRIC_GROUPS = [
    (
        "16-class",
        [
            ("balanced_accuracy", "BAcc"),
            ("accuracy", "Acc"),
            ("macro_f1", "F1"),
        ],
    ),
    (
        "Fall $\\Delta$",
        [
            ("fall_sensitivity", "Se"),
            ("fall_specificity", "Sp"),
            ("fall_f1", "F1"),
        ],
    ),
    (
        "Fallen $\\Delta$",
        [
            ("fallen_sensitivity", "Se"),
            ("fallen_specificity", "Sp"),
            ("fallen_f1", "F1"),
        ],
    ),
    (
        "Fall $\\cup$ Fallen",
        [
            ("fall_union_fallen_sensitivity", "Se"),
            ("fall_union_fallen_specificity", "Sp"),
            ("fall_union_fallen_f1", "F1"),
        ],
    ),
]

EXPECTED_RUNS = 10  # unique experiments from run_component_ablations.py


# ============================================================================
# W&B data fetching
# ============================================================================


def _config_key(prompt_config: dict) -> tuple:
    """Build a hashable key from the prompt config fields we care about."""
    return (
        prompt_config.get("output_format"),
        prompt_config.get("role_variant"),
        prompt_config.get("task_variant"),
        prompt_config.get("labels_variant"),
        prompt_config.get("definitions_variant"),
    )


def _lowercase_dict_keys(value: Any) -> Any:
    """Recursively lowercase string dictionary keys."""
    if isinstance(value, dict):
        return {
            (key.lower() if isinstance(key, str) else key): _lowercase_dict_keys(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_lowercase_dict_keys(item) for item in value]
    return value


def _normalize_config_value(value: Any) -> Any:
    """Normalize prompt config values for matching against the baseline grid."""
    if not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return normalized


def _extract_prompt_config(raw_config: Any) -> dict[str, Any]:
    """Extract prompt config from nested or flattened W&B config payloads."""
    normalized_config = _lowercase_dict_keys(raw_config)
    if not isinstance(normalized_config, dict):
        return dict(BASELINE)

    prompt_config: dict[str, Any] = {}

    nested_prompt = normalized_config.get("prompt")
    if isinstance(nested_prompt, dict):
        prompt_config.update(nested_prompt)

    for field in PROMPT_CONFIG_FIELDS:
        flat_key = f"prompt.{field}"
        if flat_key in normalized_config:
            prompt_config[field] = normalized_config[flat_key]

    return {
        field: _normalize_config_value(prompt_config.get(field, BASELINE[field]))
        for field in PROMPT_CONFIG_FIELDS
    }


def _summary_get_metric(summary: Any, metric_key: str) -> float | None:
    """Fetch a metric from W&B summary, tolerating lowercased summary keys."""
    for candidate in (metric_key, metric_key.lower()):
        val = summary.get(candidate)
        if val is not None:
            return val
    return None


def get_metric_groups(include_fall_union_fallen: bool) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return metric groups, optionally omitting the Fall union Fallen block."""
    if include_fall_union_fallen:
        return METRIC_GROUPS
    return METRIC_GROUPS[:-1]


def get_metric_suffixes(include_fall_union_fallen: bool) -> list[str]:
    """Flatten metric suffixes for the active table layout."""
    metric_groups = get_metric_groups(include_fall_union_fallen)
    return [suffix for _, metrics in metric_groups for suffix, _ in metrics]


def fetch_component_ablation_runs(
    api,
    entity: str,
    project: str,
    model_tag: str,
    dataset: str,
    split: str,
) -> dict[tuple, dict[str, float | None]]:
    """Fetch component ablation runs from W&B.

    Returns:
        Dict keyed by (output_format, role_variant, task_variant,
        labels_variant, definitions_variant) with values being metric dicts.
    """
    metric_suffixes = get_metric_suffixes(include_fall_union_fallen=True)
    metric_keys = [f"{dataset}_{split}_{suffix}" for suffix in metric_suffixes]

    filters = {
        "$and": [
            {"tags": "ablation"},
            {"tags": "component"},
            {"tags": model_tag},
            {"config.data.mode": "test"},
        ]
    }

    runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at")

    results: dict[tuple, dict[str, float | None]] = {}
    for run in runs:
        logger.info(f"Processing run: {run.name} ({run.id})")
        prompt_config = _extract_prompt_config(run.config)
        key = _config_key(prompt_config)

        # Skip duplicates — keep the most recent run for each config
        if key in results:
            continue

        summary = run.summary
        metrics: dict[str, float | None] = {}
        for mk, suffix in zip(metric_keys, metric_suffixes):
            val = _summary_get_metric(summary, mk)
            metrics[suffix] = val * 100 if val is not None else None

        results[key] = metrics

    return results


# ============================================================================
# LaTeX table generation
# ============================================================================


def _make_config_key_for_variant(component_key: str, variant_value) -> tuple:
    """Build the full config key tuple for a single-component variant.

    Starts from BASELINE and overrides *component_key* with *variant_value*.
    """
    cfg = dict(BASELINE)
    cfg[component_key] = variant_value
    return (
        cfg["output_format"],
        cfg["role_variant"],
        cfg["task_variant"],
        cfg["labels_variant"],
        cfg["definitions_variant"],
    )


def find_best_in_section(
    data: dict[tuple, dict[str, float | None]],
    keys: list[tuple],
    metric_suffixes: list[str],
) -> dict[str, float | None]:
    """Find the best (max) value for each metric across a section's rows."""
    bests: dict[str, float | None] = {}
    for suffix in metric_suffixes:
        values = [data.get(k, {}).get(suffix) for k in keys]
        values = [v for v in values if v is not None]
        bests[suffix] = max(values) if values else None
    return bests


def format_value(val: float | None, best_val: float | None) -> str:
    """Format a metric value, bolding if it equals the section best."""
    if val is None:
        return "--"
    val_rounded = round(val, 1)
    formatted = f"{val_rounded:.1f}"
    if best_val is not None and round(best_val, 1) == val_rounded:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_latex_table(
    data: dict[tuple, dict[str, float | None]],
    include_fall_union_fallen: bool = True,
) -> str:
    """Generate the full LaTeX table string."""
    metric_groups = get_metric_groups(include_fall_union_fallen)
    metric_suffixes = get_metric_suffixes(include_fall_union_fallen)

    if include_fall_union_fallen:
        col_spec = "@{}l l rrr rrr rrr rrr@{}"
    else:
        col_spec = "@{}l l rrr rrr rrr@{}"

    group_headers = " &\n".join(
        f"\\multicolumn{{{len(metrics)}}}{{c}}{{{group_label}}}"
        for group_label, metrics in metric_groups
    )

    cmidrules: list[str] = []
    col_start = 3
    sub_headers: list[str] = []
    for _, metrics in metric_groups:
        col_end = col_start + len(metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
        sub_headers.extend(
            f"\\multicolumn{{1}}{{c}}{{{display_name}}}" for _, display_name in metrics
        )

    sub_header_row = " & ".join(sub_headers)

    rows: list[str] = []

    for sec_idx, (section_label, config_key, variants) in enumerate(COMPONENT_SECTIONS):
        # Build config keys for all variants in this section
        section_keys = [_make_config_key_for_variant(config_key, v) for v in variants]
        bests = find_best_in_section(data, section_keys, metric_suffixes)

        for row_idx, (variant, cfg_key) in enumerate(zip(variants, section_keys)):
            # Component column: multirow on first row, empty on subsequent
            if row_idx == 0:
                comp_col = f"\\multirow{{{len(variants)}}}{{*}}{{{section_label}}}"
            else:
                comp_col = ""

            # Variant display name with baseline marker
            display_name = VARIANT_DISPLAY_NAMES.get(variant, str(variant))
            if variant == BASELINE.get(config_key):
                display_name += "$^{\\ast}$"

            # Metric values
            run_metrics = data.get(cfg_key, {})
            metric_cells = []
            for suffix in metric_suffixes:
                val = run_metrics.get(suffix)
                metric_cells.append(format_value(val, bests[suffix]))

            metrics_str = " & ".join(metric_cells)
            rows.append(f"        {comp_col} & {display_name} & {metrics_str} \\\\")

        # Add midrule between sections (but not after the last one)
        if sec_idx < len(COMPONENT_SECTIONS) - 1:
            rows.append("        \\midrule")

    rows_str = "\n".join(rows)

    union_clause = (
        " and combined Fall/Fallen binary metrics"
        if include_fall_union_fallen
        else " and binary metrics for the Fall and Fallen classes"
    )

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
    \\centering
    \\resizebox{{\\columnwidth}}{{!}}{{%
    \\begin{{tabular}}{{{col_spec}}}
        \\toprule
        \\multirow{{2}}{{*}}{{Component}} & \\multirow{{2}}{{*}}{{Variant}} &
{group_headers} \\\\
        {" ".join(cmidrules)}
        & & {sub_header_row} \\\\
        \\midrule

{rows_str}

        \\bottomrule
    \\end{{tabular}}}}
    \\caption{{\\textbf{{Prompt Component Ablation.}} One-at-a-time ablation over prompt components (role, task instruction, label formatting, and class definitions) with output format fixed to free text. We report 16-class{union_clause}. Baseline variants are marked with $\\ast$. Best results per component section are \\textbf{{bolded}}.}}
    \\label{{tab:component_ablation}}
\\end{{table}}
\\endgroup"""

    return table


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table for prompt component ablation results"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=ENTITY,
        help=f"W&B entity (default: {ENTITY})",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=PROJECT,
        help=f"W&B project (default: {PROJECT})",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="qwen",
        help="W&B model tag to filter runs (default: qwen)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="OOPS",
        help="Dataset name as it appears in W&B metric keys (default: OOPS)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="cs",
        help="Dataset split (default: cs)",
    )
    parser.add_argument(
        "--no-fall-union-fallen",
        dest="include_fall_union_fallen",
        action="store_false",
        help="Disable the combined Fall/Fallen metric columns",
    )
    parser.set_defaults(include_fall_union_fallen=True)
    args = parser.parse_args()

    api = wandb.Api()  # pyright: ignore[reportAttributeAccessIssue]

    print(f"Fetching component ablation runs from {args.entity}/{args.project}...")
    print(f"  Model tag: {args.model_tag}")
    print(f"  Metrics prefix: {args.dataset}_{args.split}_*")
    print(f"  Include Fall union Fallen: {'yes' if args.include_fall_union_fallen else 'no'}")
    print()

    data = fetch_component_ablation_runs(
        api,
        entity=args.entity,
        project=args.project,
        model_tag=args.model_tag,
        dataset=args.dataset,
        split=args.split,
    )

    print(f"Found {len(data)} unique config runs")
    if len(data) < EXPECTED_RUNS:
        print(
            f"Warning: Expected {EXPECTED_RUNS} runs for a complete component ablation, "
            f"got {len(data)}. Some rows may show '--'."
        )
    print()

    latex_table = generate_latex_table(
        data,
        include_fall_union_fallen=args.include_fall_union_fallen,
    )
    print(latex_table)


if __name__ == "__main__":
    main()
