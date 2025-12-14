def format_subgroup_latex_table(
    subgroup_results: dict[str, dict[str, dict[str, float]]],
    subgroup_key: str,
    metric_keys: list[str] = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "fall_sensitivity",
        "fall_f1",
    ],
) -> str:
    """
    Format subgroup results as a LaTeX table.

    Args:
        subgroup_results: Results from compute_subgroup_metrics()
        subgroup_key: Which subgroup dimension to format (e.g., "age_group", "ethnicity")
        metric_keys: Which metrics to include in the table

    Returns:
        LaTeX table as string
    """
    if subgroup_key not in subgroup_results:
        return f"% No data for subgroup: {subgroup_key}\n"

    data = subgroup_results[subgroup_key]

    if not data:
        return f"% No categories found for subgroup: {subgroup_key}\n"

    # Start table
    num_metrics = len(metric_keys)
    col_spec = "l" + "r" * num_metrics + "r"  # +1 for sample count

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Performance by {subgroup_key.replace('_', ' ').title()}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header with aligned ampersands
    header_label = subgroup_key.replace("_", " ").title()
    header_row = f"{header_label:<25}"
    for m in metric_keys:
        header_row += f" & {m.replace('_', ' ').title()}"
    header_row += " & N \\\\"
    lines.append(header_row)
    lines.append("\\midrule")

    # Data rows with aligned ampersands
    for category in sorted(data.keys()):
        metrics = data[category]
        category_label = category.replace("_", " ").title()

        # Start row with left-padded category name
        row = f"{category_label:<25}"

        # Add metric values
        for metric_key in metric_keys:
            value = metrics.get(metric_key, 0.0)
            # Format as percentage with 1 decimal place
            row += f" & {value * 100:.1f}"

        # Add sample count
        sample_count = metrics.get("sample_count", 0)
        row += f" & {sample_count} \\\\"

        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)
