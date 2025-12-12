"""
Visualization utilities for displaying evaluation metrics using plotext.

This module provides functions for creating terminal-based visualizations of
evaluation results, including bar charts and performance statistics.
"""

import re
from typing import Any

import plotext as plt

from infreqact.data.video_dataset import idx2label


def visualize_evaluation_results(
    results: dict[str, Any], title: str = "Evaluation Results", width: int = 100
):
    """
    Create beautiful terminal visualizations for evaluation metrics using plotext.

    Args:
        results: Dictionary containing evaluation metrics
        title: Title for the visualization
        width: Width of the plots in characters
    """
    # Extract prefix (e.g., "eval_combined_" or "eval_")
    prefix = None
    for key in results:
        if key.startswith("eval_"):
            match = re.match(r"(eval_(?:\w+_)?)", key)
            if match:
                prefix = match.group(1)
                break

    if prefix is None:
        prefix = ""

    # 1. Overall Performance Metrics
    _plot_overall_metrics(results, prefix, title, width)

    # 2. Fall Detection Metrics (Binary)
    _plot_fall_detection_metrics(results, prefix, width)

    # 3. Per-Class Metrics (Table + Graphs)
    _plot_per_class_f1_scores(results, prefix, width)

    # 4. Per-Class Confusion Matrices
    _plot_confusion_matrices(results, prefix, width)

    # 5. Class Distribution
    _plot_class_distribution(results, prefix, width)

    # 6. Performance Statistics
    _print_performance_statistics(results, prefix, width)

    print("\n" + "=" * width + "\n")


def _plot_overall_metrics(results: dict[str, Any], prefix: str, title: str, width: int):
    """Plot overall performance metrics."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    overall_metrics = {}
    metric_names = ["loss", "accuracy", "balanced_accuracy", "macro_f1"]

    for metric in metric_names:
        key = f"{prefix}{metric}"
        if key in results:
            overall_metrics[metric.replace("_", " ").title()] = results[key]

    if overall_metrics:
        print("\n" + "=" * width)
        print(f"{title} - Overall Performance".center(width))
        print("=" * width)

        # Create Rich table for overall metrics
        table = Table(
            title="Overall Performance Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        # Add sample count if available
        sample_count_key = f"{prefix}sample_count"
        if sample_count_key in results:
            table.add_row("Sample Count", f"{int(results[sample_count_key]):,}")

        for metric_name, metric_value in overall_metrics.items():
            if "Loss" in metric_name:
                table.add_row(metric_name, f"{metric_value:.4f}")
            else:
                table.add_row(metric_name, f"{metric_value * 100:.2f}%")

        console.print(table)

        # Plot bar chart
        labels = list(overall_metrics.keys())
        values = [v * 100 if "Loss" not in k else v for k, v in overall_metrics.items()]

        print("\n" + "─" * width)
        plt.simple_bar(labels, values, width=width, title="Overall Metrics (%)")
        plt.show()
        plt.clear_figure()


def _plot_fall_detection_metrics(results: dict[str, Any], prefix: str, width: int):
    """Plot fall detection metrics (binary classification)."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Organize fall metrics by category
    fall_categories = {
        "Fall": ["fall_sensitivity", "fall_specificity", "fall_f1"],
        "Fallen": ["fallen_sensitivity", "fallen_specificity", "fallen_f1"],
        "Fall∪Fallen": [
            "fall_union_fallen_sensitivity",
            "fall_union_fallen_specificity",
            "fall_union_fallen_f1",
        ],
    }

    # Check if we have any fall metrics
    has_fall_metrics = False
    for metrics in fall_categories.values():
        for metric in metrics:
            if f"{prefix}{metric}" in results:
                has_fall_metrics = True
                break
        if has_fall_metrics:
            break

    if has_fall_metrics:
        print("\n" + "=" * width)
        print("Fall Detection Metrics (Binary Classification)".center(width))
        print("=" * width)

        # Create Rich table
        table = Table(
            title="Binary Fall Detection Performance",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Sensitivity (%)", justify="right", style="yellow")
        table.add_column("Specificity (%)", justify="right", style="magenta")
        table.add_column("F1 Score (%)", justify="right", style="blue")

        for category_name, metric_keys in fall_categories.items():
            sens_key = f"{prefix}{metric_keys[0]}"
            spec_key = f"{prefix}{metric_keys[1]}"
            f1_key = f"{prefix}{metric_keys[2]}"

            if sens_key in results or spec_key in results or f1_key in results:
                table.add_row(
                    category_name,
                    f"{results.get(sens_key, 0.0) * 100:.2f}" if sens_key in results else "N/A",
                    f"{results.get(spec_key, 0.0) * 100:.2f}" if spec_key in results else "N/A",
                    f"{results.get(f1_key, 0.0) * 100:.2f}" if f1_key in results else "N/A",
                )

        console.print(table)

        # Plot bar chart
        fall_metrics = {}
        fall_metric_names = [
            ("fall_sensitivity", "Fall Sensitivity"),
            ("fall_specificity", "Fall Specificity"),
            ("fall_f1", "Fall F1"),
            ("fallen_sensitivity", "Fallen Sensitivity"),
            ("fallen_specificity", "Fallen Specificity"),
            ("fallen_f1", "Fallen F1"),
            ("fall_union_fallen_sensitivity", "Fall∪Fallen Sens"),
            ("fall_union_fallen_specificity", "Fall∪Fallen Spec"),
            ("fall_union_fallen_f1", "Fall∪Fallen F1"),
        ]

        for metric_key, display_name in fall_metric_names:
            key = f"{prefix}{metric_key}"
            if key in results:
                fall_metrics[display_name] = results[key]

        labels = list(fall_metrics.keys())
        values = [v * 100 for v in fall_metrics.values()]

        print("\n" + "─" * width)
        plt.simple_bar(labels, values, width=width, title="Fall Detection Performance (%)")
        plt.show()
        plt.clear_figure()


def _plot_per_class_f1_scores(results: dict[str, Any], prefix: str, width: int):
    """Plot per-class metrics (accuracy, sensitivity, F1) with both graphs and tables."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    class_names = [
        "walk",
        "fall",
        "fallen",
        "sit_down",
        "sitting",
        "lie_down",
        "lying",
        "stand_up",
        "standing",
        "other",
        # WanFall-specific classes
        "kneel_down",
        "kneeling",
        "squat_down",
        "squatting",
        "crawl",
        "jump",
    ]

    # Collect all per-class metrics
    class_metrics = {}
    for class_name in class_names:
        key_acc = f"{prefix}{class_name}_accuracy"
        key_prec = f"{prefix}{class_name}_precision"
        key_sens = f"{prefix}{class_name}_sensitivity"
        key_spec = f"{prefix}{class_name}_specificity"
        key_npv = f"{prefix}{class_name}_npv"
        key_f1 = f"{prefix}{class_name}_f1"
        key_true_dist = f"{prefix}true_dist_{class_name}"
        key_pred_dist = f"{prefix}pred_dist_{class_name}"
        key_sample_count = f"{prefix}sample_count_{class_name}"

        # Only include classes that have metrics
        if key_f1 in results or key_acc in results or key_sens in results:
            true_dist = results.get(key_true_dist, 0.0)
            pred_dist = results.get(key_pred_dist, 0.0)
            deviation = pred_dist - true_dist

            class_metrics[class_name] = {
                "accuracy": results.get(key_acc, 0.0),
                "precision": results.get(key_prec, 0.0),
                "sensitivity": results.get(key_sens, 0.0),
                "specificity": results.get(key_spec, 0.0),
                "npv": results.get(key_npv, 0.0),
                "f1": results.get(key_f1, 0.0),
                "true_dist": true_dist,
                "pred_dist": pred_dist,
                "deviation": deviation,
                "sample_count": results.get(key_sample_count, 0),
            }

    if class_metrics:
        print("\n" + "=" * width)
        print("Per-Class Metrics".center(width))
        print("=" * width)

        # Sort by F1 score for better visualization
        sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)

        # Create Rich table
        table = Table(
            title="Per-Class Performance Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Class", style="cyan", no_wrap=True)
        table.add_column("Samples", justify="right", style="white")
        table.add_column("Accuracy (%)", justify="right", style="green")
        table.add_column("Precision (%)", justify="right", style="bright_blue")
        table.add_column("Sensitivity (%)", justify="right", style="yellow")
        table.add_column("Specificity (%)", justify="right", style="magenta")
        table.add_column("NPV (%)", justify="right", style="bright_cyan")
        table.add_column("F1 Score (%)", justify="right", style="blue")
        table.add_column("True Dist (%)", justify="right", style="white")
        table.add_column("Pred Dist (%)", justify="right", style="white")
        table.add_column("Deviation", justify="right", style="white")

        for class_name, metrics in sorted_classes:
            display_name = class_name.replace("_", " ").title()

            # Color-code deviation based on magnitude
            deviation_pct = metrics["deviation"] * 100
            if abs(deviation_pct) < 1.0:
                deviation_style = "green"
            elif abs(deviation_pct) < 3.0:
                deviation_style = "yellow"
            else:
                deviation_style = "red"

            # Format deviation with sign
            deviation_str = f"{deviation_pct:+.1f}%"

            table.add_row(
                display_name,
                f"{int(metrics['sample_count']):,}",
                f"{metrics['accuracy'] * 100:.2f}",
                f"{metrics['precision'] * 100:.2f}",
                f"{metrics['sensitivity'] * 100:.2f}",
                f"{metrics['specificity'] * 100:.2f}",
                f"{metrics['npv'] * 100:.2f}",
                f"{metrics['f1'] * 100:.2f}",
                f"{metrics['true_dist'] * 100:.1f}",
                f"{metrics['pred_dist'] * 100:.1f}",
                f"[{deviation_style}]{deviation_str}[/{deviation_style}]",
            )

        console.print(table)

        # Plot metrics
        labels = [k.replace("_", " ").title() for k, v in sorted_classes]

        # Plot Accuracy
        print("\n" + "─" * width)
        acc_values = [v["accuracy"] * 100 for k, v in sorted_classes]
        plt.simple_bar(labels, acc_values, width=width, title="Per-Class Accuracy (%)")
        plt.show()
        plt.clear_figure()

        # Plot Precision
        print("\n" + "─" * width)
        prec_values = [v["precision"] * 100 for k, v in sorted_classes]
        plt.simple_bar(labels, prec_values, width=width, title="Per-Class Precision (%)")
        plt.show()
        plt.clear_figure()

        # Plot Sensitivity
        print("\n" + "─" * width)
        sens_values = [v["sensitivity"] * 100 for k, v in sorted_classes]
        plt.simple_bar(labels, sens_values, width=width, title="Per-Class Sensitivity (%)")
        plt.show()
        plt.clear_figure()

        # Plot Specificity
        print("\n" + "─" * width)
        spec_values = [v["specificity"] * 100 for k, v in sorted_classes]
        plt.simple_bar(labels, spec_values, width=width, title="Per-Class Specificity (%)")
        plt.show()
        plt.clear_figure()

        # Plot NPV
        print("\n" + "─" * width)
        npv_values = [v["npv"] * 100 for k, v in sorted_classes]
        plt.simple_bar(labels, npv_values, width=width, title="Per-Class NPV (%)")
        plt.show()
        plt.clear_figure()

        # Plot F1 scores
        print("\n" + "─" * width)
        f1_values = [v["f1"] * 100 for k, v in sorted_classes]
        plt.simple_bar(labels, f1_values, width=width, title="Per-Class F1 Scores (%)")
        plt.show()
        plt.clear_figure()


def _plot_class_distribution(results: dict[str, Any], prefix: str, width: int):
    """Plot true vs predicted class distribution with deviation highlighting."""
    class_names = [
        "walk",
        "fall",
        "fallen",
        "sit_down",
        "sitting",
        "lie_down",
        "lying",
        "stand_up",
        "standing",
        "other",
        # WanFall-specific classes
        "kneel_down",
        "kneeling",
        "squat_down",
        "squatting",
        "crawl",
        "jump",
    ]

    true_dist = {}
    pred_dist = {}

    for class_name in class_names:
        true_key = f"{prefix}true_dist_{class_name}"
        pred_key = f"{prefix}pred_dist_{class_name}"

        if true_key in results:
            true_dist[class_name.replace("_", " ").title()] = results[true_key]
        if pred_key in results:
            pred_dist[class_name.replace("_", " ").title()] = results[pred_key]

    # Only plot if we have both distributions
    if true_dist and pred_dist:
        print("\n" + "=" * width)
        print("Class Distribution: True vs Predicted".center(width))
        print("=" * width)

        # Get all classes present in either distribution
        all_classes = sorted(set(true_dist.keys()) | set(pred_dist.keys()))

        # Prepare data for plotting
        true_values = [true_dist.get(cls, 0) * 100 for cls in all_classes]
        pred_values = [pred_dist.get(cls, 0) * 100 for cls in all_classes]

        # Create simple multiple bar plot
        plt.simple_multiple_bar(
            all_classes,
            [true_values, pred_values],
            labels=["True", "Predicted"],
            width=width,
            title="Class Distribution (%)",
        )
        plt.show()
        plt.clear_figure()

        # Print deviation summary
        print("\nDistribution Deviations (Predicted - True):")
        print("-" * width)
        deviations = []
        for cls in all_classes:
            true_val = true_dist.get(cls, 0) * 100
            pred_val = pred_dist.get(cls, 0) * 100
            deviation = pred_val - true_val
            deviations.append((cls, deviation, true_val, pred_val))

        # Sort by absolute deviation
        deviations.sort(key=lambda x: abs(x[1]), reverse=True)

        for cls, dev, true_val, pred_val in deviations[:10]:  # Show top 10 deviations
            sign = "+" if dev >= 0 else ""
            print(
                f"  {cls:.<25} True: {true_val:5.1f}%  Pred: {pred_val:5.1f}%  Dev: {sign}{dev:5.1f}%"
            )


def _plot_confusion_matrices(results: dict[str, Any], prefix: str, width: int):
    """Plot per-class confusion matrices using plotext."""
    class_names = [
        "walk",
        "fall",
        "fallen",
        "sit_down",
        "sitting",
        "lie_down",
        "lying",
        "stand_up",
        "standing",
        "other",
        # WanFall-specific classes
        "kneel_down",
        "kneeling",
        "squat_down",
        "squatting",
        "crawl",
        "jump",
    ]

    # Check if we have the raw predictions and references (needed for confusion matrix)
    # These should be stored during evaluation
    if f"{prefix}predictions" not in results or f"{prefix}references" not in results:
        return  # Can't create confusion matrix without raw predictions

    import numpy as np

    predictions = np.array(results[f"{prefix}predictions"])
    references = np.array(results[f"{prefix}references"])
    unique_true = np.unique(references)

    print("\n" + "=" * width)
    print("Per-Class Confusion Matrices".center(width))
    print("=" * width)

    # Create confusion matrix for each class (one-vs-rest)
    for class_idx in unique_true:
        class_name = idx2label.get(class_idx, f"unknown_{class_idx}")

        # Skip if not in our list
        if class_name not in class_names:
            continue

        # Create binary predictions (this class vs all others)
        binary_pred = (predictions == class_idx).astype(int)
        binary_ref = (references == class_idx).astype(int)

        # Get metrics for this class
        key_prec = f"{prefix}{class_name}_precision"
        key_sens = f"{prefix}{class_name}_sensitivity"
        key_spec = f"{prefix}{class_name}_specificity"
        key_npv = f"{prefix}{class_name}_npv"

        precision = results.get(key_prec, 0.0) * 100
        sensitivity = results.get(key_sens, 0.0) * 100
        specificity = results.get(key_spec, 0.0) * 100
        npv = results.get(key_npv, 0.0) * 100

        # Plot using plotext cmatrix
        print(f"\n{class_name.replace('_', ' ').title()} (Class {class_idx} vs All Others)")
        print("─" * width)

        # Create actual and predicted lists for plotext
        # Note: plotext's cmatrix expects lists of class labels, not binary
        actual_labels = []
        predicted_labels = []
        for i in range(len(binary_ref)):
            actual_labels.append(1 if binary_ref[i] == 1 else 0)
            predicted_labels.append(1 if binary_pred[i] == 1 else 0)

        # Use plotext's confusion matrix
        plt.cmatrix(actual_labels, predicted_labels, labels=["Negative", "Positive"])
        plt.show()
        plt.clear_figure()

        # Print metrics summary
        print(f"  Precision: {precision:5.2f}% | Sensitivity: {sensitivity:5.2f}%")
        print(f"  Specificity: {specificity:5.2f}% | NPV: {npv:5.2f}%")
        print()


def _print_performance_statistics(results: dict[str, Any], prefix: str, width: int):
    """Print performance statistics (runtime, throughput, etc.)."""
    perf_metrics = {}
    perf_keys = [
        ("sample_count", "Samples"),
        ("runtime", "Runtime (s)"),
        ("samples_per_second", "Samples/s"),
    ]

    for metric_key, display_name in perf_keys:
        key = f"{prefix}{metric_key}"
        if key in results:
            perf_metrics[display_name] = results[key]

    if perf_metrics:
        print("\n" + "=" * width)
        print("Performance Statistics".center(width))
        print("=" * width)
        for name, value in perf_metrics.items():
            if isinstance(value, float):
                print(f"  {name:.<30} {value:.2f}")
            else:
                print(f"  {name:.<30} {value}")


def visualize_subgroup_results(
    subgroup_results: dict[str, dict[str, dict[str, float]]],
    dataset_name: str = "WanFall",
    width: int = 100,
):
    """
    Create terminal visualizations for subgroup evaluation metrics using horizontal bar charts.

    Args:
        subgroup_results: Dictionary from compute_subgroup_metrics containing:
                         {subgroup_key: {category: {metric: value}}}
        dataset_name: Name of the dataset (for title)
        width: Width of the plots in characters
    """
    if not subgroup_results:
        return

    print("\n" + "=" * width)
    print(f"{dataset_name} - Subgroup Evaluation Results".center(width))
    print("=" * width)

    # Metrics to visualize
    key_metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1", "Macro F1"),
        ("fall_sensitivity", "Fall Sensitivity"),
        ("fall_f1", "Fall F1"),
    ]

    for subgroup_key, categories in subgroup_results.items():
        if not categories:
            continue

        subgroup_title = subgroup_key.replace("_", " ").title()

        print(f"\n{subgroup_title}:")
        print("-" * width)

        # For each metric, create a horizontal bar chart
        for metric_key, metric_name in key_metrics:
            # Extract values for this metric across all categories
            category_names = []
            metric_values = []
            sample_counts = []

            for category in sorted(categories.keys()):
                metrics = categories[category]
                if metric_key in metrics:
                    category_names.append(category.replace("_", " ").title())
                    metric_values.append(metrics[metric_key] * 100)  # Convert to percentage
                    sample_counts.append(metrics.get("sample_count", 0))

            if metric_values:
                # Create horizontal bar chart
                print(f"\n  {metric_name}:")
                plt.simple_bar(
                    category_names,
                    metric_values,
                    width=width - 4,
                    title=f"{metric_name} by {subgroup_title} (%)",
                )
                plt.show()
                plt.clear_figure()

                # Print detailed stats
                print(f"  {'Category':<25} {'Value':>8} {'N':>8}")
                print("  " + "-" * 43)
                for cat, val, n in zip(category_names, metric_values, sample_counts):
                    print(f"  {cat:<25} {val:>7.1f}% {n:>8}")

    print("\n" + "=" * width + "\n")


def print_evaluation_plan(test_datasets: dict[str, Any]) -> int:
    """
    Print a summary of all evaluations that will be performed.

    Args:
        test_datasets: Dictionary containing combined and individual datasets

    Returns:
        Total number of evaluations to perform
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("TEST EVALUATION PLAN")
    logger.info("=" * 80)

    eval_count = 0

    # List combined dataset evaluations
    combined_keys = [k for k in test_datasets if k.endswith("_combined")]
    if combined_keys:
        logger.info(f"\nCombined Dataset Evaluations ({len(combined_keys)}):")
        for dataset_key in sorted(combined_keys):
            eval_count += 1
            group_name = dataset_key.replace("_combined", "")
            dataset_size = len(test_datasets[dataset_key])
            logger.info(f"  [{eval_count}] {group_name}_combined ({dataset_size} samples)")

    # List individual dataset evaluations
    individual_datasets = test_datasets.get("individual", {})
    if individual_datasets:
        logger.info(f"\nIndividual Dataset Evaluations ({len(individual_datasets)}):")
        for dataset_name in sorted(individual_datasets.keys()):
            eval_count += 1
            dataset_size = len(individual_datasets[dataset_name])
            is_wanfall = dataset_name.lower() in ["wanfall", "wan_fall"]
            subgroup_note = " [includes subgroup evaluation]" if is_wanfall else ""
            logger.info(f"  [{eval_count}] {dataset_name} ({dataset_size} samples){subgroup_note}")

    logger.info(f"\nTotal evaluations to perform: {eval_count}")
    logger.info("=" * 80)
    logger.info("")

    return eval_count
