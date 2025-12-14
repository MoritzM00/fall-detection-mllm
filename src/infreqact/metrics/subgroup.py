"""
Subgroup evaluation utilities for demographic fairness analysis.

This module provides functions to compute and visualize metrics across
demographic subgroups (age, ethnicity, gender, BMI, etc.) for WanFall dataset.
"""

import logging

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, recall_score

logger = logging.getLogger(__name__)


def compute_subgroup_metrics(
    predictions: np.ndarray,
    references: np.ndarray,
    metadata: dict[str, list],
    subgroup_keys: list[str] = ["age_group", "ethnicity", "gender", "bmi_band"],
    idx2label: dict[int, str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute metrics for each demographic subgroup.

    Args:
        predictions: Array of predicted class labels
        references: Array of true class labels
        metadata: Dictionary mapping metadata keys to lists of values
                  e.g., {"age_group": ["elderly_65_plus", "children_5_12", ...], ...}
        subgroup_keys: List of metadata keys to evaluate (must exist in metadata dict)
        idx2label: Optional mapping from label indices to label names

    Returns:
        Dictionary structure:
        {
            "age_group": {
                "children_5_12": {
                    "accuracy": 0.95,
                    "balanced_accuracy": 0.94,
                    "macro_f1": 0.92,
                    "fall_sensitivity": 0.98,
                    "fall_specificity": 0.96,
                    "fall_f1": 0.97,
                    "sample_count": 1234,
                },
                "elderly_65_plus": {...},
                ...
            },
            "ethnicity": {...},
            ...
        }
    """
    results = {}

    # Filter out keys that don't have metadata
    valid_subgroup_keys = [
        key for key in subgroup_keys if key in metadata and metadata[key] is not None
    ]

    if not valid_subgroup_keys:
        logger.warning("No valid subgroup metadata found. Skipping subgroup evaluation.")
        return results

    for subgroup_key in valid_subgroup_keys:
        subgroup_values = metadata[subgroup_key]

        # Handle None values
        if all(v is None for v in subgroup_values):
            logger.info(f"Skipping subgroup '{subgroup_key}': all values are None")
            continue

        results[subgroup_key] = {}

        # Get unique subgroup categories
        unique_categories = set(v for v in subgroup_values if v is not None)

        for category in sorted(unique_categories):
            # Get indices for this subgroup
            indices = np.array([i for i, v in enumerate(subgroup_values) if v == category])

            if len(indices) == 0:
                continue

            # Extract predictions and references for this subgroup
            subgroup_preds = predictions[indices]
            subgroup_refs = references[indices]

            # Compute metrics
            metrics = _compute_metrics_for_group(subgroup_preds, subgroup_refs)
            metrics["sample_count"] = len(indices)

            results[subgroup_key][category] = metrics

    return results


def _compute_metrics_for_group(predictions: np.ndarray, references: np.ndarray) -> dict[str, float]:
    """
    Compute standard metrics for a single group.

    Returns metrics dictionary with:
    - accuracy
    - balanced_accuracy
    - macro_f1
    - fall_sensitivity, fall_specificity, fall_f1 (binary metrics for class 1)
    - fallen_sensitivity, fallen_specificity, fallen_f1 (binary metrics for class 2)
    - fall_union_fallen metrics (binary metrics for classes 1 or 2)
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    metrics = {}

    # Multi-class metrics
    metrics["accuracy"] = accuracy_score(references, predictions)
    metrics["balanced_accuracy"] = balanced_accuracy_score(references, predictions)
    metrics["macro_f1"] = f1_score(references, predictions, average="macro", zero_division=0)

    # Binary metrics for fall (class 1)
    binary_fall_pred = (predictions == 1).astype(int)
    binary_fall_ref = (references == 1).astype(int)

    if 1 in references:
        _, fall_sensitivity, fall_f1, _ = precision_recall_fscore_support(
            binary_fall_ref, binary_fall_pred, average="binary", beta=1, zero_division=0
        )
        fall_specificity = recall_score(1 - binary_fall_ref, 1 - binary_fall_pred, zero_division=0)
    else:
        fall_sensitivity, fall_f1, fall_specificity = 0, 0, 0

    metrics["fall_sensitivity"] = fall_sensitivity
    metrics["fall_specificity"] = fall_specificity
    metrics["fall_f1"] = fall_f1

    # Binary metrics for fallen (class 2)
    binary_fallen_pred = (predictions == 2).astype(int)
    binary_fallen_ref = (references == 2).astype(int)

    if 2 in references:
        _, fallen_sensitivity, fallen_f1, _ = precision_recall_fscore_support(
            binary_fallen_ref, binary_fallen_pred, average="binary", beta=1, zero_division=0
        )
        fallen_specificity = recall_score(
            1 - binary_fallen_ref, 1 - binary_fallen_pred, zero_division=0
        )
    else:
        fallen_sensitivity, fallen_f1, fallen_specificity = 0, 0, 0

    metrics["fallen_sensitivity"] = fallen_sensitivity
    metrics["fallen_specificity"] = fallen_specificity
    metrics["fallen_f1"] = fallen_f1

    # Binary metrics for fall ∪ fallen (classes 1 or 2)
    binary_union_pred = ((predictions == 1) | (predictions == 2)).astype(int)
    binary_union_ref = ((references == 1) | (references == 2)).astype(int)

    if 1 in references or 2 in references:
        _, union_sensitivity, union_f1, _ = precision_recall_fscore_support(
            binary_union_ref, binary_union_pred, average="binary", beta=1, zero_division=0
        )
        union_specificity = recall_score(
            1 - binary_union_ref, 1 - binary_union_pred, zero_division=0
        )
    else:
        union_sensitivity, union_f1, union_specificity = 0, 0, 0

    metrics["fall_union_fallen_sensitivity"] = union_sensitivity
    metrics["fall_union_fallen_specificity"] = union_specificity
    metrics["fall_union_fallen_f1"] = union_f1

    return metrics


def print_subgroup_summary(subgroup_results: dict[str, dict[str, dict[str, float]]]) -> None:
    """
    Print a summary of subgroup evaluation results to console.

    Args:
        subgroup_results: Results from compute_subgroup_metrics()
    """
    if not subgroup_results:
        logger.info("No subgroup results to display")
        return

    print("\n" + "=" * 80)
    print("SUBGROUP EVALUATION SUMMARY".center(80))
    print("=" * 80)

    for subgroup_key, categories in subgroup_results.items():
        print(f"\n{subgroup_key.replace('_', ' ').title()}:")
        print("-" * 80)

        if not categories:
            print("  (No data)")
            continue

        # Print header
        print(f"  {'Category':<25} {'Acc':>8} {'Bal Acc':>8} {'F1':>8} {'Fall Sen':>10} {'N':>8}")
        print("  " + "-" * 78)

        # Print each category
        for category in sorted(categories.keys()):
            metrics = categories[category]
            acc = metrics.get("accuracy", 0.0) * 100
            bal_acc = metrics.get("balanced_accuracy", 0.0) * 100
            f1 = metrics.get("macro_f1", 0.0) * 100
            fall_sen = metrics.get("fall_sensitivity", 0.0) * 100
            n = metrics.get("sample_count", 0)

            category_display = category.replace("_", " ").title()
            print(
                f"  {category_display:<25} {acc:>7.1f}% {bal_acc:>7.1f}% {f1:>7.1f}% {fall_sen:>9.1f}% {n:>8}"
            )

    print("\n" + "=" * 80)
