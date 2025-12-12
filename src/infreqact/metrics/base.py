"""Metrics computation for action recognition evaluation.

Adapted from fall-da/training/metrics_factory.py to work without HuggingFace Trainer.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from infreqact.data.video_dataset import idx2label, label2idx


def compute_metrics(
    y_pred: list[str] | list[int] | np.ndarray,
    y_true: list[str] | list[int] | np.ndarray,
) -> dict[str, float]:
    """
    Compute comprehensive evaluation metrics for action recognition.

    This function computes:
    - Multi-class: accuracy, balanced accuracy, macro F1
    - Per-class: accuracy, precision, sensitivity (recall), specificity, NPV, F1
    - Binary fall detection: sensitivity, specificity, F1
    - Binary fallen detection: sensitivity, specificity, F1
    - Binary fall∪fallen: sensitivity, specificity, F1
    - Class distributions: true and predicted percentages
    - Sample counts per class

    Args:
        y_pred: Predicted labels. Can be:
            - List of label strings (e.g., ["walk", "fall", "fallen"])
            - List of numeric indices (e.g., [0, 1, 2])
            - NumPy array of numeric indices
        y_true: Ground truth labels. Same format options as y_pred.

    Returns:
        Dictionary of computed metrics. Keys include:
            - accuracy, balanced_accuracy, macro_f1
            - fall_sensitivity, fall_specificity, fall_f1
            - fallen_sensitivity, fallen_specificity, fallen_f1
            - fall_union_fallen_sensitivity, fall_union_fallen_specificity, fall_union_fallen_f1
            - {class_name}_accuracy, {class_name}_precision, {class_name}_sensitivity,
              {class_name}_specificity, {class_name}_npv, {class_name}_f1
            - true_dist_{class_name}, pred_dist_{class_name}
            - sample_count_{class_name}, sample_count

    Example:
        >>> y_true = ["walk", "fall", "fallen", "walk"]
        >>> y_pred = ["walk", "fall", "walk", "walk"]
        >>> metrics = compute_metrics(y_pred, y_true)
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
        >>> print(f"Fall F1: {metrics['fall_f1']:.2f}")
    """

    # Convert string labels to numeric indices if needed
    def to_numeric(labels):
        if isinstance(labels, list) and labels and isinstance(labels[0], str):
            return np.array([label2idx.get(label, -1) for label in labels])
        return np.array(labels)

    predictions = to_numeric(y_pred)
    references = to_numeric(y_true)

    # Validate inputs
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have the same length. "
            f"Got {len(predictions)} predictions and {len(references)} references."
        )

    if len(predictions) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    # Multi-class metrics
    acc = accuracy_score(references, predictions)
    bacc = balanced_accuracy_score(references, predictions)
    macro_f1 = f1_score(references, predictions, average="macro", zero_division=0)

    # True class distribution (ground truth labels)
    unique_true, counts_true = np.unique(references, return_counts=True)
    true_class_distribution = {
        idx2label.get(int(label), f"unknown_{label}"): count / len(references)
        for label, count in zip(unique_true, counts_true)
    }

    # Per-class sample counts (absolute counts, not percentages)
    class_sample_counts = {
        idx2label.get(int(label), f"unknown_{label}"): int(count)
        for label, count in zip(unique_true, counts_true)
    }

    # Predicted class distribution
    unique_pred, counts_pred = np.unique(predictions, return_counts=True)
    pred_class_distribution = {
        idx2label.get(int(label), f"unknown_{label}"): count / len(predictions)
        for label, count in zip(unique_pred, counts_pred)
    }

    # Per-class metrics (accuracy, precision, sensitivity, specificity, NPV, F1)
    # These are computed per-class treating each class as a binary problem (one-vs-rest)
    per_class_f1_scores = f1_score(references, predictions, average=None, zero_division=0)
    per_class_recall_scores = recall_score(references, predictions, average=None, zero_division=0)
    per_class_precision_scores = precision_score(
        references, predictions, average=None, zero_division=0
    )

    # Per-class accuracy, specificity, and NPV: computed per class (one-vs-rest)
    per_class_accuracy_scores = []
    per_class_specificity_scores = []
    per_class_npv_scores = []

    for class_idx in range(len(per_class_f1_scores)):
        if class_idx in unique_true:
            # Binary problem: this class vs all others
            binary_pred = (predictions == class_idx).astype(int)
            binary_ref = (references == class_idx).astype(int)

            # Accuracy = (TP + TN) / (TP + TN + FP + FN)
            class_acc = (binary_pred == binary_ref).sum() / len(binary_ref)
            per_class_accuracy_scores.append(class_acc)

            # Specificity = TN / (TN + FP) = recall of negative class
            class_spec = recall_score(1 - binary_ref, 1 - binary_pred, zero_division=0)
            per_class_specificity_scores.append(class_spec)

            # NPV = TN / (TN + FN)
            # TN = correctly predicted as negative
            # FN = incorrectly predicted as negative (actually positive)
            tn = ((binary_pred == 0) & (binary_ref == 0)).sum()
            fn = ((binary_pred == 0) & (binary_ref == 1)).sum()
            class_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            per_class_npv_scores.append(class_npv)
        else:
            per_class_accuracy_scores.append(0.0)
            per_class_specificity_scores.append(0.0)
            per_class_npv_scores.append(0.0)

    # Build per-class metrics dictionaries with clear naming
    per_class_metrics_dict = {}
    for i, (acc_score, prec, sens, spec, npv, f1) in enumerate(
        zip(
            per_class_accuracy_scores,
            per_class_precision_scores,
            per_class_recall_scores,
            per_class_specificity_scores,
            per_class_npv_scores,
            per_class_f1_scores,
        )
    ):
        if i in unique_true:
            class_name = idx2label.get(i, f"unknown_{i}")
            per_class_metrics_dict[f"{class_name}_accuracy"] = acc_score
            per_class_metrics_dict[f"{class_name}_precision"] = prec
            per_class_metrics_dict[f"{class_name}_sensitivity"] = sens
            per_class_metrics_dict[f"{class_name}_specificity"] = spec
            per_class_metrics_dict[f"{class_name}_npv"] = npv
            per_class_metrics_dict[f"{class_name}_f1"] = f1

    # Binary metrics for fall (class 1)
    binary_fall_pred = (predictions == 1).astype(int)
    binary_fall_ref = (references == 1).astype(int)

    if 1 in unique_true:
        # Use sklearn for binary metrics
        _, fall_sensitivity, fall_f1, _ = precision_recall_fscore_support(
            binary_fall_ref, binary_fall_pred, average="binary", beta=1, zero_division=0
        )
        # Specificity = TN / (TN + FP) = recall of the negative class
        fall_specificity = recall_score(1 - binary_fall_ref, 1 - binary_fall_pred, zero_division=0)
    else:
        # No fall examples in this dataset
        fall_sensitivity, fall_f1, fall_specificity = 0.0, 0.0, 0.0

    # Binary metrics for fallen (class 2)
    binary_fallen_pred = (predictions == 2).astype(int)
    binary_fallen_ref = (references == 2).astype(int)

    if 2 in unique_true:
        _, fallen_sensitivity, fallen_f1, _ = precision_recall_fscore_support(
            binary_fallen_ref, binary_fallen_pred, average="binary", beta=1, zero_division=0
        )
        fallen_specificity = recall_score(
            1 - binary_fallen_ref, 1 - binary_fallen_pred, zero_division=0
        )
    else:
        # No fallen examples in this dataset
        fallen_sensitivity, fallen_f1, fallen_specificity = 0.0, 0.0, 0.0

    # Binary metrics for fall ∪ fallen (classes 1 or 2)
    binary_fall_union_fallen_pred = ((predictions == 1) | (predictions == 2)).astype(int)
    binary_fall_union_fallen_ref = ((references == 1) | (references == 2)).astype(int)

    if 1 in unique_true or 2 in unique_true:
        _, fall_union_fallen_sensitivity, fall_union_fallen_f1, _ = precision_recall_fscore_support(
            binary_fall_union_fallen_ref,
            binary_fall_union_fallen_pred,
            average="binary",
            beta=1,
            zero_division=0,
        )
        # Specificity = TN / (TN + FP) = recall of the negative class
        fall_union_fallen_specificity = recall_score(
            1 - binary_fall_union_fallen_ref, 1 - binary_fall_union_fallen_pred, zero_division=0
        )
    else:
        # No fall or fallen examples in this dataset
        fall_union_fallen_sensitivity = 0.0
        fall_union_fallen_f1 = 0.0
        fall_union_fallen_specificity = 0.0

    # Build metrics dictionary
    metrics_dict = {
        # Multi-class metrics
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        # Binary metrics: fall vs non-fall
        "fall_sensitivity": fall_sensitivity,
        "fall_specificity": fall_specificity,
        "fall_f1": fall_f1,
        # Binary metrics: fallen vs non-fallen
        "fallen_sensitivity": fallen_sensitivity,
        "fallen_specificity": fallen_specificity,
        "fallen_f1": fallen_f1,
        # Binary metrics: fall ∪ fallen vs others
        "fall_union_fallen_sensitivity": fall_union_fallen_sensitivity,
        "fall_union_fallen_specificity": fall_union_fallen_specificity,
        "fall_union_fallen_f1": fall_union_fallen_f1,
    }

    # Add true and predicted class distributions and per-class metrics
    metrics_dict.update({f"true_dist_{k}": v for k, v in true_class_distribution.items()})
    metrics_dict.update({f"pred_dist_{k}": v for k, v in pred_class_distribution.items()})
    metrics_dict.update({f"sample_count_{k}": v for k, v in class_sample_counts.items()})
    metrics_dict.update(per_class_metrics_dict)  # Per-class accuracy, sensitivity, F1
    metrics_dict["sample_count"] = len(references)

    return metrics_dict
