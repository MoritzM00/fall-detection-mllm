"""Tests for metrics computation functions."""

import numpy as np
import pytest

from falldet.metrics.base import compute_metrics


class TestComputeMetrics:
    """Test suite for compute_metrics function."""

    def test_perfect_predictions_string_labels(self):
        """Test metrics with perfect predictions using string labels."""
        y_true = ["walk", "fall", "fallen", "sitting", "walk"]
        y_pred = ["walk", "fall", "fallen", "sitting", "walk"]

        metrics = compute_metrics(y_pred, y_true)

        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_perfect_predictions_numeric_labels(self):
        """Test metrics with perfect predictions using numeric labels."""
        y_true = [0, 1, 2, 4, 0]  # walk, fall, fallen, sitting, walk
        y_pred = [0, 1, 2, 4, 0]

        metrics = compute_metrics(y_pred, y_true)

        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_mixed_predictions(self):
        """Test metrics with some correct and some incorrect predictions."""
        y_true = ["walk", "fall", "fallen", "walk"]
        y_pred = ["walk", "fall", "walk", "walk"]  # Misclassified fallen as walk

        metrics = compute_metrics(y_pred, y_true)

        assert metrics["accuracy"] == 0.75  # 3 out of 4 correct
        assert "fall_f1" in metrics
        assert "fallen_f1" in metrics

    def test_binary_fall_metrics(self):
        """Test binary fall detection metrics."""
        # 2 fall, 2 non-fall
        y_true = ["fall", "fall", "walk", "sitting"]
        y_pred = ["fall", "walk", "walk", "sitting"]  # Missed one fall

        metrics = compute_metrics(y_pred, y_true)

        # Fall sensitivity = TP / (TP + FN) = 1 / 2 = 0.5
        assert metrics["fall_sensitivity"] == 0.5
        # Fall specificity = TN / (TN + FP) = 2 / 2 = 1.0
        assert metrics["fall_specificity"] == 1.0

    def test_fall_union_fallen_metrics(self):
        """Test fall ∪ fallen binary metrics."""
        y_true = ["fall", "fallen", "walk", "sitting"]
        y_pred = ["fall", "walk", "walk", "sitting"]  # Missed fallen

        metrics = compute_metrics(y_pred, y_true)

        # Fall ∪ fallen sensitivity = TP / (TP + FN) = 1 / 2 = 0.5
        assert metrics["fall_union_fallen_sensitivity"] == 0.5
        # Specificity = TN / (TN + FP) = 2 / 2 = 1.0
        assert metrics["fall_union_fallen_specificity"] == 1.0

    def test_per_class_metrics(self):
        """Test per-class metrics are computed correctly."""
        y_true = ["walk", "walk", "fall", "fall"]
        y_pred = ["walk", "fall", "fall", "fall"]

        metrics = compute_metrics(y_pred, y_true)

        # Should have per-class metrics for walk and fall
        assert "walk_accuracy" in metrics
        assert "walk_precision" in metrics
        assert "walk_sensitivity" in metrics
        assert "walk_specificity" in metrics
        assert "walk_f1" in metrics

        assert "fall_accuracy" in metrics
        assert "fall_f1" in metrics

    def test_class_distributions(self):
        """Test class distribution computation."""
        y_true = ["walk", "walk", "fall", "sitting"]
        y_pred = ["walk", "walk", "walk", "sitting"]

        metrics = compute_metrics(y_pred, y_true)

        # True distribution
        assert metrics["true_dist_walk"] == 0.5  # 2/4
        assert metrics["true_dist_fall"] == 0.25  # 1/4
        assert metrics["true_dist_sitting"] == 0.25  # 1/4

        # Predicted distribution
        assert metrics["pred_dist_walk"] == 0.75  # 3/4
        assert metrics["pred_dist_sitting"] == 0.25  # 1/4

    def test_sample_counts(self):
        """Test sample count computation."""
        y_true = ["walk", "walk", "fall", "sitting"]
        y_pred = ["walk", "walk", "walk", "sitting"]

        metrics = compute_metrics(y_pred, y_true)

        assert metrics["sample_count"] == 4
        assert metrics["sample_count_walk"] == 2
        assert metrics["sample_count_fall"] == 1
        assert metrics["sample_count_sitting"] == 1

    def test_no_fall_examples(self):
        """Test metrics when there are no fall examples in the dataset."""
        y_true = ["walk", "sitting", "standing"]
        y_pred = ["walk", "sitting", "standing"]

        metrics = compute_metrics(y_pred, y_true)

        # Fall metrics should be 0 when no fall examples exist
        assert metrics["fall_sensitivity"] == 0.0
        assert metrics["fall_specificity"] == 0.0
        assert metrics["fall_f1"] == 0.0

    def test_empty_inputs_raise_error(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="Cannot compute metrics on empty arrays"):
            compute_metrics([], [])

    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched input lengths raise ValueError."""
        y_true = ["walk", "fall"]
        y_pred = ["walk", "fall", "sitting"]

        with pytest.raises(ValueError, match="must have the same length"):
            compute_metrics(y_pred, y_true)

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 0, 0])

        metrics = compute_metrics(y_pred, y_true)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics

    def test_unknown_labels(self):
        """Test handling of unknown label strings."""
        # Unknown labels get mapped to -1
        y_true = ["walk", "unknown_action", "fall"]
        y_pred = ["walk", "unknown_action", "fall"]

        metrics = compute_metrics(y_pred, y_true)

        # Should still compute metrics, treating -1 as a valid class
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics


def test_example_usage():
    """Example usage from docstring."""
    y_true = ["walk", "fall", "fallen", "walk"]
    y_pred = ["walk", "fall", "walk", "walk"]

    metrics = compute_metrics(y_pred, y_true)

    assert "accuracy" in metrics
    assert "fall_f1" in metrics
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Fall F1: {metrics['fall_f1']:.2f}")


def test_per_class_metrics_stay_aligned_when_classes_are_absent():
    # fall (1) and standing (8) only: every class in between is absent
    references = np.array([1] * 4 + [8] * 4)
    predictions = np.array([1] * 8)

    metrics = compute_metrics(predictions, references)

    assert metrics["fall_sensitivity"] == 1.0
    assert metrics["fall_precision"] == 0.5
    assert metrics["standing_sensitivity"] == 0.0
    assert metrics["standing_f1"] == 0.0
