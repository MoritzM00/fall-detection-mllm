"""Tests for the confusion matrix visualization function."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

from falldet.visualization import plot_confusion_matrix

# Use non-interactive backend so tests don't pop up windows
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """Simple 3-class data with some misclassifications."""
    y_true = ["fall", "fall", "walk", "walk", "sitting", "sitting"]
    y_pred = ["fall", "walk", "walk", "walk", "sitting", "fall"]
    return y_true, y_pred


@pytest.fixture
def multiclass_data():
    """Larger multi-class data for subset tests."""
    y_true = [
        "fall",
        "fall",
        "fallen",
        "walk",
        "walk",
        "sitting",
        "standing",
        "lying",
        "crawl",
        "jump",
    ]
    y_pred = [
        "fall",
        "walk",
        "fallen",
        "walk",
        "sitting",
        "sitting",
        "standing",
        "lying",
        "crawl",
        "walk",
    ]
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasicFunctionality:
    """Tests for basic confusion matrix plotting."""

    def test_returns_fig_and_ax(self, simple_data):
        """Function returns a (fig, ax) tuple."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(fig)

    def test_no_normalize_shows_raw_counts(self, simple_data):
        """With normalize=None the matrix shows integer counts."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize=None)

        # The annotations should be integer strings
        # Collect text objects from the axes
        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        for t in texts:
            # Should be parseable as an integer
            assert t.lstrip("-").isdigit(), f"Expected integer annotation, got {t!r}"
        plt.close(fig)

    def test_normalize_true(self, simple_data):
        """normalize='true' produces row-normalized values."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        for t in texts:
            val = float(t)
            assert 0.0 <= val <= 1.0, f"Row-normalized value out of range: {val}"
        plt.close(fig)

    def test_normalize_pred(self, simple_data):
        """normalize='pred' produces column-normalized values."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="pred")

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        for t in texts:
            val = float(t)
            assert 0.0 <= val <= 1.0, f"Col-normalized value out of range: {val}"
        plt.close(fig)

    def test_normalize_all(self, simple_data):
        """normalize='all' normalizes over the entire matrix."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="all")

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        for t in texts:
            val = float(t)
            assert 0.0 <= val <= 1.0, f"All-normalized value out of range: {val}"
        plt.close(fig)

    def test_labels_sorted_alphabetically(self, simple_data):
        """Without subset, labels are sorted alphabetically."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred)

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xlabels == sorted(set(y_true) | set(y_pred))
        plt.close(fig)

    def test_custom_title(self, simple_data):
        """The title parameter is applied."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, title="My Title")
        assert ax.get_title() == "My Title"
        plt.close(fig)

    def test_custom_figsize(self, simple_data):
        """The figsize parameter controls the figure dimensions."""
        y_true, y_pred = simple_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.5
        assert abs(h - 8) < 0.5
        plt.close(fig)

    def test_existing_ax(self, simple_data):
        """Plotting into an existing axes works."""
        y_true, y_pred = simple_data
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_confusion_matrix(y_true, y_pred, ax=ax_ext)
        assert ax is ax_ext
        assert fig is fig_ext
        plt.close(fig)

    def test_perfect_predictions(self):
        """Perfect predictions produce a diagonal matrix."""
        labels = ["fall", "walk", "sitting"]
        y_true = labels * 3
        y_pred = labels * 3

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize=None)

        # Collect non-empty annotations
        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        values = [int(t) for t in texts]
        # The diagonal entries should be 3, off-diagonal 0
        # With 3 classes, there are 9 cells; 3 diagonal = 3, 6 off-diagonal = 0
        assert values.count(3) == 3
        assert values.count(0) == 6
        plt.close(fig)


# ---------------------------------------------------------------------------
# Normalization correctness
# ---------------------------------------------------------------------------


class TestNormalizationValues:
    """Verify that the actual numeric values in the matrix are correct."""

    def test_normalize_true_row_sums_to_one(self):
        """Each row (true class) sums to ~1.0 under normalize='true'."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b", "a"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")

        # Extract the heatmap data from the axes
        # The seaborn heatmap stores data in the QuadMesh
        collections = ax.collections
        assert len(collections) > 0
        # Alternatively, verify via annotations
        # Row 0 (a): TP=2, FP_pred_b=1 -> [2/3, 1/3] -> [0.67, 0.33]
        # Row 1 (b): FP_pred_a=1, TP=1 -> [1/2, 1/2] -> [0.50, 0.50]
        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        # 4 cells: [0.67, 0.33, 0.50, 0.50] (row-major)
        vals = [float(t) for t in texts]
        assert abs(vals[0] + vals[1] - 1.0) < 0.01
        assert abs(vals[2] + vals[3] - 1.0) < 0.01
        plt.close(fig)

    def test_normalize_pred_col_sums_to_one(self):
        """Each column (predicted class) sums to ~1.0 under normalize='pred'."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b", "a"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="pred")

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        vals = [float(t) for t in texts]
        # Col 0: vals[0] + vals[2] should be 1.0
        # Col 1: vals[1] + vals[3] should be 1.0
        assert abs(vals[0] + vals[2] - 1.0) < 0.01
        assert abs(vals[1] + vals[3] - 1.0) < 0.01
        plt.close(fig)

    def test_normalize_all_sums_to_one(self):
        """All cells sum to ~1.0 under normalize='all'."""
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "a"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="all")

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        vals = [float(t) for t in texts]
        assert abs(sum(vals) - 1.0) < 0.01
        plt.close(fig)


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------


class TestSubset:
    """Tests for the subset parameter."""

    def test_subset_shows_only_requested_classes(self, multiclass_data):
        """Only the subset classes appear in the tick labels."""
        y_true, y_pred = multiclass_data
        subset = ["fall", "walk"]
        fig, ax = plot_confusion_matrix(y_true, y_pred, subset=subset)

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xlabels == subset
        # No other real class labels should appear
        other_classes = (set(y_true) | set(y_pred)) - set(subset)
        for cls in other_classes:
            assert cls not in xlabels
        plt.close(fig)

    def test_subset_order_preserved(self, multiclass_data):
        """Subset classes appear in the order specified, not alphabetical."""
        y_true, y_pred = multiclass_data
        subset = ["walk", "fall", "fallen"]
        fig, ax = plot_confusion_matrix(y_true, y_pred, subset=subset)

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xlabels == subset
        plt.close(fig)

    def test_subset_values_from_full_matrix(self):
        """Subset cell values are taken from the full confusion matrix."""
        # Build data where we know exact counts:
        # true=a, pred=a: 2 times
        # true=a, pred=b: 1 time
        # true=b, pred=a: 1 time
        # true=b, pred=b: 1 time
        # true=c, pred=c: 1 time  (will be omitted by subset)
        y_true = ["a", "a", "a", "b", "b", "c"]
        y_pred = ["a", "a", "b", "a", "b", "c"]
        subset = ["a", "b"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize=None, subset=subset)

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        values = [int(t) for t in texts]
        # The 2x2 submatrix for a, b (in that order) should be:
        #       a  b
        #   a [ 2  1 ]
        #   b [ 1  1 ]
        assert values == [2, 1, 1, 1]
        plt.close(fig)

    def test_subset_with_normalization(self):
        """Subset + normalization: values come from the full normalized matrix."""
        y_true = ["a", "a", "a", "b", "b", "c"]
        y_pred = ["a", "a", "b", "a", "b", "c"]
        subset = ["a", "b"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true", subset=subset)

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        vals = [float(t) for t in texts]
        # Row "a" in full matrix (normalized by true): [2/3, 1/3, 0] -> a=0.67, b=0.33
        # Row "b" in full matrix (normalized by true): [1/2, 1/2, 0] -> a=0.50, b=0.50
        assert abs(vals[0] - 2 / 3) < 0.01
        assert abs(vals[1] - 1 / 3) < 0.01
        assert abs(vals[2] - 0.50) < 0.01
        assert abs(vals[3] - 0.50) < 0.01
        plt.close(fig)

    def test_subset_unknown_label_raises(self, simple_data):
        """Subset with a label not in the data raises ValueError."""
        y_true, y_pred = simple_data
        with pytest.raises(ValueError, match="not present in the data"):
            plot_confusion_matrix(y_true, y_pred, subset=["nonexistent"])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for error handling on invalid inputs."""

    def test_empty_y_true_raises(self):
        """Empty y_true raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_confusion_matrix([], ["a"])

    def test_empty_y_pred_raises(self):
        """Empty y_pred raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_confusion_matrix(["a"], [])

    def test_mismatched_lengths_raises(self):
        """Different lengths for y_true and y_pred raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            plot_confusion_matrix(["a", "b"], ["a"])

    def test_invalid_normalize_raises(self):
        """Invalid normalize value raises ValueError."""
        with pytest.raises(ValueError, match="normalize must be one of"):
            plot_confusion_matrix(["a"], ["a"], normalize="invalid")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_class(self):
        """Works with a single class."""
        y_true = ["fall", "fall", "fall"]
        y_pred = ["fall", "fall", "fall"]
        fig, ax = plot_confusion_matrix(y_true, y_pred)

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xlabels == ["fall"]
        plt.close(fig)

    def test_single_sample(self):
        """Works with a single sample."""
        fig, ax = plot_confusion_matrix(["fall"], ["walk"])
        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        # 2x2 matrix: fall/walk x fall/walk
        assert len(texts) == 4
        plt.close(fig)

    def test_subset_single_class(self, multiclass_data):
        """Subset with a single class works."""
        y_true, y_pred = multiclass_data
        fig, ax = plot_confusion_matrix(y_true, y_pred, subset=["fall"])

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xlabels == ["fall"]
        plt.close(fig)

    def test_class_only_in_pred(self):
        """A class appearing only in y_pred is still included."""
        y_true = ["a", "a"]
        y_pred = ["a", "b"]
        fig, ax = plot_confusion_matrix(y_true, y_pred)

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert "b" in xlabels
        plt.close(fig)
