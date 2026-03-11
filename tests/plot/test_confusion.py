import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from falldet.plot import plot_confusion_matrix

from .helpers import (
    extract_heatmap_annotations,
    extract_heatmap_collection,
    extract_heatmap_texts,
)


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

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        for t in texts:
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

    def test_default_figsize_uses_rc_params(self, simple_data):
        """When figsize is omitted, the current rc figure size is used."""
        y_true, y_pred = simple_data
        original = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = [5.9, 5.0]
        try:
            fig, _ = plot_confusion_matrix(y_true, y_pred)
            w, h = fig.get_size_inches()
            assert (w, h) == pytest.approx((5.9, 5.0))
            plt.close(fig)
        finally:
            plt.rcParams["figure.figsize"] = original

    def test_existing_ax(self, simple_data):
        """Plotting into an existing axes works."""
        y_true, y_pred = simple_data
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_confusion_matrix(y_true, y_pred, ax=ax_ext)
        assert ax is ax_ext
        assert fig is fig_ext
        plt.close(fig)

    def test_cbar_disabled_by_default(self, simple_data):
        """Color bar is hidden unless explicitly requested."""
        y_true, y_pred = simple_data
        fig, _ = plot_confusion_matrix(y_true, y_pred)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_cbar_enabled_when_requested(self, simple_data):
        """Color bar can be enabled explicitly."""
        y_true, y_pred = simple_data
        fig, _ = plot_confusion_matrix(y_true, y_pred, cbar=True)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_perfect_predictions(self):
        """Perfect predictions produce a diagonal matrix."""
        labels = ["fall", "walk", "sitting"]
        y_true = labels * 3
        y_pred = labels * 3

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize=None)

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        values = [int(t) for t in texts]
        assert values.count(3) == 3
        assert values.count(0) == 6
        plt.close(fig)

    def test_heatmap_uses_split_colormaps(self):
        """Diagonal and off-diagonal cells use different color semantics."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b", "a"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")

        diagonal = extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        off_diagonal = extract_heatmap_collection(ax, 1, n=2).filled(np.nan)

        assert ax.collections[0].cmap.name == "Blues"
        assert ax.collections[1].cmap.name == "normal_confusion_offdiagonal"
        assert not isinstance(ax.collections[1].norm, matplotlib.colors.TwoSlopeNorm)
        assert diagonal[0, 0] == pytest.approx(2 / 3)
        assert np.isnan(diagonal[0, 1])
        assert np.isnan(diagonal[1, 0])
        assert diagonal[1, 1] == pytest.approx(1 / 2)
        assert np.isnan(off_diagonal[0, 0])
        assert off_diagonal[0, 1] == pytest.approx(1 / 3)
        assert off_diagonal[1, 0] == pytest.approx(1 / 2)
        assert np.isnan(off_diagonal[1, 1])
        plt.close(fig)

    def test_dark_cells_use_light_annotation_text(self):
        """Very dark normal-confusion cells switch annotations to white."""
        y_true = ["a", "a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "a", "a", "a", "b"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")

        texts = extract_heatmap_texts(ax, n=2)
        assert texts[(0, 0)].get_color() == "white"
        plt.close(fig)


class TestNormalizationValues:
    """Verify that the actual numeric values in the matrix are correct."""

    def test_normalize_true_row_sums_to_one(self):
        """Each row (true class) sums to ~1.0 under normalize='true'."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b", "a"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")

        collections = ax.collections
        assert len(collections) > 0
        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
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


class TestSubset:
    """Tests for the subset parameter."""

    def test_subset_shows_only_requested_classes(self, multiclass_data):
        """Only the subset classes appear in the tick labels."""
        y_true, y_pred = multiclass_data
        subset = ["fall", "walk"]
        fig, ax = plot_confusion_matrix(y_true, y_pred, subset=subset)

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xlabels == subset
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
        y_true = ["a", "a", "a", "b", "b", "c"]
        y_pred = ["a", "a", "b", "a", "b", "c"]
        subset = ["a", "b"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize=None, subset=subset)

        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        values = [int(t) for t in texts]
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

    def test_negative_annotation_threshold_raises(self):
        """Negative annotation thresholds are rejected."""
        with pytest.raises(ValueError, match="annot_threshold must be non-negative"):
            plot_confusion_matrix(["a"], ["a"], annot_threshold=-1)


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

    def test_annotation_threshold_hides_small_raw_counts(self):
        """Raw-count annotations below the threshold are blanked."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "a", "b"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, annot_threshold=2)

        texts = extract_heatmap_annotations(ax, n=2)
        assert texts[(0, 0)] == "2"
        assert texts[(0, 1)] == ""
        assert texts[(1, 0)] == ""
        assert texts[(1, 1)] == ""
        plt.close(fig)

    def test_annotation_threshold_hides_small_normalized_values(self):
        """Normalized annotations use the threshold as a proportion."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b", "a"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true", annot_threshold=0.4)

        texts = extract_heatmap_annotations(ax, n=2)
        assert texts[(0, 0)] == "0.67"
        assert texts[(0, 1)] == ""
        assert texts[(1, 0)] == "0.50"
        assert texts[(1, 1)] == "0.50"
        plt.close(fig)
