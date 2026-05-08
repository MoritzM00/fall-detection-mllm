import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from falldet.plot import plot_relative_confusion_matrix

from .helpers import (
    extract_heatmap_annotations,
    extract_heatmap_collection,
    extract_heatmap_texts,
)


class TestRelativeConfusionMatrix:
    """Tests for relative confusion matrix plotting."""

    def test_returns_fig_and_ax(self):
        """Function returns a (fig, ax) tuple."""
        y_true_a = ["a", "a", "b", "b"]
        y_pred_a = ["a", "b", "b", "a"]
        y_true_b = ["a", "a", "b", "b"]
        y_pred_b = ["a", "a", "b", "a"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(fig)

    def test_default_figsize_uses_rc_params(self):
        """Relative confusion matrix also uses the current rc figure size."""
        original = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = [5.9, 5.0]
        try:
            fig, _ = plot_relative_confusion_matrix(
                ["a", "a", "b", "b"],
                ["a", "b", "b", "a"],
                ["a", "a", "b", "b"],
                ["a", "a", "b", "a"],
            )
            w, h = fig.get_size_inches()
            assert (w, h) == pytest.approx((5.9, 5.0))
            plt.close(fig)
        finally:
            plt.rcParams["figure.figsize"] = original

    def test_diagonal_improvement_is_plus(self):
        """Higher diagonal mass for run B is marked with a plus."""
        y_true_a = ["a", "a", "b", "b"]
        y_pred_a = ["a", "b", "b", "a"]
        y_true_b = ["a", "a", "b", "b"]
        y_pred_b = ["a", "a", "b", "a"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        texts = extract_heatmap_annotations(ax, n=2)
        assert texts[(0, 0)] == "+50"
        assert texts[(1, 1)] == ""
        plt.close(fig)

    def test_off_diagonal_reduction_is_negative(self):
        """Lower off-diagonal mass for run B shows a negative raw difference."""
        y_true_a = ["a", "a", "b", "b"]
        y_pred_a = ["a", "b", "b", "a"]
        y_true_b = ["a", "a", "b", "b"]
        y_pred_b = ["a", "a", "b", "a"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        texts = extract_heatmap_annotations(ax, n=2)
        assert texts[(0, 1)] == "-50"
        assert texts[(1, 0)] == ""
        plt.close(fig)

    def test_off_diagonal_increase_is_positive(self):
        """Higher off-diagonal mass for run B shows a positive raw difference."""
        y_true_a = ["a", "a", "b", "b"]
        y_pred_a = ["a", "a", "b", "a"]
        y_true_b = ["a", "a", "b", "b"]
        y_pred_b = ["a", "b", "b", "a"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        texts = extract_heatmap_annotations(ax, n=2)
        assert texts[(0, 1)] == "+50"
        assert texts[(0, 0)] == "-50"
        plt.close(fig)

    def test_heatmap_uses_split_colormaps(self):
        """Diagonal and off-diagonal cells use different color semantics."""
        y_true_a = ["a", "a", "b", "b"]
        y_pred_a = ["a", "b", "b", "a"]
        y_true_b = ["a", "a", "b", "b"]
        y_pred_b = ["a", "a", "b", "a"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        diagonal = extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        off_diagonal = extract_heatmap_collection(ax, 1, n=2).filled(np.nan)

        assert ax.collections[0].cmap.name == "RdBu"
        assert isinstance(ax.collections[0].norm, matplotlib.colors.TwoSlopeNorm)
        assert ax.collections[0].norm.vcenter == 0.0
        assert ax.collections[1].cmap.name == "RdBu_r"
        assert isinstance(ax.collections[1].norm, matplotlib.colors.TwoSlopeNorm)
        assert ax.collections[1].norm.vcenter == 0.0

        assert diagonal[0, 0] == pytest.approx(50.0)
        assert np.isnan(diagonal[0, 1])
        assert np.isnan(diagonal[1, 0])
        assert diagonal[1, 1] == pytest.approx(0.0)

        assert np.isnan(off_diagonal[0, 0])
        assert off_diagonal[0, 1] == pytest.approx(-50.0)
        assert off_diagonal[1, 0] == pytest.approx(0.0)
        assert np.isnan(off_diagonal[1, 1])
        plt.close(fig)

    def test_negative_diagonal_uses_signed_color_scale(self):
        """A diagonal regression keeps its negative sign in the heatmap data."""
        y_true_a = ["a", "a", "b", "b"]
        y_pred_a = ["a", "a", "b", "a"]
        y_true_b = ["a", "a", "b", "b"]
        y_pred_b = ["a", "b", "b", "a"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        diagonal = extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        assert diagonal[0, 0] == pytest.approx(-50.0)

        rgba = ax.collections[0].cmap(ax.collections[0].norm(diagonal[0, 0]))
        assert rgba[0] > rgba[1]
        assert rgba[0] > rgba[2]
        plt.close(fig)

    def test_positive_diagonal_uses_blue_signed_color_scale(self):
        """A diagonal improvement maps to the blue side of the signed scale."""
        y_true_a = ["a", "a", "b", "b"]
        y_pred_a = ["a", "b", "b", "a"]
        y_true_b = ["a", "a", "b", "b"]
        y_pred_b = ["a", "a", "b", "a"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        diagonal = extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        assert diagonal[0, 0] == pytest.approx(50.0)

        rgba = ax.collections[0].cmap(ax.collections[0].norm(diagonal[0, 0]))
        assert rgba[2] > rgba[0]
        assert rgba[2] > rgba[1]
        plt.close(fig)

    def test_dark_cells_use_light_annotation_text(self):
        """Very dark heatmap cells switch annotations to white."""
        y_true_a = ["a", "b"]
        y_pred_a = ["b", "a"]
        y_true_b = ["a", "b"]
        y_pred_b = ["a", "b"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        texts = extract_heatmap_texts(ax, n=2)
        assert texts[(0, 0)].get_color() == "white"
        assert texts[(0, 1)].get_color() == "white"
        plt.close(fig)

    def test_subset_slices_from_full_relative_matrix(self):
        """Subset values are taken from the full relative matrix."""
        y_true_a = ["a", "a", "b", "b", "c", "c"]
        y_pred_a = ["a", "b", "b", "a", "c", "a"]
        y_true_b = ["a", "a", "b", "b", "c", "c"]
        y_pred_b = ["a", "a", "b", "a", "c", "c"]

        fig, ax = plot_relative_confusion_matrix(
            y_true_a,
            y_pred_a,
            y_true_b,
            y_pred_b,
            subset=["a", "b"],
        )

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xlabels == ["a", "b"]
        diagonal = extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        off_diagonal = extract_heatmap_collection(ax, 1, n=2).filled(np.nan)
        assert diagonal[0, 0] == pytest.approx(50.0)
        assert off_diagonal[0, 1] == pytest.approx(-50.0)
        plt.close(fig)

    def test_subset_unknown_label_raises(self):
        """Unknown subset labels raise ValueError."""
        with pytest.raises(ValueError, match="not present in the data"):
            plot_relative_confusion_matrix(
                ["a"],
                ["a"],
                ["a"],
                ["a"],
                subset=["missing"],
            )

    def test_mismatched_lengths_raise(self):
        """Each run validates its own label-array lengths."""
        with pytest.raises(ValueError, match="same length"):
            plot_relative_confusion_matrix(["a"], ["a", "b"], ["a"], ["a"])

    def test_different_ground_truths_raise(self):
        """Comparing runs with different ground-truth labels raises ValueError."""
        with pytest.raises(ValueError, match="y_true_a and y_true_b must be identical"):
            plot_relative_confusion_matrix(["a", "b"], ["a", "b"], ["a", "a"], ["a", "a"])
