"""Tests for the confusion matrix visualization functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from falldet.visualization import (
    compute_publication_figsize,
    plot_confusion_matrix,
    plot_relative_confusion_matrix,
    set_publication_rc_defaults,
)

# Use non-interactive backend so tests don't pop up windows
matplotlib.use("Agg")


def _extract_heatmap_annotations(ax: matplotlib.axes.Axes, n: int) -> dict[tuple[int, int], str]:
    """Return a ``{(row, col): text}`` dict for the *n* x *n* heatmap cells.

    Seaborn annotations live in data-coordinates (cell centres at 0.5, 1.5, …).
    We keep only data-coordinate texts whose rounded indices fall inside the
    matrix.
    """
    result: dict[tuple[int, int], str] = {}
    for t in ax.texts:
        if t.get_transform() != ax.transData:
            continue
        x, y = t.get_position()
        col = round(x - 0.5)
        row = round(y - 0.5)
        if 0 <= row < n and 0 <= col < n:
            result[(row, col)] = t.get_text()
    return result


def _extract_heatmap_texts(
    ax: matplotlib.axes.Axes,
    n: int,
) -> dict[tuple[int, int], matplotlib.text.Text]:
    """Return a ``{(row, col): text_artist}`` dict for the *n* x *n* heatmap cells."""
    result: dict[tuple[int, int], matplotlib.text.Text] = {}
    for t in ax.texts:
        if t.get_transform() != ax.transData:
            continue
        x, y = t.get_position()
        col = round(x - 0.5)
        row = round(y - 0.5)
        if 0 <= row < n and 0 <= col < n:
            result[(row, col)] = t
    return result


def _extract_heatmap_collection(
    ax: matplotlib.axes.Axes,
    collection_index: int,
    n: int,
) -> np.ma.MaskedArray:
    """Return a heatmap collection as an ``n x n`` masked array."""
    return np.ma.asarray(ax.collections[collection_index].get_array()).reshape(n, n)


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


class TestPublicationRcDefaults:
    """Tests for publication-style rc defaults."""

    def test_returns_resolved_rc_dict(self):
        """The helper returns the rcParams it applies."""
        rc, figsize = set_publication_rc_defaults()
        assert isinstance(rc, dict)
        assert isinstance(figsize, tuple)
        assert rc["font.family"] == "serif"
        assert rc["text.usetex"] is False
        assert rc["axes.labelsize"] == 9
        assert rc["xtick.labelsize"] == 8
        assert rc["ytick.labelsize"] == 8
        assert figsize == pytest.approx((427.43153 / 72.27, (427.43153 / 72.27) * 0.66))
        assert rc["figure.figsize"] == pytest.approx(figsize)

    def test_paper_target_uses_smaller_defaults(self):
        """Paper preset uses a narrower default figure width."""
        rc, figsize = set_publication_rc_defaults(target="paper")
        assert rc["axes.labelsize"] == 9
        assert figsize == pytest.approx((246.0 / 72.27, (246.0 / 72.27) * 0.66))

    def test_applies_custom_rc_overrides(self):
        """Explicit rc overrides are applied last."""
        rc, _ = set_publication_rc_defaults(rc={"axes.labelsize": 13, "lines.linewidth": 2.5})
        assert rc["axes.labelsize"] == 13
        assert rc["lines.linewidth"] == 2.5
        assert plt.rcParams["axes.labelsize"] == 13
        assert plt.rcParams["lines.linewidth"] == 2.5

    def test_enables_tex_when_requested(self):
        """LaTeX text rendering is opt-in."""
        rc, _ = set_publication_rc_defaults(use_tex=True)
        assert rc["text.usetex"] is True
        assert "text.latex.preamble" in rc

    def test_custom_text_width_and_fraction_control_figure_size(self):
        """Figure size can be derived from a custom LaTeX text width."""
        rc, figsize = set_publication_rc_defaults(
            text_width_pt=360.0,
            width_fraction=0.5,
            height_ratio=0.75,
        )
        width_in = (360.0 / 72.27) * 0.5
        assert figsize == pytest.approx((width_in, width_in * 0.75))
        assert rc["figure.figsize"] == pytest.approx(figsize)

    def test_invalid_target_raises(self):
        """Unknown target presets are rejected."""
        with pytest.raises(ValueError, match="target must be one of"):
            set_publication_rc_defaults(target="poster")


class TestPublicationFigsize:
    """Tests for the standalone publication figsize helper."""

    def test_default_thesis_figsize(self):
        """The helper uses the thesis text width by default."""
        figsize = compute_publication_figsize()
        assert figsize == pytest.approx((427.43153 / 72.27, (427.43153 / 72.27) * 0.66))

    def test_custom_fraction_and_ratio(self):
        """Width fraction and height ratio scale the computed size."""
        figsize = compute_publication_figsize(
            text_width_pt=360.0, width_fraction=0.5, height_ratio=0.75
        )
        width_in = (360.0 / 72.27) * 0.5
        assert figsize == pytest.approx((width_in, width_in * 0.75))

    def test_invalid_figsize_target_raises(self):
        """Unknown targets are rejected."""
        with pytest.raises(ValueError, match="target must be one of"):
            compute_publication_figsize(target="poster")

    def test_non_positive_dimensions_raise(self):
        """Invalid width and ratio arguments are rejected."""
        with pytest.raises(ValueError, match="text_width_pt must be positive"):
            compute_publication_figsize(text_width_pt=0.0)
        with pytest.raises(ValueError, match="width_fraction must be positive"):
            compute_publication_figsize(width_fraction=0.0)
        with pytest.raises(ValueError, match="height_ratio must be positive"):
            compute_publication_figsize(height_ratio=0.0)


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

        # Collect non-empty annotations
        texts = [t.get_text() for t in ax.texts if t.get_text() != ""]
        values = [int(t) for t in texts]
        # The diagonal entries should be 3, off-diagonal 0
        # With 3 classes, there are 9 cells; 3 diagonal = 3, 6 off-diagonal = 0
        assert values.count(3) == 3
        assert values.count(0) == 6
        plt.close(fig)

    def test_heatmap_uses_split_colormaps(self):
        """Diagonal and off-diagonal cells use different color semantics."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b", "a"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, normalize="true")

        diagonal = _extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        off_diagonal = _extract_heatmap_collection(ax, 1, n=2).filled(np.nan)

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

        texts = _extract_heatmap_texts(ax, n=2)
        assert texts[(0, 0)].get_color() == "white"
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

    def test_negative_annotation_threshold_raises(self):
        """Negative annotation thresholds are rejected."""
        with pytest.raises(ValueError, match="annot_threshold must be non-negative"):
            plot_confusion_matrix(["a"], ["a"], annot_threshold=-1)


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

    def test_annotation_threshold_hides_small_raw_counts(self):
        """Raw-count annotations below the threshold are blanked."""
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "a", "b"]

        fig, ax = plot_confusion_matrix(y_true, y_pred, annot_threshold=2)

        texts = _extract_heatmap_annotations(ax, n=2)
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

        texts = _extract_heatmap_annotations(ax, n=2)
        assert texts[(0, 0)] == "0.67"
        assert texts[(0, 1)] == ""
        assert texts[(1, 0)] == "0.50"
        assert texts[(1, 1)] == "0.50"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Relative confusion matrix
# ---------------------------------------------------------------------------


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

        texts = _extract_heatmap_annotations(ax, n=2)
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

        texts = _extract_heatmap_annotations(ax, n=2)
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

        texts = _extract_heatmap_annotations(ax, n=2)
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

        diagonal = _extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        off_diagonal = _extract_heatmap_collection(ax, 1, n=2).filled(np.nan)

        assert ax.collections[0].cmap.name == "Blues"
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

    def test_dark_cells_use_light_annotation_text(self):
        """Very dark heatmap cells switch annotations to white."""
        y_true_a = ["a", "b"]
        y_pred_a = ["b", "a"]
        y_true_b = ["a", "b"]
        y_pred_b = ["a", "b"]

        fig, ax = plot_relative_confusion_matrix(y_true_a, y_pred_a, y_true_b, y_pred_b)

        texts = _extract_heatmap_texts(ax, n=2)
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
        diagonal = _extract_heatmap_collection(ax, 0, n=2).filled(np.nan)
        off_diagonal = _extract_heatmap_collection(ax, 1, n=2).filled(np.nan)
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
