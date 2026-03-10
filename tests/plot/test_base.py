import matplotlib.pyplot as plt
import pytest

from falldet.plot import compute_publication_figsize, set_publication_rc_defaults


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
