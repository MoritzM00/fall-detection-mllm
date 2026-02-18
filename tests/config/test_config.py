"""Tests for configuration utility functions."""

import pytest

from falldet.config import resolve_model_name_from_config, resolve_model_path_from_config
from falldet.schemas import ModelConfig

# --- Fixtures ---


@pytest.fixture
def make_model_config():
    """Factory fixture to create model configs with defaults."""

    def _make_config(
        family="Qwen",
        org=None,
        version="3",
        variant="Instruct",
        params="4B",
        active_params=None,
    ):
        # Auto-resolve org from family if not provided
        if org is None:
            org = {"Qwen": "Qwen", "InternVL": "OpenGVLab", "Molmo": "allenai"}.get(family, family)
        return ModelConfig(
            org=org,
            family=family,
            version=version,
            variant=variant,
            params=params,
            active_params=active_params,
        )

    return _make_config


@pytest.fixture
def qwen_config(make_model_config):
    """Standard Qwen Instruct config."""
    return make_model_config()


@pytest.fixture
def qwen_moe_config(make_model_config):
    """Qwen MoE config."""
    return make_model_config(params="30B", active_params="A3B")


@pytest.fixture
def internvl_config(make_model_config):
    """Standard InternVL config."""
    return make_model_config(family="InternVL", version="3_5", variant=None, params="2B")


@pytest.fixture
def internvl_moe_config(make_model_config):
    """InternVL MoE config."""
    return make_model_config(
        family="InternVL", version="3_5", variant=None, params="20B", active_params="A4B"
    )


# --- Tests ---


class TestResolveModelNameFromConfig:
    """Test suite for resolve_model_name_from_config function."""

    def test_qwen_standard_instruct(self, qwen_config):
        """Test standard Qwen Instruct model name resolution."""
        assert resolve_model_name_from_config(qwen_config) == "Qwen3-VL-4B-Instruct"

    def test_qwen_standard_thinking(self, make_model_config):
        """Test standard Qwen Thinking model name resolution."""
        config = make_model_config(variant="Thinking", params="8B")
        assert resolve_model_name_from_config(config) == "Qwen3-VL-8B-Thinking"

    def test_qwen_moe_model(self, qwen_moe_config):
        """Test Qwen MoE model name resolution."""
        assert resolve_model_name_from_config(qwen_moe_config) == "Qwen3-VL-30B-A3B-Instruct"

    def test_qwen_larger_moe_model(self, make_model_config):
        """Test larger Qwen MoE model name resolution."""
        config = make_model_config(params="235B", active_params="A22B")
        assert resolve_model_name_from_config(config) == "Qwen3-VL-235B-A22B-Instruct"

    def test_internvl_model(self, internvl_config):
        """Test InternVL model name resolution."""
        assert resolve_model_name_from_config(internvl_config) == "InternVL3_5-2B-HF"

    def test_internvl_larger_model(self, make_model_config):
        """Test larger InternVL model name resolution."""
        config = make_model_config(family="InternVL", version="3_5", variant=None, params="8B")
        assert resolve_model_name_from_config(config) == "InternVL3_5-8B-HF"

    def test_internvl_moe_model(self, internvl_moe_config):
        """Test InternVL MoE model name resolution."""
        assert resolve_model_name_from_config(internvl_moe_config) == "InternVL3_5-20B-A4B-HF"

    def test_internvl_larger_moe_model(self, make_model_config):
        """Test larger InternVL MoE model name resolution."""
        config = make_model_config(
            family="InternVL", version="3_5", variant=None, params="241B", active_params="A28B"
        )
        assert resolve_model_name_from_config(config) == "InternVL3_5-241B-A28B-HF"

    def test_variant_lowercase_normalized(self, make_model_config):
        """Test that lowercase variant is normalized to title case."""
        config = make_model_config(variant="instruct")
        assert resolve_model_name_from_config(config) == "Qwen3-VL-4B-Instruct"

    def test_variant_uppercase_normalized(self, make_model_config):
        """Test that uppercase variant is normalized to title case."""
        config = make_model_config(variant="THINKING", params="8B")
        assert resolve_model_name_from_config(config) == "Qwen3-VL-8B-Thinking"

    def test_variant_mixed_case_normalized(self, make_model_config):
        """Test that mixed case variant is normalized to title case."""
        config = make_model_config(variant="iNsTrUcT")
        assert resolve_model_name_from_config(config) == "Qwen3-VL-4B-Instruct"


class TestResolveModelPathFromConfig:
    """Test suite for resolve_model_path_from_config function."""

    def test_qwen_standard_path(self, qwen_config):
        """Test standard Qwen model path resolution."""
        assert resolve_model_path_from_config(qwen_config) == "Qwen/Qwen3-VL-4B-Instruct"

    def test_qwen_moe_path(self, qwen_moe_config):
        """Test Qwen MoE model path resolution."""
        assert resolve_model_path_from_config(qwen_moe_config) == "Qwen/Qwen3-VL-30B-A3B-Instruct"

    def test_internvl_path(self, internvl_config):
        """Test InternVL model path resolution with OpenGVLab org."""
        assert resolve_model_path_from_config(internvl_config) == "OpenGVLab/InternVL3_5-2B-HF"

    def test_internvl_moe_path(self, make_model_config):
        """Test InternVL MoE model path resolution with OpenGVLab org."""
        config = make_model_config(
            family="InternVL", version="3_5", variant=None, params="30B", active_params="A3B"
        )
        assert resolve_model_path_from_config(config) == "OpenGVLab/InternVL3_5-30B-A3B-HF"


class TestConfigIntegration:
    """Integration tests for config utilities."""

    def test_all_supported_qwen_variants(self, make_model_config):
        """Test all supported Qwen variants."""
        variants = ["Instruct", "Thinking"]
        for variant in variants:
            config = make_model_config(variant=variant)
            name = resolve_model_name_from_config(config)
            assert variant in name
            assert name.startswith("Qwen3-VL-")

    def test_all_common_param_sizes(self, make_model_config):
        """Test common parameter sizes."""
        param_sizes = ["2B", "4B", "8B", "14B", "32B", "72B"]
        for params in param_sizes:
            config = make_model_config(params=params)
            name = resolve_model_name_from_config(config)
            assert f"-{params}-" in name

    def test_moe_detection_with_active_params(self, qwen_config, qwen_moe_config):
        """Test that MoE is correctly detected when active_params is set."""
        # Standard model (no MoE)
        standard_name = resolve_model_name_from_config(qwen_config)
        assert "A" not in standard_name.split("-")[-2]  # No "A" prefix in params

        # MoE model
        moe_name = resolve_model_name_from_config(qwen_moe_config)
        assert "-A3B-" in moe_name

    def test_path_and_name_consistency(self, qwen_config):
        """Test that path contains the correct name."""
        name = resolve_model_name_from_config(qwen_config)
        path = resolve_model_path_from_config(qwen_config)
        assert path.endswith(name)
