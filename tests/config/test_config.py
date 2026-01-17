"""Tests for configuration utility functions."""

import pytest
from omegaconf import OmegaConf

from infreqact.config import resolve_model_name_from_config, resolve_model_path_from_config


class TestResolveModelNameFromConfig:
    """Test suite for resolve_model_name_from_config function."""

    def test_qwen_standard_instruct(self):
        """Test standard Qwen Instruct model name resolution."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "4B",
                "active_params": None,
            }
        )
        assert resolve_model_name_from_config(config) == "Qwen3-VL-4B-Instruct"

    def test_qwen_standard_thinking(self):
        """Test standard Qwen Thinking model name resolution."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Thinking",
                "params": "8B",
                "active_params": None,
            }
        )
        assert resolve_model_name_from_config(config) == "Qwen3-VL-8B-Thinking"

    def test_qwen_moe_model(self):
        """Test Qwen MoE model name resolution."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "30B",
                "active_params": "A3B",
            }
        )
        assert resolve_model_name_from_config(config) == "Qwen3-VL-30B-A3B-Instruct"

    def test_qwen_larger_moe_model(self):
        """Test larger Qwen MoE model name resolution."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "235B",
                "active_params": "A22B",
            }
        )
        assert resolve_model_name_from_config(config) == "Qwen3-VL-235B-A22B-Instruct"

    def test_internvl_model(self):
        """Test InternVL model name resolution."""
        config = OmegaConf.create(
            {
                "family": "InternVL",
                "version": "3_5",
                "params": "2B",
                "active_params": None,
            }
        )
        assert resolve_model_name_from_config(config) == "InternVL3_5-2B-HF"

    def test_internvl_larger_model(self):
        """Test larger InternVL model name resolution."""
        config = OmegaConf.create(
            {
                "family": "InternVL",
                "version": "3_5",
                "params": "8B",
                "active_params": None,
            }
        )
        assert resolve_model_name_from_config(config) == "InternVL3_5-8B-HF"

    def test_internvl_moe_model(self):
        """Test InternVL MoE model name resolution."""
        config = OmegaConf.create(
            {
                "family": "InternVL",
                "version": "3_5",
                "params": "20B",
                "active_params": "A4B",
            }
        )
        assert resolve_model_name_from_config(config) == "InternVL3_5-20B-A4B-HF"

    def test_internvl_larger_moe_model(self):
        """Test larger InternVL MoE model name resolution."""
        config = OmegaConf.create(
            {
                "family": "InternVL",
                "version": "3_5",
                "params": "241B",
                "active_params": "A28B",
            }
        )
        assert resolve_model_name_from_config(config) == "InternVL3_5-241B-A28B-HF"

    def test_dict_input_requires_omegaconf(self):
        """Test that plain dict input must be wrapped in OmegaConf."""
        config = {
            "family": "Qwen",
            "version": "3",
            "variant": "Instruct",
            "params": "4B",
            "active_params": None,
        }
        # Plain dicts don't work - must use OmegaConf.create()
        with pytest.raises(AttributeError):
            resolve_model_name_from_config(config)

        # Wrapped dict works
        wrapped = OmegaConf.create(config)
        assert resolve_model_name_from_config(wrapped) == "Qwen3-VL-4B-Instruct"

    def test_variant_lowercase_normalized(self):
        """Test that lowercase variant is normalized to title case."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "instruct",  # lowercase
                "params": "4B",
                "active_params": None,
            }
        )
        assert resolve_model_name_from_config(config) == "Qwen3-VL-4B-Instruct"

    def test_variant_uppercase_normalized(self):
        """Test that uppercase variant is normalized to title case."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "THINKING",  # uppercase
                "params": "8B",
                "active_params": None,
            }
        )
        assert resolve_model_name_from_config(config) == "Qwen3-VL-8B-Thinking"

    def test_variant_mixed_case_normalized(self):
        """Test that mixed case variant is normalized to title case."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "iNsTrUcT",  # mixed case
                "params": "4B",
                "active_params": None,
            }
        )
        assert resolve_model_name_from_config(config) == "Qwen3-VL-4B-Instruct"


class TestResolveModelPathFromConfig:
    """Test suite for resolve_model_path_from_config function."""

    def test_qwen_standard_path(self):
        """Test standard Qwen model path resolution."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "4B",
                "active_params": None,
            }
        )
        assert resolve_model_path_from_config(config) == "Qwen/Qwen3-VL-4B-Instruct"

    def test_qwen_moe_path(self):
        """Test Qwen MoE model path resolution."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "30B",
                "active_params": "A3B",
            }
        )
        assert resolve_model_path_from_config(config) == "Qwen/Qwen3-VL-30B-A3B-Instruct"

    def test_internvl_path(self):
        """Test InternVL model path resolution with OpenGVLab org."""
        config = OmegaConf.create(
            {
                "family": "InternVL",
                "version": "3_5",
                "params": "2B",
                "active_params": None,
            }
        )
        assert resolve_model_path_from_config(config) == "OpenGVLab/InternVL3_5-2B-HF"

    def test_internvl_moe_path(self):
        """Test InternVL MoE model path resolution with OpenGVLab org."""
        config = OmegaConf.create(
            {
                "family": "InternVL",
                "version": "3_5",
                "params": "30B",
                "active_params": "A3B",
            }
        )
        assert resolve_model_path_from_config(config) == "OpenGVLab/InternVL3_5-30B-A3B-HF"

    def test_unknown_family_uses_family_as_org(self):
        """Test that unknown family uses family name as organization."""
        config = OmegaConf.create(
            {
                "family": "NewModel",
                "version": "1",
                "variant": "Base",
                "params": "7B",
                "active_params": None,
            }
        )
        # Unknown family falls through to standard naming pattern
        result = resolve_model_path_from_config(config)
        assert result.startswith("NewModel/")


class TestConfigIntegration:
    """Integration tests for config utilities."""

    def test_all_supported_qwen_variants(self):
        """Test all supported Qwen variants."""
        variants = ["Instruct", "Thinking"]
        for variant in variants:
            config = OmegaConf.create(
                {
                    "family": "Qwen",
                    "version": "3",
                    "variant": variant,
                    "params": "4B",
                    "active_params": None,
                }
            )
            name = resolve_model_name_from_config(config)
            assert variant in name
            assert name.startswith("Qwen3-VL-")

    def test_all_common_param_sizes(self):
        """Test common parameter sizes."""
        param_sizes = ["2B", "4B", "8B", "14B", "32B", "72B"]
        for params in param_sizes:
            config = OmegaConf.create(
                {
                    "family": "Qwen",
                    "version": "3",
                    "variant": "Instruct",
                    "params": params,
                    "active_params": None,
                }
            )
            name = resolve_model_name_from_config(config)
            assert f"-{params}-" in name

    def test_moe_detection_with_active_params(self):
        """Test that MoE is correctly detected when active_params is set."""
        # Standard model (no MoE)
        standard_config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "8B",
                "active_params": None,
            }
        )
        standard_name = resolve_model_name_from_config(standard_config)
        assert "A" not in standard_name.split("-")[-2]  # No "A" prefix in params

        # MoE model
        moe_config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "30B",
                "active_params": "A3B",
            }
        )
        moe_name = resolve_model_name_from_config(moe_config)
        assert "-A3B-" in moe_name

    def test_path_and_name_consistency(self):
        """Test that path contains the correct name."""
        config = OmegaConf.create(
            {
                "family": "Qwen",
                "version": "3",
                "variant": "Instruct",
                "params": "4B",
                "active_params": None,
            }
        )
        name = resolve_model_name_from_config(config)
        path = resolve_model_path_from_config(config)
        assert path.endswith(name)
