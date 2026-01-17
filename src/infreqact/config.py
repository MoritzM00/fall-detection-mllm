"""Configuration utilities for model path resolution."""

import logging
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def resolve_model_name_from_config(model_config: Any) -> str:
    """
    Resolve the model name from a model config.

    Args:
        model_config: Model configuration (DictConfig or dict-like object) with fields:
            - family: Model family (e.g., "Qwen", "InternVL")
            - version: Model version (e.g., "3", "3_5")
            - variant: Model variant (e.g., "Instruct", "Thinking") or None
            - params: Total parameter count (e.g., "4B", "30B")
            - active_params: Active params for MoE models (e.g., "A3B"), None for standard

    Returns:
        The model name (e.g., "Qwen3-VL-4B-Instruct", "InternVL3_5-2B-HF")
    """
    family = model_config.family
    version = model_config.version
    variant = model_config.get("variant")
    params = model_config.params
    active_params = model_config.get("active_params")

    # Normalize variant to title case (e.g., "instruct" -> "Instruct")
    if variant:
        variant = variant.title()

    if family == "InternVL":
        # InternVL models: InternVL3_5-2B-HF or InternVL3_5-20B-A4B-HF (MoE)
        if variant is not None:
            logger.warning("InternVL models do not use variants; ignoring variant field.")

        if active_params:
            # InternVL MoE: InternVL3_5-20B-A4B-HF
            return f"{family}{version}-{params}-{active_params}-HF"
        else:
            # Standard InternVL: InternVL3_5-2B-HF
            return f"{family}{version}-{params}-HF"
    elif active_params:
        # Qwen MoE model: Qwen3-VL-30B-A3B-Instruct
        return f"{family}{version}-VL-{params}-{active_params}-{variant}"
    else:
        # Standard Qwen: Qwen3-VL-4B-Instruct
        return f"{family}{version}-VL-{params}-{variant}"


def resolve_model_path_from_config(model_config: DictConfig) -> str:
    """
    Resolve the HuggingFace model checkpoint path from a model config.

    Args:
        model_config: Model configuration (DictConfig or dict-like object) with fields:
            - family: Model family (e.g., "Qwen", "InternVL")
            - version: Model version (e.g., "3", "3_5")
            - variant: Model variant (e.g., "Instruct", "Thinking") or None
            - params: Total parameter count (e.g., "4B", "30B")
            - active_params: Active params for MoE models (e.g., "A3B"), None for standard

    Returns:
        The full HuggingFace model path (e.g., "Qwen/Qwen3-VL-4B-Instruct")
    """
    family = model_config.family
    name = resolve_model_name_from_config(model_config)

    # Map family to HuggingFace organization
    org_mapping = {
        "Qwen": "Qwen",
        "InternVL": "OpenGVLab",
    }
    org = org_mapping.get(family, family)

    return f"{org}/{name}"
