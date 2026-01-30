"""Configuration utilities for model path resolution."""

import logging

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def is_moe_model(model_config: DictConfig) -> bool:
    """
    Determine if the model is a Mixture of Experts (MoE) model based on config.

    Args:
        model_config: Model configuration (DictConfig or dict-like object) with fields:
            - active_params: Active params for MoE models (e.g., "A3B"), None for standard

    Returns:
        True if the model is an MoE model, False otherwise.
    """
    return model_config.get("active_params") is not None


def resolve_model_name_from_config(model_config: DictConfig) -> str:
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

    # handle MOE
    params_str = params
    if active_params:
        params_str += f"-{active_params}"

    match family.lower():
        case "qwen":
            path = f"{family}{version}-VL-{params_str}-{variant}"
        case "internvl":
            if variant is not None:
                logger.warning("InternVL models do not use variants; ignoring variant field.")
            path = f"{family}{version}-{params_str}-HF"
        case "molmo":
            path = f"{family}{version}-{params_str}"
        case _:
            path = f"{family}-{version}-{params_str}"
    return path


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
    name = resolve_model_name_from_config(model_config)
    org = model_config.org
    return f"{org}/{name}"
