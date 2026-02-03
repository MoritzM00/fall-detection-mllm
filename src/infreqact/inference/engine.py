"""
VLLM engine initialization and configuration.

This module provides factory functions to create vLLM engines and sampling
parameters from Hydra configuration. It handles:
- Conditional selection of real vLLM or mock engine
- Model path resolution from structured config
- Tensor parallelism auto-configuration
- Merging of model-specific overrides
"""

import logging
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig, OmegaConf

from infreqact.config import is_moe_model, resolve_model_path_from_config

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def create_llm_engine(cfg: DictConfig) -> "LLM":
    """
    Create and initialize a vLLM engine (real or mock) based on config.

    Args:
        cfg: Full Hydra config containing 'vllm', 'model', and 'prompt' sections

    Returns:
        Initialized LLM instance (real vLLM or MockLLM)
    """
    # Import real or mock vLLM based on configuration
    use_mock = cfg.vllm.get("use_mock", False)
    if use_mock:
        from infreqact.inference.mock_vllm import MockLLM as LLM

        logger.warning("MOCK MODE ENABLED - Using Mock vLLM for debugging")
    else:
        from vllm import LLM

    # Resolve checkpoint path from model config
    checkpoint_path = resolve_model_path_from_config(cfg.model)
    logger.info(f"Loading model: {checkpoint_path}")

    # Auto-determine tensor_parallel_size (null -> use all GPUs)
    tensor_parallel_size = cfg.vllm.tensor_parallel_size or torch.cuda.device_count()
    logger.info(f"Using tensor_parallel_size={tensor_parallel_size}")

    # Build mm_processor_kwargs with model-specific overrides
    mm_processor_kwargs = OmegaConf.to_object(cfg.vllm.mm_processor_kwargs)
    mm_processor_kwargs |= cfg.model.get("mm_processor_kwargs", {})
    logger.info(f"Using mm_processor_kwargs={mm_processor_kwargs}")

    enable_expert_parallel = cfg.vllm.enable_expert_parallel or is_moe_model(cfg.model)

    # Compute dynamic video limit based on num_shots
    num_shots = cfg.prompt.get("num_shots", 0)
    video_limit = num_shots + 1
    limit_mm_per_prompt = {"image": 0, "video": max(1, video_limit)}
    if num_shots > 0:
        logger.info(
            f"Few-shot mode: {num_shots} exemplars, limit_mm_per_prompt={limit_mm_per_prompt}"
        )

    # Build vLLM kwargs
    vllm_kwargs = dict(
        model=checkpoint_path,
        tensor_parallel_size=tensor_parallel_size,
        mm_encoder_tp_mode=cfg.vllm.mm_encoder_tp_mode,
        mm_processor_cache_gb=cfg.vllm.mm_processor_cache_gb,
        seed=cfg.vllm.seed,
        dtype=cfg.vllm.dtype,
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        mm_processor_kwargs=mm_processor_kwargs,
        enable_expert_parallel=enable_expert_parallel,
        limit_mm_per_prompt=limit_mm_per_prompt,
        trust_remote_code=cfg.vllm.trust_remote_code,
        max_model_len=cfg.vllm.max_model_len,
        max_num_batched_tokens=cfg.vllm.max_num_batched_tokens,
        enforce_eager=cfg.vllm.enforce_eager,
        skip_mm_profiling=cfg.vllm.skip_mm_profiling,
        async_scheduling=cfg.vllm.async_scheduling,
        enable_prefix_caching=cfg.vllm.get("enable_prefix_caching", True),
    )

    # Add CoT flag and output format for mock mode
    if use_mock:
        vllm_kwargs["cot"] = cfg.prompt.cot
        vllm_kwargs["output_format"] = cfg.prompt.get("output_format", "json")

    return LLM(**vllm_kwargs)


def create_sampling_params(cfg: DictConfig) -> "SamplingParams":
    """
    Create sampling parameters based on config.

    Args:
        cfg: Full Hydra config containing 'vllm' and 'sampling' sections

    Returns:
        SamplingParams instance (real or Mock)
    """
    # Import real or mock SamplingParams based on configuration
    if cfg.vllm.get("use_mock", False):
        from infreqact.inference.mock_vllm import MockSamplingParams as SamplingParams
    else:
        from vllm import SamplingParams

    return SamplingParams(
        temperature=cfg.sampling.temperature,
        max_tokens=cfg.sampling.max_tokens,
        top_k=cfg.sampling.top_k,
        top_p=cfg.sampling.get("top_p", 1.0),
        presence_penalty=cfg.sampling.get("presence_penalty", 0.0),
        frequency_penalty=cfg.sampling.get("frequency_penalty", 0.0),
        repetition_penalty=cfg.sampling.get("repetition_penalty", 1.0),
        seed=cfg.sampling.get("seed", None),
        stop_token_ids=cfg.sampling.stop_token_ids,
    )
