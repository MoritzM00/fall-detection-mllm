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

from falldet.config import is_moe_model, resolve_model_path_from_config
from falldet.schemas import InferenceConfig

if TYPE_CHECKING:  # vllm is heavy and conditionally imported at runtime
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def create_llm_engine(config: InferenceConfig) -> "LLM":
    """
    Create and initialize a vLLM engine (real or mock) based on config.

    Args:
        config: Full validated inference configuration

    Returns:
        Initialized LLM instance (real vLLM or MockLLM)
    """
    # Import real or mock vLLM based on configuration
    use_mock = config.vllm.use_mock
    if use_mock:
        from falldet.inference.mock_vllm import MockLLM as LLM

        logger.warning("MOCK MODE ENABLED - Using Mock vLLM for debugging")
    else:
        from vllm import LLM

    # Resolve checkpoint path from model config
    checkpoint_path = resolve_model_path_from_config(config.model)
    logger.info(f"Loading model: {checkpoint_path}")

    # Auto-determine tensor_parallel_size (null -> use all GPUs)
    tensor_parallel_size = config.vllm.tensor_parallel_size or torch.cuda.device_count()
    logger.info(f"Using tensor_parallel_size={tensor_parallel_size}")

    # Build mm_processor_kwargs with model-specific overrides
    mm_processor_kwargs = dict(config.vllm.mm_processor_kwargs)
    mm_processor_kwargs |= config.model.mm_processor_kwargs
    logger.info(f"Using mm_processor_kwargs={mm_processor_kwargs}")

    enable_expert_parallel = config.vllm.enable_expert_parallel or is_moe_model(config.model)

    # for few-shot prompting, increase mm_processor_cache_gb and enable prefix_caching
    if config.prompt.num_shots > 0:
        enable_prefix_caching = True
        mm_processor_cache_gb = max(
            4, config.vllm.mm_processor_cache_gb
        )  # ensure at least 4GB cache for fewshot
    else:
        enable_prefix_caching = config.vllm.enable_prefix_caching
        mm_processor_cache_gb = config.vllm.mm_processor_cache_gb

    # Use schema default, but override video limit for few-shot
    limit_mm_per_prompt = dict(config.vllm.limit_mm_per_prompt)
    num_shots = config.prompt.num_shots
    if num_shots > 0:
        limit_mm_per_prompt["video"] = max(limit_mm_per_prompt.get("video", 1), num_shots + 1)
        logger.info(
            f"Few-shot mode: {num_shots} exemplars, limit_mm_per_prompt={limit_mm_per_prompt}"
        )

    # Build vLLM kwargs
    vllm_kwargs = dict(
        model=checkpoint_path,
        tensor_parallel_size=tensor_parallel_size,
        mm_encoder_tp_mode=config.vllm.mm_encoder_tp_mode,
        mm_processor_cache_gb=mm_processor_cache_gb,
        seed=config.vllm.seed,
        dtype=config.vllm.dtype,
        gpu_memory_utilization=config.vllm.gpu_memory_utilization,
        mm_processor_kwargs=mm_processor_kwargs,
        enable_expert_parallel=enable_expert_parallel,
        limit_mm_per_prompt=limit_mm_per_prompt,
        trust_remote_code=config.vllm.trust_remote_code,
        max_model_len=config.vllm.max_model_len,
        max_num_batched_tokens=config.vllm.max_num_batched_tokens,
        enforce_eager=config.vllm.enforce_eager,
        skip_mm_profiling=config.vllm.skip_mm_profiling,
        async_scheduling=config.vllm.async_scheduling,
        enable_prefix_caching=enable_prefix_caching,
    )

    # Add CoT flag and output format for mock mode
    if use_mock:
        vllm_kwargs["cot"] = config.prompt.cot
        vllm_kwargs["output_format"] = config.prompt.output_format

    return LLM(**vllm_kwargs)


def create_sampling_params(config: InferenceConfig) -> "SamplingParams":
    """
    Create sampling parameters based on config.

    Args:
        config: Full validated inference configuration

    Returns:
        SamplingParams instance (real or Mock)
    """
    # Import real or mock SamplingParams based on configuration
    if config.vllm.use_mock:
        from falldet.inference.mock_vllm import MockSamplingParams as SamplingParams
    else:
        from vllm import SamplingParams

    return SamplingParams(
        temperature=config.sampling.temperature,
        max_tokens=config.sampling.max_tokens,
        top_k=config.sampling.top_k,
        top_p=config.sampling.top_p,
        presence_penalty=config.sampling.presence_penalty,
        frequency_penalty=config.sampling.frequency_penalty,
        repetition_penalty=config.sampling.repetition_penalty,
        seed=config.sampling.seed,
        stop_token_ids=config.sampling.stop_token_ids,
    )
