import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor

from infreqact.utils.logging import reconfigure_logging_after_wandb, setup_logging

# vLLM imports are done conditionally inside main() based on cfg.vllm.use_mock
# This allows switching between real and mock vLLM without code changes
try:
    from vllm import LLM, SamplingParams
except ImportError:
    # vLLM not installed, will use mock if configured
    LLM = None
    SamplingParams = None

import wandb
from infreqact.data.video_dataset import label2idx
from infreqact.data.video_dataset_factory import get_video_datasets
from infreqact.evaluation import evaluate_predictions
from infreqact.inference.base import parse_llm_outputs, prepare_inputs_for_vllm
from infreqact.inference.zeroshot import get_system_prompt
from infreqact.utils.wandb import initialize_run_from_config

logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """
    Run inference on video dataset using vLLM with batched processing.

    Args:
        cfg: Hydra configuration containing:
            - model: Model configuration (checkpoint, vllm settings, sampling params)
            - dataset: Dataset configuration (video paths, annotations, etc.)
            - inference: Inference settings (batch_size, num_workers, cot)
            - num_samples: Number of samples to process (None for full dataset)
            - verbose: Verbosity level for output printing
    """

    # Resolve all OmegaConf interpolations once at the beginning
    OmegaConf.resolve(cfg)

    logger.info("Configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Import real or mock vLLM based on configuration
    if cfg.vllm.get("use_mock", False):
        from infreqact.inference.mock_vllm import (
            MockLLM as LLM,
        )
        from infreqact.inference.mock_vllm import (
            MockSamplingParams as SamplingParams,
        )

        logger.warning("MOCK MODE ENABLED - Using Mock vLLM for debugging")

    else:
        from vllm import LLM, SamplingParams

    # Initialize Weights & Biases
    run = initialize_run_from_config(cfg)
    reconfigure_logging_after_wandb(rich_handler, file_handler)

    # Create config structure compatible with get_video_datasets
    # Wrap dataset config as dataset_test for the factory
    temp_cfg = OmegaConf.create({"dataset_test": cfg.dataset})

    multi_dataset = get_video_datasets(
        cfg=temp_cfg,
        mode=cfg.dataset.get("mode", "test"),
        run=run,
        return_individual=True,
        split=cfg.dataset.get("split", "cs"),
        size=(cfg.input_size.height, cfg.input_size.width),
    )
    for dataset_name, dataset in multi_dataset["individual"].items():
        # TODO: support multiple datasets in vLLM inference
        # we probably need to loop over datasets and aggregate predictions for metrics computation
        if len(multi_dataset["individual"]) > 1:
            logger.warning(
                "vLLM inference currently supports only a single dataset. "
                f"Found multiple datasets: {list(multi_dataset['individual'].keys())}. "
                f"Using the first one: {dataset_name}."
            )
        break

    # Limit dataset size if specified
    if cfg.get("num_samples") is not None:
        dataset = Subset(dataset, range(min(cfg.num_samples, len(dataset))))

    logger.info(
        f"Processing {len(dataset)} samples with batch_size={cfg.batch_size}, "
        f"num_workers={cfg.num_workers}"
    )
    logger.info(f"Chain-of-thought: {cfg.cot}")

    checkpoint_path = cfg.model.checkpoint_path
    logger.info(f"Loading model and processor: {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # collate fn should be a no-op (just a list of dict as return), pytorch collates over keys by default
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: batch,
        shuffle=False,
        pin_memory=True,
    )

    # Auto-determine tensor_parallel_size (null -> use all)
    tensor_parallel_size = cfg.vllm.tensor_parallel_size or torch.cuda.device_count()
    logger.info(f"Using tensor_parallel_size={tensor_parallel_size}")

    mm_processor_kwargs = OmegaConf.to_object(cfg.vllm.mm_processor_kwargs)
    # update model-specific overrides
    mm_processor_kwargs |= cfg.model.get("mm_processor_kwargs", {})
    logger.info(f"Using mm_processor_kwargs={mm_processor_kwargs}")

    vllm_kwargs = dict(
        model=checkpoint_path,
        tensor_parallel_size=tensor_parallel_size,
        mm_encoder_tp_mode=cfg.vllm.mm_encoder_tp_mode,
        mm_processor_cache_gb=cfg.vllm.mm_processor_cache_gb,
        seed=cfg.vllm.seed,
        dtype=cfg.vllm.dtype,
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        mm_processor_kwargs=mm_processor_kwargs,
        enable_expert_parallel=cfg.vllm.enable_expert_parallel,
        limit_mm_per_prompt=cfg.vllm.limit_mm_per_prompt,
        trust_remote_code=cfg.vllm.trust_remote_code,
        max_model_len=cfg.vllm.max_model_len,
        enforce_eager=cfg.vllm.enforce_eager,
        skip_mm_profiling=cfg.vllm.skip_mm_profiling,
        async_scheduling=cfg.vllm.async_scheduling,
    )

    # Add CoT flag for mock mode
    if cfg.vllm.get("use_mock", False):
        vllm_kwargs["cot"] = cfg.cot

    llm = LLM(**vllm_kwargs)

    sampling_params = SamplingParams(
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

    # Process in batches
    all_outputs = []
    all_samples = []

    prompt = get_system_prompt(cot=cfg.cot)

    logger.info("Generating predictions...")
    start = time.perf_counter()
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_inputs = []
        batch_samples = []
        for sample in batch:
            frames = sample["video"]
            metadata = {k: v for k, v in sample.items() if k != "video"}
            batch_samples.append(metadata)
            message = {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
            inputs = prepare_inputs_for_vllm(
                frames,
                message,
                processor,
                model_fps=cfg.model_fps,
                needs_video_metadata=cfg.model.needs_video_metadata,
            )

            batch_inputs.append(inputs)

        batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        all_outputs.extend(batch_outputs)
        all_samples.extend(batch_samples)

    end = time.perf_counter()
    logger.info(f"Inference completed in {end - start:.2f} seconds")
    run.summary["inference_time_seconds"] = end - start

    predictions, predicted_labels, true_labels = parse_llm_outputs(
        all_outputs, all_samples, label2idx
    )
    logger.info(f"Unique predicted labels: {set(predicted_labels)}")
    logger.info(f"Unique true labels: {set(true_labels)}")

    # Save predictions if enabled
    predictions_file = None
    if cfg.get("save_predictions", True):
        predictions_dir = Path(cfg.output_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        cot_suffix = "_cot" if cfg.cot else ""
        predictions_file = (
            predictions_dir / f"{cfg.model.name}_{dataset_name}_predictions{cot_suffix}.json"
        )
        with open(predictions_file, "w") as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"Saved predictions to {predictions_file}")

    evaluate_predictions(
        dataset=dataset,
        predictions=predicted_labels,
        references=true_labels,
        dataset_name=dataset_name,
        output_dir=cfg.output_dir,
        save_results=cfg.get("save_metrics", True),
        run=run,
        log_videos=cfg.get("log_videos", 0),
    )

    logger.info(f"Logged results to W&B: {run.url}")
    wandb.finish()

    if tensor_parallel_size > 1:
        from vllm.distributed import destroy_distributed_environment

        destroy_distributed_environment()


@hydra.main(version_base=None, config_path="../config", config_name="inference_config")
def hydra_main(cfg: DictConfig):
    """Hydra entry point for the inference script."""
    global console, rich_handler, file_handler
    console, rich_handler, file_handler = setup_logging(
        log_file="logs/local_logs.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    try:
        main(cfg)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    hydra_main()
