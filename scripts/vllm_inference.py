import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import wandb
from infreqact.data.video_dataset import label2idx
from infreqact.data.video_dataset_factory import get_video_datasets
from infreqact.evaluation import evaluate_predictions
from infreqact.inference.base import parse_llm_outputs, prepare_inputs_for_vllm
from infreqact.inference.zeroshot import collate_fn
from infreqact.utils.logging import reconfigure_logging_after_wandb, setup_logging
from infreqact.utils.wandb import initialize_run_from_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
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
    console, rich_handler, file_handler = setup_logging(
        log_file="logs/local_logs.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )

    # Resolve all OmegaConf interpolations once at the beginning
    OmegaConf.resolve(cfg)

    logger.info("Configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

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
    )
    for dataset_name, dataset in multi_dataset["individual"].items():
        # TODO: support multiple datasets in vLLM inference
        # we probably need to loop over datasets and aggregate predictions for metrics computation
        if len(dataset["individual"]) > 1:
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
    model_name = cfg.model.name
    logger.info(f"Loading model and processor: {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # Create DataLoader with custom collate function
    collate_fn_with_cot = partial(collate_fn, cot=cfg.cot)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_with_cot,
        shuffle=False,
        pin_memory=True,
    )

    # Auto-determine tensor_parallel_size (null -> use all)
    tensor_parallel_size = cfg.vllm.tensor_parallel_size
    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()
    logger.info(f"Using tensor_parallel_size={tensor_parallel_size}")

    # Initialize vLLM model with config parameters
    vllm_kwargs = {
        "model": checkpoint_path,
        "tensor_parallel_size": tensor_parallel_size,
        "mm_encoder_tp_mode": cfg.vllm.mm_encoder_tp_mode,
        "mm_processor_cache_gb": cfg.vllm.mm_processor_cache_gb,
        "seed": cfg.vllm.seed,
        "dtype": getattr(torch, cfg.vllm.dtype),
        "gpu_memory_utilization": cfg.vllm.gpu_memory_utilization,
        "mm_processor_kwargs": cfg.vllm.mm_processor_kwargs,
    }

    # Add enable_expert_parallel only if it's set to True (for MoE models)
    if cfg.vllm.get("enable_expert_parallel", False):
        vllm_kwargs["enable_expert_parallel"] = True
        logger.info("Enabling expert parallelism for MoE model")

    llm = LLM(**vllm_kwargs)

    sampling_params = SamplingParams(
        temperature=cfg.sampling.temperature,
        max_tokens=cfg.sampling.max_tokens,
        top_k=cfg.sampling.top_k,
        stop_token_ids=cfg.sampling.stop_token_ids,
    )

    # Process in batches
    all_outputs = []
    all_samples = []

    logger.info("Generating predictions...")
    for batch_messages, batch_samples in tqdm(dataloader, desc="Processing batches"):
        # Prepare inputs for vLLM
        batch_inputs = [prepare_inputs_for_vllm([msg], processor) for msg in batch_messages]

        # Generate predictions for this batch
        batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        all_outputs.extend(batch_outputs)
        all_samples.extend(batch_samples)

    logger.info(f"Generated {len(all_outputs)} predictions")

    predictions, predicted_labels, true_labels = parse_llm_outputs(
        all_outputs, all_samples, label2idx
    )

    # Create filename components for reuse
    cot_suffix = "_cot" if cfg.cot else ""

    # Save predictions if enabled
    predictions_file = None
    if cfg.get("save_predictions", True):
        predictions_dir = Path(cfg.output_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        predictions_file = (
            predictions_dir / f"{cfg.model.name}_{dataset_name}_predictions{cot_suffix}.json"
        )
        with open(predictions_file, "w") as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"Saved predictions to {predictions_file}")

    # Compute comprehensive metrics
    logger.info("=" * 80)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 80)

    # Create metrics subdirectory
    metrics_dir = Path(cfg.output_dir) / "metrics" if cfg.get("save_metrics", True) else None

    evaluate_predictions(
        dataset=dataset,
        predictions=predicted_labels,
        references=true_labels,
        dataset_name=dataset_name,
        output_dir=str(metrics_dir) if metrics_dir else None,
        save_results=cfg.get("save_metrics", True),
        run=run,
    )

    # Log predictions as artifact
    if predictions_file is not None:
        artifact = wandb.Artifact(
            name=f"{model_name}_{dataset_name}_predictions",
            type="predictions",
            description=f"Predictions for {model_name} on {dataset_name}",
        )
        artifact.add_file(str(predictions_file))
        run.log_artifact(artifact)

    # Log metrics file as artifact
    if cfg.get("save_metrics", True) and metrics_dir:
        # Find the most recent metrics file
        metrics_files = list(metrics_dir.glob(f"{model_name}_{dataset_name}_*.json"))
        if metrics_files:
            latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
            artifact = wandb.Artifact(
                name=f"{model_name}_{dataset_name}_metrics",
                type="metrics",
                description=f"Evaluation metrics for {model_name} on {dataset_name}",
            )
            artifact.add_file(str(latest_metrics))
            run.log_artifact(artifact)

    logger.info(f"Logged results to W&B: {run.url}")
    wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="inference_config")
def hydra_main(cfg: DictConfig):
    """Hydra entry point for the inference script."""
    try:
        main(cfg)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    hydra_main()
