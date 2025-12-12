import json
import logging
import os

from infreqact.utils.logging import setup_logging

# Configure logging before importing heavy libraries
console, rich_handler, file_handler = setup_logging(
    log_file="logs/local_logs.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from infreqact.data.utils import load_test_omnifall_dataset
from infreqact.inference.base import parse_llm_outputs, prepare_inputs_for_vllm
from infreqact.inference.zeroshot import collate_fn
from infreqact.metrics.base import compute_metrics

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def main(batch_size=32, num_workers=8, num_samples=None, cot=False, verbose=5):
    """
    Run inference on Omnifall dataset using vLLM with batched processing.

    Args:
        batch_size: Number of samples to process in each batch
        num_workers: Number of workers for DataLoader
        num_samples: Number of samples to process (None for full dataset)
        cot: If True, request chain-of-thought reasoning
        verbose: Verbosity level for output printing
    """
    dataset = load_test_omnifall_dataset()

    # Limit dataset size if specified
    if num_samples is not None:
        dataset = Subset(dataset, range(min(num_samples, len(dataset))))

    logger.info(
        f"Processing {len(dataset)} samples with batch_size={batch_size}, num_workers={num_workers}"
    )
    logger.info(f"Chain-of-thought: {cot}")

    checkpoint_path = "Qwen/Qwen3-VL-4B-Instruct"
    logger.info(f"Loading model and processor: {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # Create DataLoader with custom collate function
    collate_fn_with_cot = partial(collate_fn, cot=cot)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_with_cot,
        shuffle=False,
        pin_memory=True,
    )

    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=2,
        mm_encoder_tp_mode="data",
        mm_processor_cache_gb=0,
        seed=0,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        mm_processor_kwargs={"min_pixels": 16 * 32 * 32, "max_pixels": 400 * 32 * 32},
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        top_k=-1,
        stop_token_ids=[],
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
        all_outputs, all_samples, verbose=verbose
    )

    predictions_file = "outputs/vllm_inference_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=4)
    logger.info(f"Saved predictions to {predictions_file}")

    # Compute comprehensive metrics
    logger.info("=" * 80)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 80)

    metrics = compute_metrics(y_pred=predicted_labels, y_true=true_labels)
    metrics_file = "outputs/vllm_inference_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_file}")

    # Log key metrics
    logger.info("")
    logger.info("📊 Overall Performance:")
    logger.info(f"  Accuracy:          {metrics['accuracy']:.3f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    logger.info(f"  Macro F1:          {metrics['macro_f1']:.3f}")

    logger.info("")
    logger.info("🚨 Fall Detection (Binary):")
    logger.info(f"  Sensitivity:  {metrics['fall_sensitivity']:.3f}")
    logger.info(f"  Specificity:  {metrics['fall_specificity']:.3f}")
    logger.info(f"  F1 Score:     {metrics['fall_f1']:.3f}")

    logger.info("")
    logger.info("🤕 Fallen Detection (Binary):")
    logger.info(f"  Sensitivity:  {metrics['fallen_sensitivity']:.3f}")
    logger.info(f"  Specificity:  {metrics['fallen_specificity']:.3f}")
    logger.info(f"  F1 Score:     {metrics['fallen_f1']:.3f}")

    logger.info("")
    logger.info("⚠️  Fall ∪ Fallen (Binary):")
    logger.info(f"  Sensitivity:  {metrics['fall_union_fallen_sensitivity']:.3f}")
    logger.info(f"  Specificity:  {metrics['fall_union_fallen_specificity']:.3f}")
    logger.info(f"  F1 Score:     {metrics['fall_union_fallen_f1']:.3f}")

    # Log per-class F1 scores for classes present in the test set
    logger.info("")
    logger.info("📈 Per-Class F1 Scores:")
    for key, value in sorted(metrics.items()):
        if key.endswith("_f1") and not key.startswith(("fall_", "fallen_", "fall_union")):
            class_name = key.replace("_f1", "")
            logger.info(f"  {class_name:15s}: {value:.3f}")

    logger.info("")
    logger.info("📦 Sample Counts:")
    logger.info(f"  Total: {metrics['sample_count']}")
    for key, value in sorted(metrics.items()):
        if key.startswith("sample_count_"):
            class_name = key.replace("sample_count_", "")
            logger.info(f"  {class_name:15s}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run vLLM inference on Omnifall dataset")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples to process (default: all)"
    )
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought reasoning")
    parser.add_argument(
        "--verbose", type=int, default=5, help="Number of samples to print (0 for none)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )

    args = parser.parse_args()

    # Set logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    main(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        cot=args.cot,
        verbose=args.verbose,
    )
