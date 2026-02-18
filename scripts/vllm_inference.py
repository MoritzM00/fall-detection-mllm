import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor

import wandb
from falldet.config import resolve_model_name_from_config, resolve_model_path_from_config
from falldet.data.video_dataset import label2idx
from falldet.data.video_dataset_factory import get_video_datasets
from falldet.evaluation import evaluate_predictions
from falldet.inference import (
    create_conversation_builder,
    create_llm_engine,
    create_sampling_params,
)
from falldet.schemas import from_dictconfig
from falldet.utils.logging import reconfigure_logging_after_wandb, setup_logging
from falldet.utils.predictions import save_predictions_jsonl
from falldet.utils.wandb import initialize_run_from_config

logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """
    Run inference on video dataset using vLLM with batched processing.

    Args:
        cfg: Hydra configuration containing:
    """
    config = from_dictconfig(cfg)
    logger.info(config.model_dump_json(indent=2))

    run = initialize_run_from_config(config)
    reconfigure_logging_after_wandb(rich_handler, file_handler)

    multi_dataset = get_video_datasets(
        config=config,
        mode=config.data.mode,
        run=run,
        return_individual=True,
        split=config.data.split,
        size=config.data.size,
        max_size=config.data.max_size,
        seed=config.data.seed,
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
    if config.num_samples is not None:
        dataset = Subset(dataset, range(min(config.num_samples, len(dataset))))

    logger.info(
        f"Processing {len(dataset)} samples with batch_size={config.batch_size}, "
        f"num_workers={config.num_workers}"
    )
    # collate fn should be a no-op (just a list of dict as return), pytorch collates over keys by default
    prefetch_factor = config.prefetch_factor if config.num_workers > 0 else None
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=lambda batch: batch,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    # Build conversation builder (handles both zero-shot and few-shot)
    conversation_builder = create_conversation_builder(config, label2idx)
    parser = conversation_builder.parser

    logger.info(
        f"Mode: {config.prompt.num_shots}-shot ({conversation_builder.num_videos} videos/request)"
    )
    logger.info(f"Prompt:\n{conversation_builder.user_prompt}")

    checkpoint_path = resolve_model_path_from_config(config.model)
    logger.info(f"Loading model and processor: {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(
        checkpoint_path, trust_remote_code=config.vllm.trust_remote_code
    )
    # Initialize vLLM engine and sampling params
    llm = create_llm_engine(config)
    sampling_params = create_sampling_params(config)

    logger.info("Generating predictions...")

    # Process in batches
    all_outputs = []
    all_samples = []
    start = time.perf_counter()
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_inputs = []
        batch_samples = []
        for sample in batch:
            metadata = {k: v for k, v in sample.items() if k != "video"}
            batch_samples.append(metadata)

            # Build vLLM inputs using conversation builder (handles zero/few-shot uniformly)
            inputs = conversation_builder.build_vllm_inputs(sample["video"], processor)
            batch_inputs.append(inputs)

        batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        all_outputs.extend(batch_outputs)
        all_samples.extend(batch_samples)

    end = time.perf_counter()
    logger.info(f"Inference completed in {end - start:.2f} seconds")
    run.summary["inference_time_seconds"] = end - start

    # Parse outputs using the configured parser
    predictions = []
    predicted_labels = []
    true_labels = []

    for i, (output, sample) in enumerate(zip(all_outputs, all_samples)):
        generated_text = output.outputs[0].text
        true_labels.append(sample["label_str"])

        # Parse using the configured parser
        result = parser.parse(generated_text)
        predicted_labels.append(result.label)

        # Build prediction dict
        prediction = sample.copy()
        prediction["predicted_label"] = result.label
        prediction["reasoning"] = result.reasoning or ""
        prediction["raw_output"] = result.raw_text
        predictions.append(prediction)

    logger.info(f"Unique predicted labels: {set(predicted_labels)}")
    logger.info(f"Unique true labels: {set(true_labels)}")

    # Save predictions if enabled
    predictions_file = None
    if config.save_predictions:
        predictions_dir = Path(config.output_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamp to filename to avoid overwriting previous runs
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = resolve_model_name_from_config(config.model)
        predictions_file = predictions_dir / f"{model_name}_{dataset_name}_{timestamp}.jsonl"

        # Get wandb run ID if available
        wandb_run_id = run.id if run else None

        save_predictions_jsonl(
            output_path=predictions_file,
            model_name=model_name,
            dataset_name=dataset_name,
            predictions=predictions,
            config=config.model_dump(),
            wandb_run_id=wandb_run_id,
        )
        run.save(predictions_file.as_posix())

    evaluate_predictions(
        dataset=dataset,
        predictions=predicted_labels,
        references=true_labels,
        dataset_name=dataset_name,
        output_dir=config.output_dir,
        save_results=config.save_metrics,
        run=run,
        log_videos=config.log_videos,
    )

    logger.info(f"Saved predictions to {predictions_file}")
    logger.info(f"Logged results to W&B: {run.url}")
    wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="inference_config")
def hydra_main(cfg: DictConfig):
    """Hydra entry point for the inference script."""
    global console, rich_handler, file_handler
    console, rich_handler, file_handler = setup_logging(
        log_file="logs/local_logs.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    try:
        main(cfg)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # default fork does not work!
    hydra_main()
