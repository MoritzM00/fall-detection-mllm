"""Score every class label per clip with teacher forcing (vLLM prompt logprobs).

Builds the DPO prompt and completion for each of the 16 labels and records their
log-probabilities under the model (optionally with a LoRA adapter). The output is
a prediction JSONL (``predicted_label`` = highest-scoring label) with an extra
``label_logprobs`` field per row, consumed by ``preference=scores``.

    python scripts/score_labels.py dataset=omnifall/video/oops data.mode=train \
        lora.path=<sft adapter> wandb.project=fall-detection-label-scores

Requests with prompt logprobs do not read the prefix cache (vLLM's default; reading
it returns corrupt prompt logprobs in vLLM 0.20), so every label costs a full prefill.
"""

import logging
import os
import time
from typing import cast

import hydra
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoProcessor

from falldet.config import resolve_model_path_from_config
from falldet.data.video_dataset import label2idx
from falldet.data.video_dataset_factory import get_video_datasets
from falldet.evaluation import evaluate_predictions
from falldet.inference import create_llm_engine
from falldet.inference.conversation import ConversationBuilder
from falldet.inference.label_scoring import LabelCompletions, ScoredOutput, score_clip
from falldet.schemas import from_dictconfig
from falldet.training.collator import completion_text
from falldet.training.dataset import answer_text, preference_prompt_config
from falldet.utils.logging import reconfigure_logging_after_wandb, setup_logging
from falldet.utils.predictions import save_predictions_jsonl
from falldet.utils.wandb import get_prediction_output_path, initialize_run_from_config

logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    _, rich_handler, file_handler = setup_logging(
        log_file="logs/local_logs.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    config = from_dictconfig(cfg)
    if config.vllm.use_mock:
        raise ValueError("Label scoring needs prompt logprobs from a real vLLM engine")
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
    individual = cast(dict[str, dict[str, Dataset]], multi_dataset)["individual"]
    if len(individual) != 1:
        raise ValueError(f"Label scoring supports one dataset, got {list(individual)}")
    dataset_name, dataset = next(iter(individual.items()))
    if config.num_samples is not None:
        dataset = Subset(dataset, range(min(config.num_samples, len(dataset))))
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=lambda batch: batch,
        shuffle=False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    # Same prompt and completion text as the DPO pairs
    conversation_builder = ConversationBuilder(
        config=preference_prompt_config(config.prompt),
        label2idx=label2idx,
        model_fps=config.model_fps,
        needs_video_metadata=config.model.needs_video_metadata,
    )
    logger.info(f"Prompt:\n{conversation_builder.user_prompt}")

    checkpoint_path = resolve_model_path_from_config(config.model)
    processor = AutoProcessor.from_pretrained(
        checkpoint_path, trust_remote_code=config.vllm.trust_remote_code
    )
    tokenizer = processor.tokenizer
    labels = tuple(label2idx)
    texts = [completion_text(answer_text(label), tokenizer.eos_token) for label in labels]
    completions = LabelCompletions.build(
        labels, texts, [tokenizer(text, add_special_tokens=False).input_ids for text in texts]
    )
    logger.info(
        f"Scoring {len(labels)} labels on {len(dataset)} clips; "
        f"{completions.shared_prefix} shared completion tokens are not scored"
    )

    from vllm import SamplingParams

    llm = create_llm_engine(config)
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)
    lora_request = None
    if config.lora.path is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(config.lora.name, 1, config.lora.path)

    predictions: list[dict] = []
    num_labels = len(labels)
    start = time.perf_counter()
    for batch in tqdm(dataloader, desc="Scoring batches"):
        requests = []
        for sample in batch:
            base = conversation_builder.build_vllm_inputs(sample["video"], processor)
            requests.append([base | {"prompt": base["prompt"] + text} for text in texts])

        flat: list[ScoredOutput] = llm.generate(
            [r for clip in requests for r in clip],
            sampling_params=params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        for i, sample in enumerate(batch):
            scores = score_clip(flat[i * num_labels : (i + 1) * num_labels], completions)
            if scores is None:
                raise RuntimeError(f"vLLM returned incomplete prompt logprobs for clip {i}")
            row = {k: v for k, v in sample.items() if k != "video"}
            row["predicted_label"] = max(scores, key=scores.__getitem__)
            row["label_logprobs"] = scores
            predictions.append(row)

    elapsed = time.perf_counter() - start
    logger.info(f"Scored {len(predictions)} clips in {elapsed:.1f}s")
    run.summary.update({"inference_time_seconds": elapsed})

    predictions_file = get_prediction_output_path(config.output_dir, config.wandb.project, run.id)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    save_predictions_jsonl(
        output_path=predictions_file,
        predictions=predictions,
        config=config.model_dump()
        | {
            "label_scoring": {
                "user_prompt": conversation_builder.user_prompt,
                "completions": dict(zip(labels, texts, strict=True)),
                "shared_prefix_tokens": completions.shared_prefix,
            }
        },
        wandb_run_id=run.id,
    )
    run.save(predictions_file.as_posix())

    evaluate_predictions(
        dataset=dataset,
        predictions=[row["predicted_label"] for row in predictions],
        references=[row["label_str"] for row in predictions],
        dataset_name=dataset_name,
        output_dir=config.output_dir,
        save_results=config.save_metrics,
        run=run,
        log_videos=0,
    )
    logger.info(f"Saved label scores to {predictions_file}")
    wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="inference_config")
def hydra_main(cfg: DictConfig):
    try:
        main(cfg)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True, extra={"markup": False})
        wandb.finish(exit_code=1)
        # sys.exit can hang on the vLLM engine process and keep the Slurm job alive
        os._exit(1)


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # default fork does not work!
    hydra_main()
