"""Hydra-driven video DPO for Qwen3-VL Instruct.

With ``dpo.sft_adapter_path=null``, training starts from the pretrained Instruct
model with a fresh LoRA. Set the path to continue from an existing SFT LoRA.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from omegaconf import DictConfig
from peft import LoraConfig as PeftLoraConfig
from peft import PeftConfig, PeftModel
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import DPOConfig

from falldet.data.video_dataset import label2idx
from falldet.data.video_dataset_factory import get_video_datasets
from falldet.embeddings import load_embeddings
from falldet.inference.conversation import ConversationBuilder
from falldet.schemas import DPOTrainingConfig, from_dictconfig_dpo
from falldet.training.collator import VideoPreferenceCollator
from falldet.training.dataset import DPOConversationDataset, as_lazy_hf_dataset
from falldet.training.dpo_trainer import VideoDPOTrainer
from falldet.training.eval_sampling import stratified_sample_indices
from falldet.training.preferences import (
    EmbeddingSimilarityNegativeSelector,
    NegativeSelector,
    RandomNegativeSelector,
    align_embeddings_to_dataset,
    label_for_index,
    observed_labels,
)
from falldet.utils.logging import disable_logging_for_non_main_process, setup_logging
from falldet.utils.wandb import initialize_run_from_config, log_adapter_artifact

logger = logging.getLogger(__name__)


def _validate_adapter(path: Path, expected_base: str) -> PeftLoraConfig:
    if not path.is_dir():
        raise ValueError(f"SFT adapter directory does not exist: {path}")
    config = PeftConfig.from_pretrained(path)
    if not isinstance(config, PeftLoraConfig):
        raise ValueError(f"DPO supports standard LoRA adapters, got {config.peft_type}")
    if config.base_model_name_or_path != expected_base:
        raise ValueError(
            f"Adapter base model {config.base_model_name_or_path!r} does not match {expected_base!r}"
        )
    if config.target_parameters:
        raise ValueError("LoRA target_parameters are unsupported for DPO")
    if config.modules_to_save:
        raise ValueError("LoRA modules_to_save are unsupported for DPO")
    if config.bias != "none":
        raise ValueError("DPO requires a LoRA adapter with bias='none'")
    return config


def _set_fail_fast_video_loading(dataset: Dataset) -> None:
    """Prevent random replacement videos from changing preference row identity."""

    children = getattr(dataset, "datasets", None)
    if children is not None:
        for child in children:
            _set_fail_fast_video_loading(child)
    if hasattr(dataset, "max_retries"):
        dataset.max_retries = 1


def _label_counts(dataset: Dataset) -> dict[str, int]:
    length = len(dataset)  # ty: ignore[invalid-argument-type]
    return dict(Counter(label_for_index(dataset, index) for index in range(length)))


def _adapter_rank(config: PeftLoraConfig) -> int:
    ranks = [config.r, *(config.rank_pattern or {}).values()]
    return max(int(rank) for rank in ranks)


def _labels_for_rows(dataset: Dataset) -> tuple[str, ...]:
    length = len(dataset)  # ty: ignore[invalid-argument-type]
    return tuple(label_for_index(dataset, index) for index in range(length))


def _similarity_summary(
    train_selector: EmbeddingSimilarityNegativeSelector,
    validation_selector: EmbeddingSimilarityNegativeSelector,
    train_path: Path,
    validation_path: Path,
) -> dict:
    return {
        "preference_strategy": "similarity",
        "train_embeddings_path": str(train_path),
        "validation_embeddings_path": str(validation_path),
        "train_negative_similarity_mean": sum(train_selector.scores) / len(train_selector.scores),
        "validation_negative_similarity_mean": sum(validation_selector.scores)
        / len(validation_selector.scores),
        "train_negative_similarity_min": min(train_selector.scores),
        "train_negative_similarity_max": max(train_selector.scores),
        "validation_negative_similarity_min": min(validation_selector.scores),
        "validation_negative_similarity_max": max(validation_selector.scores),
    }


def _write_similarity_manifest(
    output_dir: Path,
    train_selector: EmbeddingSimilarityNegativeSelector,
    validation_selector: EmbeddingSimilarityNegativeSelector,
) -> Path:
    path = output_dir / "preference_mining.jsonl"
    with path.open("w") as file:
        for split, selector in (
            ("train", train_selector),
            ("validation", validation_selector),
        ):
            for query_index, (positive, negative, corpus_index, score) in enumerate(
                zip(
                    selector.query_labels,
                    selector.negative_labels,
                    selector.corpus_indices,
                    selector.scores,
                    strict=True,
                )
            ):
                record = {
                    "split": split,
                    "query_index": query_index,
                    "positive_label": positive,
                    "corpus_index": corpus_index,
                    "negative_label": negative,
                    "cosine_similarity": score,
                }
                file.write(json.dumps(record) + "\n")
    return path


def _build_negative_selectors(
    config: DPOTrainingConfig,
    train_dataset: Dataset,
    validation_dataset: Dataset,
) -> tuple[NegativeSelector, NegativeSelector, dict]:
    train_labels = _labels_for_rows(train_dataset)
    validation_labels = _labels_for_rows(validation_dataset)
    label_universe = tuple(sorted(set(train_labels), key=label2idx.__getitem__))

    if config.preference.strategy == "random":
        return (
            RandomNegativeSelector(label_universe, seed=config.preference.seed),
            RandomNegativeSelector(label_universe, seed=config.preference.seed),
            {"preference_strategy": "random"},
        )

    assert config.preference.train_embeddings_path is not None
    assert config.preference.validation_embeddings_path is not None
    train_path = Path(config.preference.train_embeddings_path).expanduser().resolve()
    validation_path = Path(config.preference.validation_embeddings_path).expanduser().resolve()
    train_embeddings, train_samples = load_embeddings(train_path)
    validation_embeddings, validation_samples = load_embeddings(validation_path)
    train_embeddings = align_embeddings_to_dataset(train_dataset, train_embeddings, train_samples)
    validation_embeddings = align_embeddings_to_dataset(
        validation_dataset, validation_embeddings, validation_samples
    )

    train_selector = EmbeddingSimilarityNegativeSelector(
        query_embeddings=train_embeddings,
        corpus_embeddings=train_embeddings,
        query_labels=train_labels,
        corpus_labels=train_labels,
        chunk_size=config.preference.chunk_size,
    )
    validation_selector = EmbeddingSimilarityNegativeSelector(
        query_embeddings=validation_embeddings,
        corpus_embeddings=train_embeddings,
        query_labels=validation_labels,
        corpus_labels=train_labels,
        chunk_size=config.preference.chunk_size,
    )
    return (
        train_selector,
        validation_selector,
        _similarity_summary(train_selector, validation_selector, train_path, validation_path),
    )


@hydra.main(config_path="../config", config_name="dpo_config", version_base=None)
def main(cfg: DictConfig) -> None:
    state = PartialState()
    setup_logging(
        log_file="logs/dpo_training.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    disable_logging_for_non_main_process(state.local_process_index)
    config: DPOTrainingConfig = from_dictconfig_dpo(cfg)
    logger.info(config.model_dump_json(indent=2))

    adapter_path = (
        Path(config.dpo.sft_adapter_path).expanduser().resolve()
        if config.dpo.sft_adapter_path is not None
        else None
    )
    adapter_config = (
        _validate_adapter(adapter_path, config.model.path) if adapter_path is not None else None
    )
    rank = _adapter_rank(adapter_config) if adapter_config is not None else config.lora.r

    run = initialize_run_from_config(config)
    run_name = run.name
    output_dir = Path(config.output_dir) / run_name
    adapter_dir = output_dir / "adapter"
    if state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    state.wait_for_everyone()

    logger.info(f"Loading processor and base model: {config.model.path}")
    processor = AutoProcessor.from_pretrained(config.model.path, trust_remote_code=True)
    model_kwargs: dict = {"dtype": torch.bfloat16, "trust_remote_code": True}
    if config.dpo.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.dpo.attn_implementation
    base_model = AutoModelForImageTextToText.from_pretrained(config.model.path, **model_kwargs)
    base_model.config.use_cache = False
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        peft_config = None
        initialization_source = "sft_adapter"
    else:
        model = base_model
        peft_config = PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            target_modules=list(config.lora.target_modules),
        )
        initialization_source = "pretrained_instruct"

    prompt_config = config.prompt.model_copy(
        update={"labels": list(label2idx), "output_format": "text", "num_shots": 0, "cot": False}
    )
    conversation_builder = ConversationBuilder(
        config=prompt_config,
        label2idx=label2idx,
        model_fps=config.model_fps,
        needs_video_metadata=config.model.needs_video_metadata,
    )

    with state.main_process_first():
        train_base = get_video_datasets(
            config=config,
            mode="train",
            run=run,
            return_individual=False,
            split=config.data.split,
            size=config.data.size,
            max_size=config.data.max_size,
            seed=config.data.seed,
        )
        val_base = get_video_datasets(
            config=config,
            mode="val",
            run=run,
            return_individual=False,
            split=config.data.split,
            size=config.data.size,
            max_size=config.data.max_size,
            seed=0,
        )
    _set_fail_fast_video_loading(train_base)
    _set_fail_fast_video_loading(val_base)

    if config.dpo.max_eval_samples_per_ds is not None:
        count = min(config.dpo.max_eval_samples_per_ds, len(val_base))
        indices = stratified_sample_indices(val_base, count, seed=0)
        val_base = Subset(val_base, indices)
        if state.is_main_process:
            (output_dir / "validation_subset_indices.json").write_text(
                json.dumps(indices, indent=2) + "\n"
            )

    train_labels = observed_labels(train_base)
    validation_labels = set(observed_labels(val_base))
    missing_labels = validation_labels - set(train_labels)
    if missing_labels:
        raise ValueError(
            f"Validation labels absent from the training label universe: {sorted(missing_labels)}"
        )

    train_selector, validation_selector, preference_summary = _build_negative_selectors(
        config, train_base, val_base
    )
    if (
        state.is_main_process
        and isinstance(train_selector, EmbeddingSimilarityNegativeSelector)
        and isinstance(validation_selector, EmbeddingSimilarityNegativeSelector)
    ):
        mining_manifest = _write_similarity_manifest(
            output_dir, train_selector, validation_selector
        )
        preference_summary["preference_mining_manifest"] = str(mining_manifest)
    train_dataset = as_lazy_hf_dataset(
        DPOConversationDataset(train_base, conversation_builder, train_selector)
    )
    eval_dataset = as_lazy_hf_dataset(
        DPOConversationDataset(val_base, conversation_builder, validation_selector)
    )
    collator = VideoPreferenceCollator(
        processor,
        needs_video_metadata=config.model.needs_video_metadata,
    )

    effective_batch = (
        config.dpo.per_device_train_batch_size
        * config.dpo.gradient_accumulation_steps
        * state.num_processes
    )
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = (
        config.dpo.max_steps
        if config.dpo.max_steps > 0
        else int(steps_per_epoch * config.dpo.num_train_epochs)
    )
    warmup_steps = (
        int(round(config.dpo.warmup_ratio * total_steps))
        if config.dpo.warmup_ratio > 0
        else config.dpo.warmup_steps
    )

    args = DPOConfig(
        output_dir=str(output_dir),
        beta=config.dpo.beta,
        loss_type=config.dpo.loss_type,
        per_device_train_batch_size=config.dpo.per_device_train_batch_size,
        per_device_eval_batch_size=config.dpo.per_device_eval_batch_size,
        gradient_accumulation_steps=config.dpo.gradient_accumulation_steps,
        num_train_epochs=config.dpo.num_train_epochs,
        max_steps=config.dpo.max_steps,
        learning_rate=config.dpo.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.dpo.weight_decay,
        max_grad_norm=config.dpo.max_grad_norm,
        lr_scheduler_type=config.dpo.lr_scheduler_type,
        bf16=config.dpo.bf16,
        fp16=config.dpo.fp16,
        logging_steps=config.dpo.logging_steps,
        save_strategy=config.dpo.save_strategy,
        save_steps=config.dpo.save_steps,
        save_total_limit=config.dpo.save_total_limit,
        eval_strategy=config.dpo.eval_strategy,
        eval_steps=config.dpo.eval_steps,
        eval_on_start=config.dpo.eval_on_start,
        load_best_model_at_end=config.dpo.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=config.dpo.gradient_checkpointing,
        max_length=None,
        report_to=config.dpo.report_to,
        seed=config.dpo.seed,
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=config.pin_memory,
        dataloader_persistent_workers=config.persistent_workers and config.num_workers > 0,
        dataloader_prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        precompute_ref_log_probs=False,
        use_liger_kernel=False,
        padding_free=False,
        optim=config.dpo.optim,
        adam_beta1=config.dpo.adam_beta1,
        adam_beta2=config.dpo.adam_beta2,
        adam_epsilon=config.dpo.adam_epsilon,
        deepspeed=config.dpo.deepspeed,
    )
    trainer = VideoDPOTrainer(
        model=model,
        ref_model=None,
        peft_config=peft_config,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    unexpected_trainable = [
        name
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and ".default." not in name
    ]
    if unexpected_trainable:
        raise ValueError(f"Unexpected trainable parameters: {unexpected_trainable[:10]}")

    if state.is_main_process:
        run.summary.update(
            {
                "initialization_source": initialization_source,
                "parent_sft_adapter": str(adapter_path) if adapter_path is not None else None,
                "resume_from_checkpoint": config.dpo.resume_from_checkpoint,
                "lora_rank": rank,
                "preference_seed": config.preference.seed,
                **preference_summary,
                "train_pairs": len(train_dataset),
                "validation_pairs": len(eval_dataset),
                "train_label_distribution": _label_counts(train_base),
                "validation_label_distribution": _label_counts(val_base),
                "effective_batch_size": effective_batch,
            }
        )

    training_completed = False
    try:
        trainer.train(resume_from_checkpoint=config.dpo.resume_from_checkpoint)
        training_completed = True
    except KeyboardInterrupt:
        logger.warning("DPO interrupted; no final adapter will be exported")

    if training_completed and trainer.state.global_step > 0:
        selected_checkpoint = trainer.state.best_model_checkpoint
        selected_step = (
            int(Path(selected_checkpoint).name.removeprefix("checkpoint-"))
            if selected_checkpoint
            else trainer.state.global_step
        )
        policy = trainer.accelerator.unwrap_model(trainer.model)
        if state.is_main_process:
            policy.save_pretrained(adapter_dir, selected_adapters=["default"])
            metadata = {
                "initialization_source": initialization_source,
                "parent_sft_adapter": str(adapter_path) if adapter_path is not None else None,
                "resume_from_checkpoint": config.dpo.resume_from_checkpoint,
                "base_model": config.model.path,
                "selected_checkpoint": selected_checkpoint,
                "selected_step": selected_step,
                "lora_rank": rank,
                "preference_seed": config.preference.seed,
                **preference_summary,
            }
            (adapter_dir / "dpo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            run.summary.update(metadata | {"final_adapter": str(adapter_dir)})
            if config.wandb.log_model != "false":
                log_adapter_artifact(
                    run=run,
                    adapter_dir=adapter_dir,
                    run_name=run_name,
                    log_model=config.wandb.log_model,
                    best_metric=trainer.state.best_metric,
                    metric_for_best_model="eval_loss",
                )
            logger.info(
                "Run vLLM inference with:\n"
                f"  python scripts/vllm_inference.py model.params={config.model.params} "
                f"lora.path={adapter_dir} lora.max_rank={rank}"
            )

    if state.is_main_process:
        run.finish(exit_code=0 if training_completed else 130)


if __name__ == "__main__":
    main()
