"""Hydra-driven SFT training: TRL SFTTrainer + LoRA on Qwen3-VL.

Composes the same config groups as inference (dataset/model/prompt) plus a
training/ group for SFTConfig knobs and a lora/train.yaml for peft.LoraConfig.

Run from the repo root in the ``falldet-finetune`` conda env:

    python scripts/train_sft.py                     # uses training=quick
    python scripts/train_sft.py training=full       # full run preset
    python scripts/train_sft.py training=smoke      # short wiring check

Multi-GPU (single node) via accelerate or torchrun:

    accelerate launch --config_file config/accelerate/ddp_bf16.yaml \\
        --num_processes 4 scripts/train_sft.py training=quick
    torchrun --nproc_per_node=4 scripts/train_sft.py training=quick

Outputs land under ``${output_dir}/<run_name>/``; final adapter at
``${output_dir}/<run_name>/adapter``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from omegaconf import DictConfig
from peft import LoraConfig as PeftLoraConfig
from torch.utils.data import Subset
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from falldet.data.video_dataset import label2idx as omnifall_label2idx
from falldet.data.video_dataset_factory import get_video_datasets
from falldet.inference.conversation import ConversationBuilder
from falldet.schemas import TrainingConfig, from_dictconfig_training
from falldet.training.collator import PromptMaskedSFTCollator
from falldet.training.dataset import SFTConversationDataset
from falldet.utils.logging import disable_logging_for_non_main_process, setup_logging
from falldet.utils.wandb import initialize_run_from_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="training_config", version_base=None)
def main(cfg: DictConfig) -> None:
    state = PartialState()
    setup_logging(
        log_file="logs/training.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    disable_logging_for_non_main_process(state.local_process_index)
    logger.info(f"Distributed state: {state}")
    config: TrainingConfig = from_dictconfig_training(cfg)
    logger.info(config.model_dump_json(indent=2))

    run = initialize_run_from_config(config)
    run_name = run.name

    output_dir = Path(config.output_dir) / run_name
    adapter_dir = output_dir / "adapter"
    if state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    state.wait_for_everyone()

    model_path = config.model.path
    logger.info(f"Loading processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)

    logger.info("Loading model")
    model_kwargs = {"dtype": torch.bfloat16}
    if config.training.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.training.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False

    labels = list(omnifall_label2idx.keys())
    prompt_config = config.prompt.model_copy(update={"labels": labels, "output_format": "text"})
    conv_builder = ConversationBuilder(
        config=prompt_config,
        label2idx=omnifall_label2idx,
        model_fps=config.model_fps,
        needs_video_metadata=config.model.needs_video_metadata,
    )

    with state.main_process_first():
        base = get_video_datasets(
            config=config,
            mode=config.data.mode,
            run=run,
            return_individual=False,
            split=config.data.split,
            size=config.data.size,
            max_size=config.data.max_size,
            seed=config.data.seed,
        )
    logger.info(f"Base train dataset: {len(base)} samples")
    train_ds = SFTConversationDataset(base, conv_builder)
    collator = PromptMaskedSFTCollator(processor, max_length=config.training.max_length)

    eval_ds = None
    if config.training.eval_strategy != "no":
        with state.main_process_first():
            val_base = get_video_datasets(
                config=config,
                mode="val",
                run=run,
                return_individual=False,
                split=config.data.split,
                size=config.data.size,
                max_size=config.data.max_size,
                seed=0,  # val is deterministic regardless of data.seed
            )
        logger.info(f"Base val dataset: {len(val_base)} samples")
        if config.training.max_eval_samples is not None:
            n = min(config.training.max_eval_samples, len(val_base))
            val_base = Subset(val_base, list(range(n)))
            logger.info(f"Capped val dataset to {n} samples")
        eval_ds = SFTConversationDataset(val_base, conv_builder)

    peft_lora = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=list(config.lora.target_modules),
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        eval_on_start=config.training.eval_on_start,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_length=config.training.max_length,
        report_to=config.training.report_to,
        seed=config.training.seed,
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=config.pin_memory,
        dataloader_persistent_workers=config.persistent_workers and config.num_workers > 0,
        dataloader_prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        use_liger_kernel=config.training.use_liger_kernel,
        optim=config.training.optim,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_lora,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
    )

    world_size = state.num_processes
    effective_batch = (
        config.training.per_device_train_batch_size
        * config.training.gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    if config.training.max_steps > 0:
        total_steps = config.training.max_steps
    else:
        total_steps = int(steps_per_epoch * config.training.num_train_epochs)
    logger.info(
        f"Train samples={len(train_ds)} | effective_batch={effective_batch} "
        f"(per_device={config.training.per_device_train_batch_size} "
        f"x grad_accum={config.training.gradient_accumulation_steps} x world={world_size}) "
        f"| steps/epoch={steps_per_epoch} | total_steps={total_steps}"
    )

    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving adapter to {adapter_dir}")
    trainer.save_model(str(adapter_dir))

    logger.info(
        "Run vLLM inference with this adapter via:\n"
        f"  python scripts/vllm_inference.py "
        f"lora.path={adapter_dir} lora.max_rank={config.lora.r}"
    )
    if state.is_main_process:
        run.finish()


if __name__ == "__main__":
    main()
