"""End-to-end smoke test: TRL SFTTrainer + LoRA on Qwen3-VL-2B with random data.

Run from repo root in the ``falldet-finetune`` conda env:

    python finetune/train_smoke.py

Verifies the full training loop wires together. Output adapter goes to
``finetune/outputs/smoke/adapter``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

from falldet.inference.conversation import ConversationBuilder
from falldet.schemas import PromptConfig
from finetune.collator import Qwen3VLSFTCollator
from finetune.dataset import Qwen3VLFallDataset
from finetune.dummy_dataset import DummyVideoDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("smoke")

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR = Path("finetune/outputs/smoke")
LABELS = ["fall", "walk", "sitting", "lying"]


def main() -> None:
    log.info("Loading processor: %s", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    log.info("Loading model")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    for p in model.model.visual.parameters():
        p.requires_grad = False

    label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    prompt_config = PromptConfig(labels=LABELS, output_format="text")
    conv_builder = ConversationBuilder(
        config=prompt_config,
        label2idx=label2idx,
        model_fps=8.0,
        needs_video_metadata=True,
    )

    base = DummyVideoDataset(n=8, T=8, H=224, W=224, labels=tuple(LABELS))
    train_ds = Qwen3VLFallDataset(base, conv_builder)
    collator = Qwen3VLSFTCollator(processor)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        max_length=None,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=lora_config,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=processor,
    )

    log.info("Starting training")
    trainer.train()

    adapter_dir = OUTPUT_DIR / "adapter"
    log.info("Saving adapter to %s", adapter_dir)
    trainer.save_model(str(adapter_dir))


if __name__ == "__main__":
    main()
