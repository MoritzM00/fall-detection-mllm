"""Standalone vLLM adapter-load smoke test.

Run from repo root in the *inference* conda env (not the training env):

    python -m finetune.test_vllm_lora

Loads Qwen3-VL-2B-Instruct with LoRA enabled, applies the adapter saved by
``train_smoke.py``, and generates a single response from a random video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from falldet.inference.conversation import ConversationBuilder
from falldet.schemas import PromptConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("vllm_smoke")

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_DIR = Path("finetune/outputs/smoke/adapter")
LABELS = ["fall", "walk", "sitting", "lying"]


def main() -> None:
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_DIR}. Run `python -m finetune.train_smoke` first."
        )

    log.info("Building random video tensor")
    video = torch.randint(0, 256, (8, 3, 224, 224), dtype=torch.uint8)

    log.info("Building conversation")
    label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    prompt_config = PromptConfig(labels=LABELS, output_format="text")
    conv_builder = ConversationBuilder(
        config=prompt_config,
        label2idx=label2idx,
        model_fps=8.0,
        needs_video_metadata=True,
    )

    log.info("Loading vLLM engine with LoRA enabled")
    llm = LLM(
        model=MODEL_ID,
        enable_lora=True,
        max_lora_rank=8,
        limit_mm_per_prompt={"video": 1},
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
    )

    processor = llm.get_tokenizer()  # for chat template if needed
    # Build inputs via the same path used by scripts/vllm_inference.py
    from transformers import AutoProcessor

    hf_processor = AutoProcessor.from_pretrained(MODEL_ID)
    prompt_input = conv_builder.build_vllm_inputs(video, hf_processor)

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    lora_request = LoRARequest("smoke", 1, str(ADAPTER_DIR))

    log.info("Generating with adapter")
    outputs = llm.generate(
        [prompt_input], sampling_params=sampling_params, lora_request=lora_request
    )

    text = outputs[0].outputs[0].text
    log.info("=== Generated text ===\n%s", text)


if __name__ == "__main__":
    main()
