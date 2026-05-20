"""SFT collator for chat-template VLMs in TRL's prompt-completion format.

Each example is ``{"prompt": [...], "completion": [...], "video_metadata": [...]}``.
The collator renders prompt and completion text via the chat template, runs the
processor on the prompt (with videos) and the completion (text-only)
separately, concatenates the token streams, and builds ``labels`` so loss is
computed only on the completion tokens.
"""

from __future__ import annotations

import torch


def _extract_videos(messages: list[dict]) -> list:
    out = []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "video":
                out.append(part["video"])
    return out


class PromptMaskedSFTCollator:
    def __init__(
        self,
        processor,
        needs_video_metadata: bool = True,
    ):
        self.processor = processor
        self.needs_video_metadata = needs_video_metadata

    def __call__(self, examples: list[dict]) -> dict:
        prompt_texts = [
            self.processor.apply_chat_template(
                e["prompt"], tokenize=False, add_generation_prompt=True
            )
            for e in examples
        ]
        eos = self.processor.tokenizer.eos_token
        completion_texts = [e["completion"][0]["content"][0]["text"] + eos for e in examples]
        videos = [_extract_videos(e["prompt"]) for e in examples]

        prompt_processor_kwargs: dict = dict(
            text=prompt_texts,
            videos=videos,
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )
        if self.needs_video_metadata:
            prompt_processor_kwargs["video_metadata"] = [e["video_metadata"] for e in examples]

        processed_prompts = self.processor(**prompt_processor_kwargs)
        processed_completions = self.processor(
            text=completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        prompt_ids = processed_prompts["input_ids"]
        prompt_mask = processed_prompts["attention_mask"]
        completion_ids = processed_completions["input_ids"]
        completion_mask = processed_completions["attention_mask"]

        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_only_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        mm_token_type_ids = processed_prompts.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = torch.cat(
                (mm_token_type_ids, torch.zeros_like(completion_ids)), dim=1
            )

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[completion_only_mask == 0] = -100

        batch = dict(processed_prompts)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        if mm_token_type_ids is not None:
            batch["mm_token_type_ids"] = mm_token_type_ids
            assert mm_token_type_ids.shape == input_ids.shape, (
                f"mm_token_type_ids {mm_token_type_ids.shape} vs input_ids {input_ids.shape}"
            )
        return batch
