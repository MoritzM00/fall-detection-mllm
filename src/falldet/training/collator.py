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
    def __init__(self, processor, max_length: int | None = None):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict:
        prompt_texts = [
            self.processor.apply_chat_template(
                e["prompt"], tokenize=False, add_generation_prompt=True
            )
            for e in examples
        ]
        completion_texts = [
            self.processor.apply_chat_template(e["completion"], tokenize=False) for e in examples
        ]
        videos = [_extract_videos(e["prompt"]) for e in examples]
        video_metadata = [e["video_metadata"] for e in examples]

        processed_prompts = self.processor(
            text=prompt_texts,
            videos=videos,
            video_metadata=video_metadata,
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )
        processed_completions = self.processor(
            text=completion_texts,
            return_tensors="pt",
            padding=True,
        )

        prompt_ids = processed_prompts["input_ids"]
        prompt_mask = processed_prompts["attention_mask"]
        completion_ids = processed_completions["input_ids"]
        completion_mask = processed_completions["attention_mask"]

        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_only_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        if self.max_length is not None and input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_only_mask = completion_only_mask[:, : self.max_length]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[completion_only_mask == 0] = -100

        batch = dict(processed_prompts)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        return batch
