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
        completion_texts = [e["completion"][0]["content"][0]["text"] + eos + "\n" for e in examples]
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


def _answer_text(example: dict, field: str) -> str:
    messages = example[field]
    if len(messages) != 1 or messages[0].get("role") != "assistant":
        raise ValueError(f"{field} must contain exactly one assistant message")
    content = messages[0].get("content", [])
    if len(content) != 1 or content[0].get("type") != "text":
        raise ValueError(f"{field} must contain exactly one text part")
    return content[0]["text"]


class VideoPreferenceCollator:
    """Create TRL's all-chosen/all-rejected combined batch for video DPO."""

    def __init__(self, processor, needs_video_metadata: bool = True):
        self.processor = processor
        self.needs_video_metadata = needs_video_metadata

    def __call__(self, examples: list[dict]) -> dict:
        if not examples:
            raise ValueError("Cannot collate an empty preference batch")

        prompt_texts = [
            self.processor.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            )
            for example in examples
        ]
        eos = self.processor.tokenizer.eos_token
        if not eos:
            raise ValueError("The processor tokenizer must define an EOS token")
        completion_texts = [
            *[_answer_text(example, "chosen") + eos + "\n" for example in examples],
            *[_answer_text(example, "rejected") + eos + "\n" for example in examples],
        ]
        if any(
            _answer_text(example, "chosen") == _answer_text(example, "rejected")
            for example in examples
        ):
            raise ValueError("Chosen and rejected answers must differ")

        prompt_kwargs = {
            "text": prompt_texts,
            "videos": [_extract_videos(example["prompt"]) for example in examples],
            "do_sample_frames": False,
            "do_resize": False,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.needs_video_metadata:
            prompt_kwargs["video_metadata"] = [example["video_metadata"] for example in examples]
        processed_prompts = self.processor(**prompt_kwargs)
        processed_completions = self.processor(
            text=completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        prompt_ids = processed_prompts["input_ids"]
        prompt_masks = processed_prompts["attention_mask"]
        completion_ids = processed_completions["input_ids"]
        completion_masks = processed_completions["attention_mask"]
        batch_size = len(examples)
        rows = []
        completion_only_masks = []
        sequence_token_types: dict[str, list[torch.Tensor]] = {}
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in processed_prompts:
                sequence_token_types[key] = []

        for row_index in range(2 * batch_size):
            prompt_index = row_index % batch_size
            prompt_keep = prompt_masks[prompt_index].bool()
            completion_keep = completion_masks[row_index].bool()
            prompt_row = prompt_ids[prompt_index][prompt_keep]
            completion_row = completion_ids[row_index][completion_keep]
            rows.append(torch.cat((prompt_row, completion_row)))
            completion_only_masks.append(
                torch.cat((torch.zeros_like(prompt_row), torch.ones_like(completion_row)))
            )
            for key, values in sequence_token_types.items():
                prompt_types = processed_prompts[key][prompt_index][prompt_keep]
                values.append(torch.cat((prompt_types, torch.zeros_like(completion_row))))

        max_length = max(row.numel() for row in rows)
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = torch.full((2 * batch_size, max_length), pad_id, dtype=prompt_ids.dtype)
        attention_mask = torch.zeros_like(input_ids)
        completion_mask = torch.zeros_like(input_ids)
        batch = {}
        for row_index, (row, row_completion_mask) in enumerate(
            zip(rows, completion_only_masks, strict=True)
        ):
            length = row.numel()
            input_ids[row_index, :length] = row
            attention_mask[row_index, :length] = 1
            completion_mask[row_index, :length] = row_completion_mask

        excluded = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
        for key, value in processed_prompts.items():
            if key in excluded:
                continue
            if isinstance(value, torch.Tensor):
                batch[key] = torch.cat((value, value), dim=0)
            elif isinstance(value, list):
                batch[key] = [*value, *value]
            elif isinstance(value, tuple):
                batch[key] = (*value, *value)
            else:
                batch[key] = value

        batch.update(
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
        )
        for key, values in sequence_token_types.items():
            padded = torch.zeros_like(input_ids)
            for row_index, value in enumerate(values):
                padded[row_index, : value.numel()] = value
            batch[key] = padded
        return batch
