"""SFT collator with prompt masking for chat-template VLMs.

Each example is ``{"messages": [...]}`` with a final assistant turn carrying
the gold label. The collator runs the processor on the full batch in one call
and masks the prompt portion of ``labels`` to ``-100`` so loss is computed
only on the assistant response.
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
        self._first_batch = True

        padding_side = getattr(processor.tokenizer, "padding_side", None)
        if padding_side != "right":
            raise ValueError(
                f"PromptMaskedSFTCollator requires tokenizer.padding_side='right', got {padding_side!r}. "
                "Left padding would mask leading pad tokens instead of the prompt."
            )

    def __call__(self, examples: list[dict]) -> dict:
        full_texts = [
            self.processor.apply_chat_template(e["messages"], tokenize=False) for e in examples
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                e["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            for e in examples
        ]
        videos = [_extract_videos(e["messages"]) for e in examples]
        video_metadata = [e["video_metadata"] for e in examples]

        batch = self.processor(
            text=full_texts,
            videos=videos,
            video_metadata=video_metadata,
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )

        prompt_batch = self.processor(
            text=prompt_texts,
            videos=videos,
            video_metadata=video_metadata,
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )
        pad_id = self.processor.tokenizer.pad_token_id
        prompt_lens = [int((ids != pad_id).sum()) for ids in prompt_batch["input_ids"]]

        full_ids = batch["input_ids"]
        labels = full_ids.clone()
        if self._first_batch:
            self._validate_alignment(full_ids, prompt_batch["input_ids"], prompt_lens)
            self._first_batch = False
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        labels[labels == pad_id] = -100
        batch["labels"] = labels

        if self.max_length is not None and full_ids.shape[1] > self.max_length:
            for key in ("input_ids", "attention_mask", "labels"):
                batch[key] = batch[key][:, : self.max_length]

        return batch

    @staticmethod
    def _validate_alignment(
        full_ids: torch.Tensor, prompt_ids: torch.Tensor, prompt_lens: list[int]
    ) -> None:
        for i, plen in enumerate(prompt_lens):
            if not torch.equal(full_ids[i, :plen], prompt_ids[i, :plen]):
                mismatch = (full_ids[i, :plen] != prompt_ids[i, :plen]).nonzero()
                raise AssertionError(
                    f"Prefix mismatch in example {i}: full[:{plen}] != prompt[:{plen}]. "
                    f"First mismatches at positions {mismatch[:5].flatten().tolist()}. "
                    "Tokenizing prompt+response is not a pure prefix-extension of the prompt; "
                    "loss masking would be misaligned."
                )
            if plen >= full_ids.shape[1]:
                raise AssertionError(
                    f"Example {i}: prompt length {plen} >= full length {full_ids.shape[1]}; "
                    "no response tokens left to train on."
                )
