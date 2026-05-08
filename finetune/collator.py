"""Collate function for Qwen3-VL SFT.

Each example is ``{"messages": [...]}`` with a final assistant turn carrying
the gold label. The collator runs the processor on the full batch in one call
and masks the prompt portion of ``labels`` to ``-100`` so loss is computed
only on the assistant response.
"""

from __future__ import annotations


def _extract_videos(messages: list[dict]) -> list:
    out = []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "video":
                out.append(part["video"])
    return out


class Qwen3VLSFTCollator:
    def __init__(self, processor):
        self.processor = processor

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
            padding=False,
        )
        prompt_lens = [
            int((ids != self.processor.tokenizer.pad_token_id).sum())
            for ids in prompt_batch["input_ids"]
        ]

        labels = batch["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
