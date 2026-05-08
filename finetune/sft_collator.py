"""SFT collator with prompt masking for chat-template VLMs.

Each example is ``{"messages": [...]}`` with a final assistant turn carrying
the gold label. The collator runs the processor on the full batch in one call
and masks the prompt portion of ``labels`` to ``-100`` so loss is computed
only on the assistant response.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


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
        max_length: int | None = None,
        debug_first_batch: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self._debug_first_batch = debug_first_batch
        self._validated = False

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
        for i, plen in enumerate(prompt_lens):
            if not torch.equal(full_ids[i, :plen], prompt_batch["input_ids"][i, :plen]):
                mismatch = (full_ids[i, :plen] != prompt_batch["input_ids"][i, :plen]).nonzero()
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
            labels[i, :plen] = -100
        labels[labels == pad_id] = -100
        batch["labels"] = labels

        if self.max_length is not None and full_ids.shape[1] > self.max_length:
            for key in ("input_ids", "attention_mask", "labels"):
                if key in batch:
                    batch[key] = batch[key][:, : self.max_length]
            full_ids = batch["input_ids"]
            labels = batch["labels"]

        if self._debug_first_batch and not self._validated:
            self._log_first_batch(examples, full_ids, labels, prompt_lens)
            self._validated = True

        return batch

    def _log_first_batch(
        self,
        examples: list[dict],
        full_ids: torch.Tensor,
        labels: torch.Tensor,
        prompt_lens: list[int],
    ) -> None:
        tok = self.processor.tokenizer
        b, t = full_ids.shape
        logger.info(
            f"[collator] first batch: shape={tuple(full_ids.shape)}, prompt_lens={prompt_lens}"
        )
        for i in range(min(b, 2)):
            plen = prompt_lens[i]
            n_supervised = int((labels[i] != -100).sum())
            n_pad = int((full_ids[i] == tok.pad_token_id).sum())
            response_ids = full_ids[i, plen:]
            response_text = tok.decode(response_ids, skip_special_tokens=False)
            last_supervised_token = tok.decode(full_ids[i, t - n_pad - 1 : t - n_pad])
            logger.info(
                f"[collator] ex{i}: total={t} prompt={plen} pad={n_pad} "
                f"supervised={n_supervised} last_real_token={last_supervised_token!r}"
            )
            logger.info(f"[collator] ex{i} response: {response_text!r}")
            gold = examples[i]["messages"][-1]["content"]
            logger.info(f"[collator] ex{i} gold message content: {gold!r}")
