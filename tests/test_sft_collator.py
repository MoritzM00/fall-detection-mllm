import torch

from falldet.training.collator import PromptMaskedSFTCollator


class FakeTokenizer:
    pad_token_id = 0


class FakeProcessor:
    """Tokenizer-only stand-in for the Qwen3-VL processor.

    Renders the chat template by joining text parts and labelling each turn,
    then tokenizes by mapping each whitespace-separated token to a stable id.
    Video parts produce a single ``<vid>`` token so the prompt stream still
    "contains" the video positionally.
    """

    tokenizer = FakeTokenizer()

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def _tok_id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
        return self.vocab[tok]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            role = m["role"]
            for part in m["content"]:
                if part.get("type") == "video":
                    parts.append(f"{role}:<vid>")
                elif part.get("type") == "text":
                    parts.append(f"{role}:{part['text'].replace(' ', '_')}")
        if add_generation_prompt:
            parts.append("assistant:<gen>")
        return " ".join(parts)

    def __call__(self, text, videos=None, video_metadata=None, **kwargs):
        del video_metadata, kwargs
        rows = [[self._tok_id(t) for t in s.split()] for s in text]
        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention_mask[i, : len(row)] = 1
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if videos is not None:
            out["pixel_values_videos"] = torch.zeros(len(videos))
        return out


def _example(label: str, video_token: str = "vidA") -> dict:
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_token},
                    {"type": "text", "text": "classify this"},
                ],
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"The best answer is: {label}"}],
            }
        ],
        "video_metadata": [{"total_num_frames": 8, "fps": 8.0, "frames_indices": list(range(8))}],
    }


def test_collator_masks_prompt_and_pad_keeps_completion():
    processor = FakeProcessor()
    collator = PromptMaskedSFTCollator(processor)

    examples = [_example("fall"), _example("standing_up")]
    batch = collator(examples)

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert "pixel_values_videos" in batch  # videos forwarded through processor

    # Each row's surviving (non -100) labels equal its completion length
    for i, ex in enumerate(examples):
        comp_text = processor.apply_chat_template(ex["completion"], tokenize=False)
        comp_len = len(comp_text.split())
        assert int((batch["labels"][i] != -100).sum()) == comp_len

    # Prompt tokens (first prompt-len positions) are all -100
    prompt_lens = [
        len(
            processor.apply_chat_template(
                ex["prompt"], tokenize=False, add_generation_prompt=True
            ).split()
        )
        for ex in examples
    ]
    for i, plen in enumerate(prompt_lens):
        assert torch.all(batch["labels"][i, :plen] == -100)

    # Pad positions (where attention_mask == 0) are -100
    pad_positions = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][pad_positions] == -100)


def test_collator_truncates_to_max_length():
    processor = FakeProcessor()
    collator = PromptMaskedSFTCollator(processor, max_length=4)

    batch = collator([_example("fall")])

    assert batch["input_ids"].shape[1] == 4
    assert batch["labels"].shape[1] == 4
    assert batch["attention_mask"].shape[1] == 4
