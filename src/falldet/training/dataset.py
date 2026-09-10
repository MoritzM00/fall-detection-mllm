"""SFT dataset wrapper.

Wraps a base video dataset and yields conversational prompt-completion samples
in the format TRL's ``SFTTrainer`` recognises:
``{"prompt": [...user/system turns...], "completion": [assistant turn], ...}``.

Reuses ``ConversationBuilder`` from the inference codebase so training and
inference see identical user prompts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from falldet.inference.conversation import ConversationBuilder
from falldet.training.preferences import NegativeSelector, canonicalize_label, label_for_index


def _assistant_answer(label: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"The best answer is: {label}"}],
        }
    ]


class SFTConversationDataset(Dataset):
    def __init__(self, base: Dataset, conversation_builder: ConversationBuilder):
        self.base = base
        self.conv = conversation_builder

    def __len__(self) -> int:
        return len(self.base)  # ty: ignore[invalid-argument-type]

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        conv_data = self.conv.build(sample["video"])
        completion = _assistant_answer(canonicalize_label(sample["label_str"]))
        return {
            "prompt": list(conv_data.messages),
            "completion": completion,
            "video_metadata": [v.metadata for v in conv_data.videos],
        }


class DPOConversationDataset(Dataset):
    """Build one chosen/rejected label pair for each base video row."""

    def __init__(
        self,
        base: Dataset,
        conversation_builder: ConversationBuilder,
        negative_selector: NegativeSelector,
    ):
        self.base = base
        self.conv = conversation_builder
        self.negative_selector = negative_selector

    def __len__(self) -> int:
        return len(self.base)  # ty: ignore[invalid-argument-type]

    def __getitem__(self, index: int) -> dict:
        expected_positive = label_for_index(self.base, index)
        sample = self.base[index]
        positive = canonicalize_label(sample["label_str"])
        if positive != expected_positive:
            raise RuntimeError(
                "Video dataset returned a replacement row after a decode failure; "
                "DPO requires stable row identity"
            )
        negative = self.negative_selector.select(index, positive)
        if negative == positive:
            raise ValueError("Chosen and rejected labels must differ")

        conv_data = self.conv.build(sample["video"])
        return {
            "prompt": list(conv_data.messages),
            "chosen": _assistant_answer(positive),
            "rejected": _assistant_answer(negative),
            "video_metadata": [video.metadata for video in conv_data.videos],
        }


class _LazyRowsTransform:
    """Resolve lightweight Hugging Face rows through a lazy PyTorch dataset."""

    def __init__(self, source: Dataset):
        self.source = source

    def __call__(self, rows: dict[str, Any]) -> dict[str, list]:
        raw_indices = rows["row_index"]
        if isinstance(raw_indices, Sequence):
            indices = [int(index) for index in raw_indices]
        else:
            indices = [int(raw_indices)]

        examples = [self.source[index] for index in indices]
        if not examples:
            return {}

        keys = examples[0].keys()
        if any(example.keys() != keys for example in examples[1:]):
            raise ValueError("Lazy dataset rows must all expose the same fields")
        return {key: [example[key] for example in examples] for key in keys}


def as_lazy_hf_dataset(source: Dataset) -> HFDataset:
    """Adapt a map-style PyTorch dataset to TRL without eager materialization.

    Only integer row indices are stored in Arrow. The source dataset is accessed
    by a lazy transform when an individual row or batch is requested, so video
    tensors are never serialized into the Hugging Face dataset.
    """

    try:
        length = len(source)  # ty: ignore[invalid-argument-type]
    except TypeError as exc:
        raise TypeError("The lazy TRL bridge requires a map-style dataset with __len__") from exc
    if length < 1:
        raise ValueError("The lazy TRL bridge requires at least one row")

    rows = HFDataset.from_dict({"row_index": range(length)})
    return rows.with_transform(_LazyRowsTransform(source))
