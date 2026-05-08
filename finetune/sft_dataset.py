"""SFT dataset wrapper.

Wraps a base video dataset and yields conversational samples in the format
expected by a TRL ``collate_fn``: ``{"messages": [...]}`` where the user turn
already contains the video tensor and the assistant turn contains the gold
label string.

Reuses ``ConversationBuilder`` from the inference codebase so training and
inference see identical user prompts.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from falldet.inference.conversation import ConversationBuilder


class SFTConversationDataset(Dataset):
    column_names = None  # signals to TRL to infer columns from a sample

    def __init__(self, base: Dataset, conversation_builder: ConversationBuilder):
        self.base = base
        self.conv = conversation_builder

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        conv_data = self.conv.build(sample["video"])
        messages = list(conv_data.messages)
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"The best answer is: {sample['label_str']}"}],
            }
        )
        return {
            "messages": messages,
            "video_metadata": [v.metadata for v in conv_data.videos],
            "input_ids": [],
        }
