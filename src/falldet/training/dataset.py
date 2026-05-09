"""SFT dataset wrapper.

Wraps a base video dataset and yields conversational prompt-completion samples
in the format TRL's ``SFTTrainer`` recognises:
``{"prompt": [...user/system turns...], "completion": [assistant turn], ...}``.

Reuses ``ConversationBuilder`` from the inference codebase so training and
inference see identical user prompts.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from falldet.inference.conversation import ConversationBuilder


class SFTConversationDataset(Dataset):
    def __init__(self, base: Dataset, conversation_builder: ConversationBuilder):
        self.base = base
        self.conv = conversation_builder

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        conv_data = self.conv.build(sample["video"])
        completion = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"The best answer is: {sample['label_str']}"}],
            }
        ]
        return {
            "prompt": list(conv_data.messages),
            "completion": completion,
            "video_metadata": [v.metadata for v in conv_data.videos],
        }
