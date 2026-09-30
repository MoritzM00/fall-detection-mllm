import torch

from falldet.training.collator import VideoPreferenceCollator


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def token_id(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize
        parts = []
        for message in messages:
            for part in message["content"]:
                if part["type"] == "video":
                    parts.append("<video>")
                else:
                    parts.extend(part["text"].split())
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts)

    def __call__(self, text, videos=None, video_metadata=None, **kwargs):
        del video_metadata, kwargs
        rows = [[self.token_id(token) for token in value.split()] for value in text]
        max_length = max(map(len, rows))
        input_ids = torch.zeros((len(rows), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        output = {"input_ids": input_ids, "attention_mask": attention_mask}
        if videos is not None:
            video_values = [example_videos[0] for example_videos in videos]
            output.update(
                pixel_values_videos=torch.tensor(video_values).unsqueeze(1),
                video_grid_thw=torch.tensor([[1, 1, value] for value in video_values]),
                mm_token_type_ids=attention_mask * 7,
            )
        return output


def example(video, prompt, chosen, rejected):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "chosen": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": chosen}],
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rejected}],
            }
        ],
        "video_metadata": [{"fps": 7.5}],
    }


def test_video_preference_collator_orders_pairs_and_visuals():
    processor = FakeProcessor()
    collator = VideoPreferenceCollator(processor)
    examples = [
        example(11, "short", "answer fall", "answer walk long"),
        example(22, "a longer prompt", "answer lie_down long", "answer standing"),
    ]

    batch = collator(examples)

    assert batch["input_ids"].shape[0] == 4
    assert batch["pixel_values_videos"].squeeze(1).tolist() == [11, 22, 11, 22]
    assert batch["video_grid_thw"][:, 2].tolist() == [11, 22, 11, 22]

    completion_tokens = []
    id_to_token = {value: key for key, value in processor.vocab.items()}
    for row, mask in zip(batch["input_ids"], batch["completion_mask"], strict=True):
        completion_tokens.append([id_to_token[int(token)] for token in row[mask.bool()]])
    assert completion_tokens == [
        ["answer", "fall<eos>"],
        ["answer", "lie_down", "long<eos>"],
        ["answer", "walk", "long<eos>"],
        ["answer", "standing<eos>"],
    ]


def test_video_preference_collator_removes_internal_padding_and_aligns_masks():
    processor = FakeProcessor()
    batch = VideoPreferenceCollator(processor)(
        [
            example(3, "short", "one", "three tokens here"),
            example(4, "a much longer prompt", "two tokens", "one"),
        ]
    )

    for attention, completion, token_types in zip(
        batch["attention_mask"],
        batch["completion_mask"],
        batch["mm_token_type_ids"],
        strict=True,
    ):
        length = int(attention.sum())
        assert torch.all(attention[:length] == 1)
        assert torch.all(attention[length:] == 0)
        first_completion = int(completion.nonzero()[0])
        assert torch.all(completion[:first_completion] == 0)
        assert torch.all(completion[first_completion:length] == 1)
        assert torch.all(token_types[:first_completion] == 7)
        assert torch.all(token_types[first_completion:] == 0)
