import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from falldet.training.dataset import as_lazy_hf_dataset


class CountingDataset(Dataset):
    def __init__(self):
        self.accessed: list[int] = []

    def __len__(self):
        return 4

    def __getitem__(self, index):
        index = int(index)
        self.accessed.append(index)
        return {"value": index, "tensor": torch.tensor([index, index + 1])}


def _identity_collator(rows):
    return rows


def test_lazy_hf_bridge_resolves_only_requested_rows():
    source = CountingDataset()
    dataset = as_lazy_hf_dataset(source)

    assert source.accessed == []
    assert dataset.column_names == ["row_index"]

    row = dataset[2]
    assert row["value"] == 2
    assert torch.equal(row["tensor"], torch.tensor([2, 3]))
    assert source.accessed == [2]

    batch = dataset[[3, 1]]
    assert batch["value"] == [3, 1]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            batch["tensor"], [torch.tensor([3, 4]), torch.tensor([1, 2])], strict=True
        )
    )
    assert source.accessed == [2, 3, 1]


def test_lazy_hf_bridge_works_with_dataloader_workers():
    dataset = as_lazy_hf_dataset(CountingDataset())
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=_identity_collator)

    batches = list(loader)

    assert [[row["value"] for row in batch] for batch in batches] == [[0, 1], [2, 3]]


class EmptyDataset(CountingDataset):
    def __len__(self):
        return 0


def test_lazy_hf_bridge_rejects_empty_dataset():
    with pytest.raises(ValueError, match="at least one row"):
        as_lazy_hf_dataset(EmptyDataset())
