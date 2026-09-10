import copy
import math

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import DPOConfig

from falldet.training.dpo_trainer import VideoDPOTrainer


class FixedPreferenceCollator:
    def __call__(self, rows):
        assert len(rows) == 1
        return {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "completion_mask": torch.tensor([[0, 0, 1], [0, 0, 1]]),
        }


def _tokenizer():
    backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "prompt": 1,
                "assistant": 2,
                "chosen": 3,
                "rejected": 4,
                "<eos>": 5,
                "<unk>": 6,
            },
            unk_token="<unk>",
        )
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _trainer(tmp_path):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    reference = copy.deepcopy(model)
    dataset = Dataset.from_dict(
        {"prompt": ["prompt"], "chosen": ["chosen"], "rejected": ["rejected"]}
    )
    trainer = VideoDPOTrainer(
        model=model,
        ref_model=reference,
        args=DPOConfig(
            output_dir=str(tmp_path),
            report_to="none",
            remove_unused_columns=False,
            max_length=None,
            precompute_ref_log_probs=False,
            use_liger_kernel=False,
            padding_free=False,
            beta=0.1,
            loss_type="sigmoid",
        ),
        train_dataset=dataset,
        data_collator=FixedPreferenceCollator(),
        processing_class=_tokenizer(),
    )
    return trainer


def test_matching_policy_and_reference_start_at_log_two(tmp_path):
    trainer = _trainer(tmp_path)
    batch = FixedPreferenceCollator()([{}])
    batch = {key: value.to(trainer.accelerator.device) for key, value in batch.items()}

    loss = trainer._compute_loss(trainer.model, batch, return_outputs=False)

    assert loss.detach().item() == pytest.approx(math.log(2), abs=1e-5)


def test_optimizer_changes_policy_but_not_reference(tmp_path):
    trainer = _trainer(tmp_path)
    batch = FixedPreferenceCollator()([{}])
    batch = {key: value.to(trainer.accelerator.device) for key, value in batch.items()}
    policy_before = {
        name: value.detach().clone() for name, value in trainer.model.named_parameters()
    }
    reference_before = {
        name: value.detach().clone() for name, value in trainer.ref_model.named_parameters()
    }

    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    optimizer.zero_grad()
    trainer._compute_loss(trainer.model, batch, return_outputs=False).backward()
    optimizer.step()

    assert any(
        not torch.equal(value, policy_before[name])
        for name, value in trainer.model.named_parameters()
    )
    assert all(
        torch.equal(value, reference_before[name])
        for name, value in trainer.ref_model.named_parameters()
    )
