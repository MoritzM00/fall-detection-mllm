import math

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel
from trl import DPOConfig

from falldet.training.dpo_trainer import VideoDPOTrainer
from tests.test_dpo_trainer import FixedPreferenceCollator, _tokenizer


def _peft_trainer(tmp_path):
    torch.manual_seed(0)
    base = GPT2LMHeadModel(
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
    model = get_peft_model(base, LoraConfig(r=2, target_modules=["c_attn"], bias="none"))
    for name, parameter in model.named_parameters():
        if ".default." in name:
            torch.nn.init.normal_(parameter, std=0.02)
    dataset = Dataset.from_dict(
        {"prompt": ["prompt"], "chosen": ["chosen"], "rejected": ["rejected"]}
    )
    return VideoDPOTrainer(
        model=model,
        ref_model=None,
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


def test_pretrained_lora_is_copied_to_frozen_reference_adapter(tmp_path):
    trainer = _peft_trainer(tmp_path)
    assert trainer.ref_model is None
    assert set(trainer.model.peft_config) == {"default", "ref"}

    reference_before = {}
    for name, parameter in trainer.model.named_parameters():
        if ".default." not in name:
            continue
        reference_name = name.replace(".default.", ".ref.")
        reference = trainer.model.get_parameter(reference_name)
        assert torch.equal(parameter, reference)
        assert not reference.requires_grad
        reference_before[reference_name] = reference.detach().clone()

    batch = FixedPreferenceCollator()([{}])
    batch = {key: value.to(trainer.accelerator.device) for key, value in batch.items()}
    loss = trainer._compute_loss(trainer.model, batch, return_outputs=False)
    assert loss.detach().item() == pytest.approx(math.log(2), abs=1e-5)

    policy_before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if ".default." in name
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in trainer.model.parameters() if parameter.requires_grad], lr=0.1
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(trainer.model.get_parameter(name), value)
        for name, value in policy_before.items()
    )
    assert all(
        torch.equal(trainer.model.get_parameter(name), value)
        for name, value in reference_before.items()
    )


def test_policy_only_export_excludes_reference_and_reloads(tmp_path):
    trainer = _peft_trainer(tmp_path / "trainer")
    export_dir = tmp_path / "adapter"

    trainer.model.save_pretrained(export_dir, selected_adapters=["default"])

    assert (export_dir / "adapter_config.json").is_file()
    assert (export_dir / "adapter_model.safetensors").is_file()
    assert not (export_dir / "ref").exists()
    fresh_base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
        )
    )
    reloaded = PeftModel.from_pretrained(fresh_base, export_dir)
    assert reloaded.active_adapter == "default"
    assert set(reloaded.peft_config) == {"default"}
