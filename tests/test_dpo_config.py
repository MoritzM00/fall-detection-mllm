from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from falldet.schemas import PreferenceConfig, from_dictconfig_dpo


@pytest.mark.parametrize("preset", ["smoke", "quick", "full"])
def test_dpo_presets_allow_pretrained_instruct_initialization(preset):
    config_dir = str(Path(__file__).parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="dpo_config", overrides=[f"dpo={preset}"])

    config = from_dictconfig_dpo(cfg)
    assert config.dpo.sft_adapter_path is None
    assert config.lora.r == 8
    assert config.model.path == "Qwen/Qwen3-VL-8B-Instruct"


def test_similarity_preference_config_requires_and_accepts_embedding_paths():
    config_dir = str(Path(__file__).parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="dpo_config",
            overrides=[
                "preference=similarity",
                "preference.train_embeddings_path=/tmp/train.pt",
                "preference.validation_embeddings_path=/tmp/val.pt",
            ],
        )

    config = from_dictconfig_dpo(cfg)
    assert config.preference.strategy == "similarity"
    assert config.preference.train_embeddings_path == "/tmp/train.pt"
    assert config.preference.validation_embeddings_path == "/tmp/val.pt"


def test_similarity_preference_rejects_missing_embedding_paths():
    with pytest.raises(ValueError, match="require train_embeddings_path"):
        PreferenceConfig(strategy="similarity")


def test_confusion_preference_config_accepts_prediction_paths():
    config_dir = str(Path(__file__).parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="dpo_config",
            overrides=[
                "preference=confusion",
                "preference.train_predictions_paths=[/tmp/a.jsonl,/tmp/b.jsonl]",
            ],
        )

    config = from_dictconfig_dpo(cfg)
    assert config.preference.strategy == "confusion"
    assert config.preference.train_predictions_paths == ["/tmp/a.jsonl", "/tmp/b.jsonl"]
    assert config.preference.validation_predictions_paths == []
    assert config.preference.uniform_mix == 0.1


def test_confusion_preference_rejects_missing_prediction_paths():
    with pytest.raises(ValueError, match="require train_predictions_paths"):
        PreferenceConfig(strategy="confusion")


def test_oops_preset_selects_on_classification_metrics_without_liger():
    config_dir = str(Path(__file__).parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="dpo_config", overrides=["dpo=oops"])

    config = from_dictconfig_dpo(cfg)
    assert config.dpo.classification_metrics is True
    assert config.dpo.use_liger_kernel is False
    assert config.dpo.metric_for_best_model == "eval_balanced_accuracy"
    assert config.dpo.greater_is_better is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["dpo.use_liger_kernel=true"], "need logits"),
        (["dpo.classification_metrics=false"], "requires dpo.classification_metrics"),
        (["dpo.greater_is_better=false"], "greater_is_better=true"),
    ],
)
def test_classification_selection_rejects_inconsistent_settings(overrides, message):
    config_dir = str(Path(__file__).parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="dpo_config", overrides=["dpo=oops", *overrides])

    with pytest.raises(ValueError, match=message):
        from_dictconfig_dpo(cfg)


def test_dpo_config_accepts_resume_checkpoint_path():
    config_dir = str(Path(__file__).parents[1] / "config")
    checkpoint = "/tmp/dpo/checkpoint-102"
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="dpo_config",
            overrides=[f"dpo.resume_from_checkpoint={checkpoint}"],
        )

    config = from_dictconfig_dpo(cfg)
    assert config.dpo.resume_from_checkpoint == checkpoint
