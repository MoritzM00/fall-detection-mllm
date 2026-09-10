from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from falldet.schemas import from_dictconfig_dpo


@pytest.mark.parametrize("preset", ["smoke", "quick", "full"])
def test_dpo_presets_allow_pretrained_instruct_initialization(preset):
    config_dir = str(Path(__file__).parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="dpo_config", overrides=[f"dpo={preset}"])

    config = from_dictconfig_dpo(cfg)
    assert config.dpo.sft_adapter_path is None
    assert config.lora.r == 8
    assert config.model.path == "Qwen/Qwen3-VL-8B-Instruct"
