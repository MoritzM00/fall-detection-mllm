"""Tests for Pydantic configuration schemas."""

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from falldet.schemas import (
    DataConfig,
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    SamplingConfig,
    VideoDatasetItemConfig,
    VLLMConfig,
    WandbConfig,
    from_dictconfig,
)

# --- VLLMConfig ---


class TestVLLMConfig:
    def test_defaults(self):
        cfg = VLLMConfig()
        assert cfg.use_mock is False
        assert cfg.tensor_parallel_size is None
        assert cfg.gpu_memory_utilization == 0.9

    def test_custom_values(self):
        cfg = VLLMConfig(use_mock=True, tensor_parallel_size=4, gpu_memory_utilization=0.85)
        assert cfg.use_mock is True
        assert cfg.tensor_parallel_size == 4
        assert cfg.gpu_memory_utilization == 0.85

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            VLLMConfig(nonexistent_field="bad")

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            VLLMConfig(use_mock="not_a_bool_or_truthy")


# --- ModelConfig ---


class TestModelConfig:
    def test_basic_construction(self):
        cfg = ModelConfig(org="Qwen", family="Qwen", version="3", params="4B")
        assert cfg.org == "Qwen"
        assert cfg.version == "3"
        assert cfg.variant is None
        assert cfg.active_params is None

    def test_version_int_coercion(self):
        cfg = ModelConfig(org="Qwen", family="Qwen", version=3, params="4B")
        assert cfg.version == "3"
        assert isinstance(cfg.version, str)

    def test_full_construction(self):
        cfg = ModelConfig(
            org="Qwen",
            family="Qwen",
            version="3",
            variant="Instruct",
            params="30B",
            active_params="A3B",
            needs_video_metadata=True,
            mm_processor_kwargs={"do_resize": False},
        )
        assert cfg.variant == "Instruct"
        assert cfg.active_params == "A3B"
        assert cfg.mm_processor_kwargs == {"do_resize": False}

    def test_name_property(self):
        cfg = ModelConfig(org="Qwen", family="Qwen", version="3", params="4B", variant="Instruct")
        assert cfg.name == "Qwen3-VL-4B-Instruct"

    def test_path_property(self):
        cfg = ModelConfig(org="Qwen", family="Qwen", version="3", params="4B", variant="Instruct")
        assert cfg.path == "Qwen/Qwen3-VL-4B-Instruct"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(org="X", family="X", version="1", params="1B", unknown="bad")


# --- SamplingConfig ---


class TestSamplingConfig:
    def test_defaults(self):
        cfg = SamplingConfig()
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512
        assert cfg.top_p == 1.0
        assert cfg.seed is None
        assert cfg.stop_token_ids is None

    def test_custom_values(self):
        cfg = SamplingConfig(temperature=0.7, max_tokens=1024, seed=42)
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1024
        assert cfg.seed == 42

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            SamplingConfig(bad_field=1)


# --- DataConfig ---


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.seed == 0
        assert cfg.split == "cs"
        assert cfg.mode == "test"
        assert cfg.size == 448
        assert cfg.max_size is None

    def test_none_optional_fields(self):
        cfg = DataConfig(size=None, max_size=None)
        assert cfg.size is None
        assert cfg.max_size is None


# --- WandbConfig ---


class TestWandbConfig:
    def test_defaults(self):
        cfg = WandbConfig()
        assert cfg.mode == "online"
        assert cfg.name is None
        assert cfg.tags is None

    def test_none_tags(self):
        cfg = WandbConfig(tags=None)
        assert cfg.tags is None

    def test_with_tags(self):
        cfg = WandbConfig(tags=["zeroshot", "qwen"])
        assert cfg.tags == ["zeroshot", "qwen"]

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            WandbConfig(mode="invalid_mode")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            WandbConfig(unknown=True)


# --- VideoDatasetItemConfig ---


class TestVideoDatasetItemConfig:
    def test_minimal(self):
        cfg = VideoDatasetItemConfig(
            name="cmdfall",
            video_root="/data/video",
            annotations_file="labels.csv",
            split_root="splits/",
        )
        assert cfg.name == "cmdfall"
        assert cfg.split is None
        assert cfg.evaluation_group is None
        assert cfg.dataset_fps is None

    def test_full(self):
        cfg = VideoDatasetItemConfig(
            name="cmdfall",
            video_root="/data/video",
            annotations_file="labels.csv",
            dataset_fps=20.0,
            split_root="splits/",
            split="cs",
            evaluation_group="omnifall",
        )
        assert cfg.dataset_fps == 20.0
        assert cfg.split == "cs"
        assert cfg.evaluation_group == "omnifall"


# --- DatasetConfig ---


class TestDatasetConfig:
    def test_basic(self):
        item = VideoDatasetItemConfig(
            name="test", video_root="/v", annotations_file="a.csv", split_root="s/"
        )
        cfg = DatasetConfig(
            name="test-ds",
            video_datasets=[item],
            target_fps=7.5,
            vid_frame_count=16,
        )
        assert cfg.name == "test-ds"
        assert len(cfg.video_datasets) == 1
        assert cfg.create_all_combined is False

    def test_extra_fields_rejected(self):
        item = VideoDatasetItemConfig(
            name="test", video_root="/v", annotations_file="a.csv", split_root="s/"
        )
        with pytest.raises(ValidationError):
            DatasetConfig(
                name="test",
                video_datasets=[item],
                target_fps=7.5,
                vid_frame_count=16,
                bad_field="x",
            )


# --- InferenceConfig ---


def _make_minimal_inference_config(**overrides) -> InferenceConfig:
    """Helper to build a minimal valid InferenceConfig."""
    ds_item = VideoDatasetItemConfig(
        name="test", video_root="/v", annotations_file="a.csv", split_root="s/"
    )
    defaults = dict(
        vllm=VLLMConfig(),
        model=ModelConfig(org="Qwen", family="Qwen", version="3", params="4B"),
        sampling=SamplingConfig(),
        data=DataConfig(),
        prompt={},
        dataset=DatasetConfig(
            name="test", video_datasets=[ds_item], target_fps=7.5, vid_frame_count=16
        ),
        wandb=WandbConfig(),
    )
    defaults.update(overrides)
    return InferenceConfig(**defaults)


class TestInferenceConfig:
    def test_minimal_construction(self):
        cfg = _make_minimal_inference_config()
        assert cfg.batch_size == 32
        assert cfg.num_samples is None
        assert cfg.dataset_train is None
        assert cfg.dataset_val is None
        assert cfg.dataset_test is None

    def test_custom_root_fields(self):
        cfg = _make_minimal_inference_config(batch_size=16, num_workers=4, num_samples=100)
        assert cfg.batch_size == 16
        assert cfg.num_workers == 4
        assert cfg.num_samples == 100

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            _make_minimal_inference_config(totally_fake_field="bad")

    def test_model_dump_produces_dict(self):
        cfg = _make_minimal_inference_config()
        dumped = cfg.model_dump()
        assert isinstance(dumped, dict)
        assert isinstance(dumped["vllm"], dict)
        assert isinstance(dumped["model"], dict)
        assert isinstance(dumped["dataset"]["video_datasets"], list)


# --- from_dictconfig ---


class TestFromDictConfig:
    def test_round_trip(self):
        raw = {
            "vllm": {
                "use_mock": True,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "mm_encoder_tp_mode": "data",
                "mm_processor_cache_gb": 0,
                "seed": 0,
                "dtype": "bfloat16",
                "enforce_eager": False,
                "max_model_len": -1,
                "max_num_batched_tokens": None,
                "trust_remote_code": True,
                "async_scheduling": True,
                "skip_mm_profiling": False,
                "enable_prefix_caching": False,
                "mm_processor_kwargs": {"do_sample_frames": False},
                "limit_mm_per_prompt": {"image": 0, "video": 1},
                "enable_expert_parallel": None,
            },
            "model": {
                "org": "Qwen",
                "family": "Qwen",
                "version": 3,  # int — should be coerced to str
                "variant": "Instruct",
                "params": "4B",
                "active_params": None,
                "needs_video_metadata": True,
                "mm_processor_kwargs": {"do_resize": False},
            },
            "sampling": {
                "temperature": 0.0,
                "max_tokens": 1024,
                "top_k": 20,
                "top_p": 0.8,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "seed": 0,
                "stop_token_ids": None,
            },
            "data": {"seed": 0, "split": "cs", "mode": "test", "size": 448, "max_size": None},
            "prompt": {
                "output_format": "text",
                "cot": False,
                "cot_start_tag": "<think>",
                "cot_end_tag": "</think>",
                "model_family": "Qwen",
                "num_shots": 0,
                "shot_selection": "balanced",
                "exemplar_seed": 42,
                "role_variant": "standard",
                "task_variant": "standard",
                "labels_variant": "bulleted",
                "definitions_variant": None,
            },
            "dataset": {
                "name": "test-dataset",
                "video_datasets": [
                    {
                        "name": "cmdfall",
                        "video_root": "/data/video",
                        "annotations_file": "labels.csv",
                        "dataset_fps": 20.0,
                        "split_root": "splits/",
                    }
                ],
                "target_fps": 7.5,
                "vid_frame_count": 16,
                "path_format": "{video_root}/{video_path}{ext}",
            },
            "wandb": {
                "mode": "disabled",
                "project": "test-project",
                "name": None,
                "tags": None,
            },
            "model_fps": 7.5,
            "num_frames": 16,
            "batch_size": 10,
            "num_workers": 0,
            "prefetch_factor": 2,
            "output_dir": "outputs",
            "save_predictions": True,
            "save_metrics": True,
            "log_videos": 0,
            "num_samples": 10,
        }
        dictconfig = OmegaConf.create(raw)
        config = from_dictconfig(dictconfig)

        assert isinstance(config, InferenceConfig)
        assert config.model.version == "3"  # coerced from int
        assert config.vllm.use_mock is True
        assert config.batch_size == 10
        assert config.wandb.mode == "disabled"

    def test_hydra_key_stripped(self):
        """Ensure the 'hydra' key injected by Hydra is removed."""
        raw = {
            "vllm": {},
            "model": {"org": "Q", "family": "Q", "version": "1", "params": "1B"},
            "sampling": {},
            "data": {},
            "prompt": {},
            "dataset": {
                "name": "t",
                "video_datasets": [
                    {
                        "name": "x",
                        "video_root": "/v",
                        "annotations_file": "a",
                        "split_root": "s",
                    }
                ],
                "target_fps": 1.0,
                "vid_frame_count": 1,
            },
            "wandb": {},
            "hydra": {"run": {"dir": "outputs"}},
        }
        dictconfig = OmegaConf.create(raw)
        config = from_dictconfig(dictconfig)
        assert isinstance(config, InferenceConfig)
