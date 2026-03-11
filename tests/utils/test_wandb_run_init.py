from types import SimpleNamespace

from falldet.schemas import InferenceConfig
from falldet.utils.wandb import create_name_and_tags_from_config, initialize_run_from_config


def _config() -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "vllm": {
                "use_mock": False,
                "gpu_memory_utilization": 0.9,
                "mm_encoder_tp_mode": "data",
                "mm_processor_cache_gb": 0,
                "seed": 0,
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "tensor_parallel_size": 1,
                "trust_remote_code": False,
                "async_scheduling": True,
                "skip_mm_profiling": False,
                "enable_prefix_caching": False,
                "limit_mm_per_prompt": {"image": 16, "video": 2},
                "enforce_eager": False,
                "max_num_batched_tokens": 16384,
                "mm_processor_kwargs": {},
                "enable_expert_parallel": None,
            },
            "model": {
                "org": "Qwen",
                "family": "Qwen",
                "version": "2.5",
                "variant": "Instruct",
                "params": "7B",
                "active_params": None,
                "needs_video_metadata": True,
                "mm_processor_kwargs": {},
            },
            "sampling": {
                "temperature": 0.0,
                "max_tokens": 64,
                "top_k": 0,
                "top_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "seed": None,
                "stop_token_ids": None,
            },
            "data": {"seed": 0, "split": "cs", "mode": "test", "size": 448},
            "prompt": {
                "system_instruction": "system",
                "output_format": "text",
                "cot": False,
                "cot_start_tag": "<think>",
                "cot_end_tag": "</think>",
                "labels": None,
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
                "name": "wanfall",
                "target_fps": 7.5,
                "vid_frame_count": 16,
                "video_datasets": [
                    {
                        "name": "wanfall",
                        "video_root": "/tmp/videos",
                        "annotations_file": "/tmp/annotations.csv",
                        "split_root": "/tmp/splits",
                    }
                ],
                "path_format": "{video_root}/{video_path}{ext}",
            },
            "wandb": {
                "mode": "disabled",
                "project": "demo-project",
                "name": None,
                "tags": ["baseline"],
            },
            "task": "classify",
            "model_fps": 7.5,
            "num_frames": 16,
            "batch_size": 32,
            "num_workers": 0,
            "prefetch_factor": 2,
            "output_dir": "outputs",
            "save_predictions": True,
            "save_metrics": True,
            "log_videos": 0,
            "num_samples": None,
            "embeddings_dir": "outputs/embeddings",
        }
    )


def test_create_name_and_tags_from_config_returns_base_name_and_tags():
    base_name, tags = create_name_and_tags_from_config(_config())

    assert base_name == "Qwen2.5-VL-7B-Instruct-F16@7.5"
    assert set(tags) == {"baseline", "wanfall", "qwen"}


def test_initialize_run_from_config_passes_generated_id_to_wandb(monkeypatch):
    config = _config()
    captured = {}

    def fake_generate_id() -> str:
        return "abc123"

    def fake_init(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name=kwargs["name"], id=kwargs["id"], tags=kwargs["tags"])

    monkeypatch.setattr("falldet.utils.wandb.wandb.util.generate_id", fake_generate_id)
    monkeypatch.setattr("falldet.utils.wandb.wandb.init", fake_init)

    run = initialize_run_from_config(config)

    assert captured["id"] == "abc123"
    assert captured["name"] == "Qwen2.5-VL-7B-Instruct-F16@7.5_abc123"
    assert run.id == "abc123"
