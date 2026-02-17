"""Pydantic validation models for Hydra configuration.

Validates the full config once after OmegaConf.resolve(), then passes typed
objects to all downstream code. All YAML files and Hydra usage remain unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

# Type aliases for prompt variant selection
RoleVariant = Literal["standard", "specialized", "video_specialized"]
TaskVariant = Literal["standard", "extended"]
LabelsVariant = Literal["bulleted", "comma", "grouped", "numbered"]
DefinitionsVariant = Literal["standard", "extended"]


class PromptConfig(BaseModel):
    """Configuration for prompt building.

    Attributes:
        output_format: Expected output format - "json" or "text"
        cot: Whether to enable chain-of-thought reasoning
        cot_start_tag: Opening tag for reasoning content (default: "<think>")
        cot_end_tag: Closing tag for reasoning content (default: "</think>")
        labels: Optional list of labels to include in prompt. If None, uses hardcoded defaults
        model_family: Model family name for model-specific adjustments (e.g., "qwen", "InternVL")
        num_shots: Number of few-shot exemplars (0 = zero-shot)
        shot_selection: Exemplar sampling strategy - "random" or "balanced"
        exemplar_seed: Random seed for exemplar sampling reproducibility
        role_variant: Which role component variant to use (None = omit role section)
        task_variant: Which task instruction variant to use
        labels_variant: Which label formatting variant to use
        definitions_variant: Which definitions component variant to use (None = omit definitions)
    """

    model_config = ConfigDict(extra="forbid")

    output_format: Literal["json", "text"] = "json"
    cot: bool = False
    cot_start_tag: str = "<think>"
    cot_end_tag: str = "</think>"
    labels: list[str] | None = None
    model_family: str = "qwen"

    # Few-shot settings
    num_shots: int = 0
    shot_selection: Literal["random", "balanced"] = "balanced"
    exemplar_seed: int = 42

    # Variant selectors
    role_variant: RoleVariant | None = "standard"
    task_variant: TaskVariant = "standard"
    labels_variant: LabelsVariant = "bulleted"
    definitions_variant: DefinitionsVariant | None = None


class VLLMConfig(BaseModel):
    """vLLM engine configuration."""

    model_config = ConfigDict(extra="forbid")

    use_mock: bool = False
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float = 0.9
    mm_encoder_tp_mode: str = "data"
    mm_processor_cache_gb: int = 0
    seed: int = 0
    dtype: str = "bfloat16"
    enforce_eager: bool = False
    max_model_len: int | None = -1
    max_num_batched_tokens: int | None = None
    trust_remote_code: bool = True
    async_scheduling: bool = True
    skip_mm_profiling: bool = False
    enable_prefix_caching: bool = False
    mm_processor_kwargs: dict[str, Any] = {}
    limit_mm_per_prompt: dict[str, int] = {"image": 0, "video": 1}
    enable_expert_parallel: bool | None = None


class ModelConfig(BaseModel):
    """Model identification and loading configuration."""

    model_config = ConfigDict(extra="forbid")

    org: str
    family: str
    version: str
    variant: str | None = None
    params: str
    active_params: str | None = None
    needs_video_metadata: bool = True
    mm_processor_kwargs: dict[str, Any] = {}

    @field_validator("version", mode="before")
    @classmethod
    def coerce_version_to_str(cls, v: Any) -> str:
        """Coerce YAML integer versions (e.g. 3) to strings."""
        return str(v)


class SamplingConfig(BaseModel):
    """Sampling / decoding configuration."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.0
    max_tokens: int = 512
    top_k: int = -1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop_token_ids: list[int] | None = None


class DataConfig(BaseModel):
    """Data loading configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    split: str = "cs"
    mode: str = "test"
    size: int | None = 448
    max_size: int | None = None


class WandbConfig(BaseModel):
    """Weights & Biases configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "fall-detection-using-mllms"
    name: str | None = None
    tags: list[str] | None = None


class VideoDatasetItemConfig(BaseModel):
    """Per-dataset entry in the video_datasets list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    video_root: str
    annotations_file: str
    dataset_fps: float | None = None
    split_root: str
    split: str | None = None
    evaluation_group: str | None = None


class DatasetConfig(BaseModel):
    """Top-level dataset group configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    video_datasets: list[VideoDatasetItemConfig]
    target_fps: float
    vid_frame_count: int
    path_format: str = "{video_root}/{video_path}{ext}"
    num_classes: int | None = None
    metric_for_best_model: str | None = None
    create_all_combined: bool = False


class InferenceConfig(BaseModel):
    """Root configuration composing all sub-configs."""

    model_config = ConfigDict(extra="forbid")

    # Sub-configs
    vllm: VLLMConfig
    model: ModelConfig
    sampling: SamplingConfig
    data: DataConfig
    prompt: PromptConfig
    dataset: DatasetConfig
    wandb: WandbConfig

    # Root-level fields
    model_fps: float = 7.5
    num_frames: int = 16
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int = 2
    output_dir: str = "outputs"
    save_predictions: bool = True
    save_metrics: bool = True
    log_videos: int = 1
    num_samples: int | None = None

    # Mode-specific dataset overrides
    dataset_train: DatasetConfig | None = None
    dataset_val: DatasetConfig | None = None
    dataset_test: DatasetConfig | None = None


def from_dictconfig(cfg: Any) -> InferenceConfig:
    """Convert a resolved OmegaConf DictConfig to a validated InferenceConfig.

    Args:
        cfg: Resolved OmegaConf DictConfig (after OmegaConf.resolve()).

    Returns:
        Validated InferenceConfig instance.
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(cfg, resolve=True)
    raw.pop("hydra", None)
    return InferenceConfig(**raw)
