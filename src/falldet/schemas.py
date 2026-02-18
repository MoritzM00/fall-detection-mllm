"""Pydantic validation models for Hydra configuration.

Validates the full config once after OmegaConf.resolve(), then passes typed
objects to all downstream code. All YAML files and Hydra usage remain unchanged.
"""

from enum import StrEnum
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfig(BaseModel):
    """Base config that forbids extra fields. All sub-configs inherit from this."""

    model_config = ConfigDict(extra="forbid")


class RoleVariant(StrEnum):
    STANDARD = "standard"
    SPECIALIZED = "specialized"
    VIDEO_SPECIALIZED = "video_specialized"


class TaskVariant(StrEnum):
    STANDARD = "standard"
    EXTENDED = "extended"


class LabelsVariant(StrEnum):
    BULLETED = "bulleted"
    COMMA = "comma"
    GROUPED = "grouped"
    NUMBERED = "numbered"


class DefinitionsVariant(StrEnum):
    STANDARD = "standard"
    EXTENDED = "extended"


class PromptConfig(BaseConfig):
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
    role_variant: RoleVariant | None = RoleVariant.STANDARD
    task_variant: TaskVariant = TaskVariant.STANDARD
    labels_variant: LabelsVariant = LabelsVariant.BULLETED
    definitions_variant: DefinitionsVariant | None = None


class VLLMConfig(BaseConfig):
    """vLLM engine configuration."""

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
    limit_mm_per_prompt: dict[str, int] = {
        "image": 0,
        "video": 1,
    }  # mutable default is safe with pydantic
    enable_expert_parallel: bool | None = None


class ModelConfig(BaseConfig):
    """Model identification and loading configuration."""

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
    def coerce_version_to_str(cls, v: str | int | float) -> str:
        """Coerce YAML integer versions (e.g. 3) to strings."""
        return str(v)

    @property
    def name(self) -> str:
        """Resolve the model name (e.g. 'Qwen3-VL-4B-Instruct')."""
        from falldet.config import resolve_model_name_from_config

        return resolve_model_name_from_config(self)

    @property
    def path(self) -> str:
        """Resolve the full HuggingFace model path (e.g. 'Qwen/Qwen3-VL-4B-Instruct')."""
        from falldet.config import resolve_model_path_from_config

        return resolve_model_path_from_config(self)


class SamplingConfig(BaseConfig):
    """Sampling / decoding configuration for vLLM text generation."""

    temperature: float = Field(
        0.0,
        ge=0.0,
        description="Controls randomness. 0 = greedy/deterministic, higher = more random.",
    )
    max_tokens: int = Field(
        512,
        gt=0,
        description="Maximum number of tokens to generate.",
    )
    top_k: int = Field(
        0,
        ge=-1,
        description="Top-k filtering: only sample from the k most likely tokens. "
        "-1 and 0 disables top-k filtering (consider all tokens).",
    )
    top_p: float = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling: sample from smallest set of tokens with "
        "cumulative probability >= top_p. 1.0 disables nucleus sampling.",
    )
    presence_penalty: float = Field(
        0.0,
        le=2.0,
        ge=-2.0,
        description="Additive penalty for tokens that have already appeared. "
        "Positive values discourage repetition. Typically in range [-2, 2].",
    )
    frequency_penalty: float = Field(
        0.0,
        le=2.0,
        ge=-2.0,
        description="Additive penalty proportional to token frequency. "
        "Positive values discourage frequent tokens. Typically in range [-2, 2].",
    )
    repetition_penalty: float = Field(
        1.0,
        ge=1.0,
        description="Multiplicative penalty for repeated tokens. "
        "1.0 = no penalty, >1.0 = penalize repetition.",
    )
    seed: int | None = Field(
        None,
        description="Random seed for reproducibility. None = random seed each run.",
    )
    stop_token_ids: list[int] | None = Field(
        None,
        description="Token IDs that trigger generation stop.",
    )


class DataConfig(BaseConfig):
    """Data loading configuration."""

    seed: int = 0
    split: str = "cs"
    mode: str = "test"
    size: int | None = 448
    max_size: int | None = None


class WandbConfig(BaseConfig):
    """Weights & Biases configuration."""

    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "fall-detection-using-mllms"
    name: str | None = None
    tags: list[str] | None = None


class VideoDatasetItemConfig(BaseConfig):
    """Per-dataset entry in the video_datasets list."""

    name: str
    video_root: str
    annotations_file: str
    dataset_fps: float | None = None
    split_root: str
    split: str | None = None
    evaluation_group: str | None = None


class DatasetConfig(BaseConfig):
    """Top-level dataset group configuration."""

    name: str
    video_datasets: list[VideoDatasetItemConfig]
    target_fps: float
    vid_frame_count: int
    path_format: str = "{video_root}/{video_path}{ext}"
    num_classes: int | None = None
    metric_for_best_model: str | None = None
    create_all_combined: bool = False


class InferenceConfig(BaseConfig):
    """Root configuration composing all sub-configs."""

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


def from_dictconfig(cfg: DictConfig) -> InferenceConfig:
    """Convert an OmegaConf DictConfig to a validated InferenceConfig.

    Resolves all OmegaConf interpolations internally, so the caller does not
    need to call OmegaConf.resolve() beforehand.

    Args:
        cfg: OmegaConf DictConfig (resolved or unresolved).

    Returns:
        Validated InferenceConfig instance.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    raw.pop("hydra", None)
    return InferenceConfig(**raw)
