"""Configuration dataclass for prompt building."""

from dataclasses import dataclass
from typing import Literal

# Type aliases for variant selection
RoleVariant = Literal["standard", "specialized", "video_specialized"]
TaskVariant = Literal["standard", "extended"]
LabelsVariant = Literal["bulleted", "comma", "grouped", "numbered"]
DefinitionsVariant = Literal["standard", "extended"]


@dataclass
class PromptConfig:
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
    role_variant: RoleVariant | None = "standard"
    task_variant: TaskVariant = "standard"
    labels_variant: LabelsVariant = "bulleted"
    definitions_variant: DefinitionsVariant | None = None
