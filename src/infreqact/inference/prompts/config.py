"""Configuration dataclass for prompt building."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptConfig:
    """Configuration for prompt building.

    Attributes:
        output_format: Expected output format - "json" or "text"
        include_role: Whether to include role description in prompt
        include_definitions: Whether to include label definitions and constraints
        include_adherence: Whether to include adherence instruction
        cot: Whether to enable chain-of-thought reasoning
        cot_start_tag: Opening tag for reasoning content (default: "<think>")
        cot_end_tag: Closing tag for reasoning content (default: "</think>")
        labels: Optional list of labels to include in prompt. If None, uses hardcoded defaults
        model_family: Model family name for model-specific adjustments (e.g., "qwen", "InternVL")
        num_shots: Number of few-shot exemplars (0 = zero-shot)
        shot_selection: Exemplar sampling strategy - "random" or "balanced"
        exemplar_seed: Random seed for exemplar sampling reproducibility
    """

    output_format: Literal["json", "text"] = "json"
    include_role: bool = True
    include_definitions: bool = True
    include_adherence: bool = True
    cot: bool = False
    cot_start_tag: str = "<think>"
    cot_end_tag: str = "</think>"
    labels: list[str] | None = None
    model_family: str = "qwen"

    # Few-shot settings
    num_shots: int = 0
    shot_selection: Literal["random", "balanced"] = "balanced"
    exemplar_seed: int = 42
