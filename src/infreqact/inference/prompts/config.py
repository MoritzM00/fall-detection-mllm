"""Configuration dataclass for prompt building."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptConfig:
    """Configuration for prompt building.

    Attributes:
        output_format: Expected output format - "json" or "text"
        include_role: Whether to include role description in prompt
        include_label_definitions: Whether to include label definitions and constraints
        include_constraints: Whether to include sequence rules and constraints
        cot: Whether to enable chain-of-thought reasoning
        cot_delimiter: Delimiter between reasoning and final answer (for CoT mode)
        few_shot_examples: Optional list of paths to few-shot example configs
        model_family: Model family name for model-specific adjustments (e.g., "qwen", "InternVL")
    """

    output_format: Literal["json", "text"] = "json"
    include_role: bool = True
    include_label_definitions: bool = True
    include_constraints: bool = True
    cot: bool = False
    cot_delimiter: str = "Final Answer:"
    few_shot_examples: list[str] | None = None
    model_family: str = "qwen"
