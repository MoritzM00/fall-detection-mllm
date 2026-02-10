"""Prompt builder for assembling video action recognition prompts."""

from .components import (
    COT_INSTRUCTION,
    DEFINITIONS_VARIANTS,
    LABEL_FORMAT_VARIANTS,
    LABELS_COMPONENT,
    OUTPUT_FORMAT_VARIANTS,
    R1_SYSTEM_PROMPT,
    ROLE_VARIANTS,
    TASK_VARIANTS,
)
from .config import PromptConfig
from .parsers import CoTOutputParser, JSONOutputParser, KeywordOutputParser, OutputParser


class PromptBuilder:
    """Builds prompts from modular components based on configuration."""

    def __init__(self, config: PromptConfig, label2idx: dict):
        """Initialize the prompt builder.

        Args:
            config: Configuration specifying which prompt components to include
            label2idx: Dictionary mapping valid labels to indices
        """
        self.config = config
        self.label2idx = label2idx

    def build_prompt(self) -> str:
        """Assemble the full prompt from components.

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. Role section (optional, controlled by variant)
        if self.config.role_variant:
            sections.append(ROLE_VARIANTS[self.config.role_variant])

        # 2. Task instruction (always included)
        sections.append(TASK_VARIANTS[self.config.task_variant])

        # 3. Labels section (always included)
        sections.append(self._build_labels_section())

        # 4. Definitions & Constraints (optional, controlled by variant)
        if self.config.definitions_variant:
            sections.append(DEFINITIONS_VARIANTS[self.config.definitions_variant])

        # 5. Chain-of-thought instruction (optional)
        if self.config.cot:
            sections.append(COT_INSTRUCTION)

        # 6. Output format instruction
        sections.append(OUTPUT_FORMAT_VARIANTS[self.config.output_format])

        return "\n\n".join(sections)

    def _needs_r1_prefix(self) -> bool:
        """Check if InternVL R1 system prompt is needed.

        Returns:
            True if model is InternVL and CoT is enabled
        """
        return self.config.cot and self.config.model_family.lower() == "internvl"

    def _build_labels_section(self) -> str:
        """Build labels section from config or use default.

        Returns:
            Labels section string
        """
        # If custom labels are provided, use the configured variant formatter
        if self.config.labels:
            formatter = LABEL_FORMAT_VARIANTS[self.config.labels_variant]
            return formatter(self.config.labels)

        # Otherwise, use the default hardcoded labels component
        return LABELS_COMPONENT

    def get_parser(self) -> OutputParser:
        """Return the appropriate parser for this prompt's output format.

        Returns:
            OutputParser instance matching the configured output format
        """
        # Select base parser based on output format
        if self.config.output_format == "json":
            base_parser = JSONOutputParser(self.label2idx)
        else:
            base_parser = KeywordOutputParser(self.label2idx)

        # Wrap with CoT parser if chain-of-thought is enabled
        if self.config.cot:
            return CoTOutputParser(
                self.label2idx,
                base_parser,
                start_tag=self.config.cot_start_tag,
                end_tag=self.config.cot_end_tag,
            )

        return base_parser

    def get_system_message(self) -> dict | None:
        """Return system message dict for models that need it (e.g., InternVL CoT).

        Returns:
            System message dict with role and content, or None if not needed
        """
        if self._needs_r1_prefix():
            return {
                "role": "system",
                "content": [{"type": "text", "text": R1_SYSTEM_PROMPT}],
            }
        return None
