"""Prompt builder for assembling video action recognition prompts."""

from .components import (
    ADHERENCE_INSTRUCTION,
    CONSTRAINTS_COMPONENT,
    COT_INSTRUCTION,
    DEFINITIONS_COMPONENT,
    INTERNVL_R1_SYSTEM_PROMPT,
    JSON_OUTPUT_FORMAT,
    LABELS_COMPONENT,
    ROLE_COMPONENT,
    TEXT_OUTPUT_FORMAT,
)
from .config import PromptConfig
from .parsers import CoTOutputParser, JSONOutputParser, KeywordOutputParser, OutputParser


class PromptBuilder:
    """Builds prompts from modular components based on configuration."""

    def __init__(self, config: PromptConfig):
        """Initialize the prompt builder.

        Args:
            config: Configuration specifying which prompt components to include
        """
        self.config = config

    def build_prompt(self) -> str:
        """Assemble the full prompt from components.

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. Role section (optional)
        if self.config.include_role:
            sections.append(ROLE_COMPONENT)

        # 2. Labels section (always included)
        sections.append(LABELS_COMPONENT)

        # 3. Definitions & Constraints (optional)
        if self.config.include_label_definitions:
            sections.append(DEFINITIONS_COMPONENT)
        if self.config.include_constraints:
            sections.append(CONSTRAINTS_COMPONENT)

        # 4. Few-shot examples (optional)
        if self.config.few_shot_examples:
            sections.append(self._build_fewshot_section())

        # 5. Chain-of-thought instruction (optional)
        if self.config.cot:
            sections.append(COT_INSTRUCTION)

        # 6. Output format instruction
        sections.append(self._build_output_format())

        # 7. Adherence instruction
        sections.append(ADHERENCE_INSTRUCTION)

        return "\n\n".join(sections)

    def _needs_internvl_r1_prefix(self) -> bool:
        """Check if InternVL R1 system prompt is needed.

        Returns:
            True if model is InternVL and CoT is enabled
        """
        return self.config.cot and self.config.model_family.lower() == "internvl"

    def _build_output_format(self) -> str:
        """Generate output format instructions based on config.

        Returns:
            Output format instruction string
        """
        if self.config.output_format == "json":
            return JSON_OUTPUT_FORMAT
        return TEXT_OUTPUT_FORMAT

    def _build_fewshot_section(self) -> str:
        """Build few-shot examples section.

        Returns:
            Few-shot examples formatted as a prompt section
        """

        import yaml

        if not self.config.few_shot_examples:
            return ""

        examples_text = ["Examples:"]

        for i, example_path in enumerate(self.config.few_shot_examples, 1):
            try:
                # Load example config
                with open(example_path) as f:
                    example = yaml.safe_load(f)

                label = example.get("label", "unknown")
                description = example.get("description", "")

                # Format example
                example_text = f"Example {i}: Video shows {description} → Label: {label}"
                examples_text.append(example_text)
            except Exception as e:
                # Log warning but continue
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load few-shot example from {example_path}: {e}")

        return "\n".join(examples_text)

    def get_parser(self) -> OutputParser:
        """Return the appropriate parser for this prompt's output format.

        Returns:
            OutputParser instance matching the configured output format
        """
        # Select base parser based on output format
        if self.config.output_format == "json":
            base_parser = JSONOutputParser()
        else:
            base_parser = KeywordOutputParser()

        # Wrap with CoT parser if chain-of-thought is enabled
        if self.config.cot:
            return CoTOutputParser(base_parser, delimiter=self.config.cot_delimiter)

        return base_parser

    def get_system_message(self) -> dict | None:
        """Return system message dict for models that need it (e.g., InternVL CoT).

        Returns:
            System message dict with role and content, or None if not needed
        """
        if self._needs_internvl_r1_prefix():
            return {
                "role": "system",
                "content": [{"type": "text", "text": INTERNVL_R1_SYSTEM_PROMPT}],
            }
        return None
