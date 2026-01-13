"""Output parsers for extracting predictions from LLM responses."""

import logging
import re
from dataclasses import dataclass
from typing import Protocol

import json_repair

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing LLM output.

    Attributes:
        label: Predicted label
        reasoning: Optional reasoning text (for CoT)
        raw_text: Raw output text from LLM
    """

    label: str
    reasoning: str | None = None
    raw_text: str = ""


class OutputParser(Protocol):
    """Protocol for output parsers."""

    def parse(self, text: str) -> ParseResult:
        """Parse LLM output text to extract prediction.

        Args:
            text: Raw output text from LLM

        Returns:
            ParseResult with extracted label and optional reasoning
        """
        ...


class JSONOutputParser:
    """Parser for JSON-formatted outputs."""

    def __init__(self, label2idx: dict):
        """Initialize JSON parser.

        Args:
            label2idx: Dictionary mapping valid labels to indices
        """
        self.label2idx = label2idx

    def parse(self, text: str) -> ParseResult:
        """Parse JSON output to extract label.

        Args:
            text: Raw output text from LLM (expected to be JSON)

        Returns:
            ParseResult with extracted label
        """
        try:
            json_obj = json_repair.loads(text)
            predicted_label = json_obj.get("label", "other")

            # Validate that the label exists in label2idx
            if predicted_label not in self.label2idx:
                logger.warning(
                    f"Invalid label '{predicted_label}' not in label2idx. Defaulting to 'other'."
                )
                predicted_label = "other"

            return ParseResult(
                label=predicted_label,
                reasoning=None,  # JSON format doesn't separate reasoning
                raw_text=text,
            )
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}\nGenerated text: {text}")
            return ParseResult(label="other", reasoning=None, raw_text=text)


class KeywordOutputParser:
    """Parser for plain text outputs using keyword matching."""

    def __init__(self, label2idx: dict):
        """Initialize keyword parser.

        Args:
            label2idx: Dictionary mapping valid labels to indices
        """
        self.label2idx = label2idx

    def parse(self, text: str) -> ParseResult:
        """Parse plain text output by searching for valid labels.

        Searches the text for any of the valid labels from label2idx
        and returns the first match found. Handles variants like "walking" -> "walk".

        Args:
            text: Raw output text from LLM

        Returns:
            ParseResult with extracted label (or "other" if no match)
        """
        text_lower = text.lower()

        # Sort labels by length (longest first) to match "sit_down" before "sitting"
        sorted_labels = sorted(self.label2idx.keys(), key=len, reverse=True)

        for label in sorted_labels:
            # Create multiple patterns to try
            patterns = []

            # For compound labels with underscores (e.g., "sit_down")
            if "_" in label:
                # Split on underscore and create flexible patterns
                parts = label.split("_")

                # Pattern 1: Match with spaces/underscores/hyphens between parts
                # "sit_down" -> matches "sit down", "sit_down", "sit-down"
                flexible_pattern = r"[\s_-]+".join([re.escape(part) for part in parts])
                patterns.append(rf"\b{flexible_pattern}\b")

                # Pattern 2: Handle conjugated forms of first part
                # "sit_down" -> matches "sits down", "sitting down"
                first_part = parts[0]
                rest_parts = parts[1:]
                rest_pattern = r"[\s_-]+".join([re.escape(part) for part in rest_parts])

                patterns.extend(
                    [
                        rf"\b{re.escape(first_part)}s[\s_-]+{rest_pattern}\b",  # "sits down"
                        rf"\b{re.escape(first_part)}ing[\s_-]+{rest_pattern}\b",  # "sitting down"
                    ]
                )

            # For simple labels without underscores
            patterns.extend(
                [
                    rf"\b{re.escape(label)}\b",  # Exact match
                    rf"\b{re.escape(label)}ing\b",  # verb-ing form
                    rf"\b{re.escape(label)}s\b",  # plural/third person
                ]
            )

            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return ParseResult(label=label, reasoning=None, raw_text=text)

        logger.warning(f"No valid label found in text. Defaulting to 'other'. Text: {text[:100]}")
        return ParseResult(label="other", reasoning=None, raw_text=text)


class CoTOutputParser:
    """Parser wrapper for chain-of-thought outputs using think tags.

    Extracts reasoning enclosed in tags (e.g., <think>...</think>)
    and parses the content after the closing tag for the label.
    Adopts vLLM's reasoning parser interface.
    """

    def __init__(
        self,
        label2idx: dict,
        answer_parser: OutputParser,
        start_tag: str = "<think>",
        end_tag: str = "</think>",
    ):
        """Initialize CoT parser.

        Args:
            label2idx: Dictionary mapping valid labels to indices
            answer_parser: Parser to use for extracting label from final answer
            start_tag: Opening tag for reasoning content (e.g., "<think>")
            end_tag: Closing tag for reasoning content (e.g., "</think>")
        """
        self.label2idx = label2idx
        self.answer_parser = answer_parser
        self.start_tag = start_tag
        self.end_tag = end_tag

    def extract_reasoning(self, text: str) -> tuple[str | None, str | None]:
        """Extract reasoning and content from model output.

        Follows vLLM's reasoning parser interface. Returns (reasoning, content)
        where reasoning is the text within tags, and content is text after tags.

        Handles 3 cases:
        1. Only end tag present (tokenizer already added start tag via add_generation_prompt)
        2. Both start and end tags present
        3. No tags present

        Args:
            text: Raw output text from LLM

        Returns:
            tuple: (reasoning, content) - either may be None
        """
        has_start = self.start_tag in text
        has_end = self.end_tag in text

        # Case 1: Only end tag (tokenizer added start tag via add_generation_prompt)
        # This is common with Qwen models where <think> is added by tokenizer
        if not has_start and has_end:
            reasoning, _, content = text.partition(self.end_tag)
            return reasoning.strip(), content.strip() if content else None

        # Case 2: No tags at all - entire text is content
        if not has_start:
            return None, text

        # Case 3: Start tag present - use existing logic
        _, _, after_start = text.partition(self.start_tag)

        if not has_end:
            # No end tag - treat everything after start as reasoning
            logger.warning(
                f"CoT end tag '{self.end_tag}' not found after start tag. "
                f"Treating remaining text as reasoning only."
            )
            return after_start.strip(), None

        # Split on end tag
        reasoning, _, content = after_start.partition(self.end_tag)
        return reasoning.strip(), content.strip() if content else None

    def parse(self, text: str) -> ParseResult:
        """Parse CoT output by extracting reasoning and parsing content.

        Args:
            text: Raw output text from LLM (with reasoning + final answer)

        Returns:
            ParseResult with extracted label and reasoning
        """
        reasoning, content = self.extract_reasoning(text)

        # Parse the content portion for the label
        if content:
            result = self.answer_parser.parse(content)
        else:
            # Fallback: parse entire text as answer
            logger.warning("No content found after reasoning tags. Parsing entire text as answer.")
            result = self.answer_parser.parse(text)

        # Add reasoning to result
        result.reasoning = reasoning
        result.raw_text = text

        return result
