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

    def parse(self, text: str, label2idx: dict) -> ParseResult:
        """Parse LLM output text to extract prediction.

        Args:
            text: Raw output text from LLM
            label2idx: Dictionary mapping valid labels to indices

        Returns:
            ParseResult with extracted label and optional reasoning
        """
        ...


class JSONOutputParser:
    """Parser for JSON-formatted outputs."""

    def parse(self, text: str, label2idx: dict) -> ParseResult:
        """Parse JSON output to extract label.

        Args:
            text: Raw output text from LLM (expected to be JSON)
            label2idx: Dictionary mapping valid labels to indices

        Returns:
            ParseResult with extracted label
        """
        try:
            json_obj = json_repair.loads(text)
            predicted_label = json_obj.get("label", "other")

            # Validate that the label exists in label2idx
            if predicted_label not in label2idx:
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

    def parse(self, text: str, label2idx: dict) -> ParseResult:
        """Parse plain text output by searching for valid labels.

        Searches the text for any of the valid labels from label2idx
        and returns the first match found. Handles variants like "walking" -> "walk".

        Args:
            text: Raw output text from LLM
            label2idx: Dictionary mapping valid labels to indices

        Returns:
            ParseResult with extracted label (or "other" if no match)
        """
        text_lower = text.lower()

        # Sort labels by length (longest first) to match "sit_down" before "sitting"
        sorted_labels = sorted(label2idx.keys(), key=len, reverse=True)

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
                    logger.debug(f"Matched label '{label}' in text: {text[:100]}")
                    return ParseResult(label=label, reasoning=None, raw_text=text)

        logger.warning(f"No valid label found in text. Defaulting to 'other'. Text: {text[:100]}")
        return ParseResult(label="other", reasoning=None, raw_text=text)


class CoTOutputParser:
    """Parser wrapper for chain-of-thought outputs.

    Splits the output on a delimiter to separate reasoning from final answer,
    then uses a wrapped parser to extract the label from the final answer.
    """

    def __init__(self, answer_parser: OutputParser, delimiter: str = "Final Answer:"):
        """Initialize CoT parser.

        Args:
            answer_parser: Parser to use for extracting label from final answer
            delimiter: String that separates reasoning from final answer
        """
        self.answer_parser = answer_parser
        self.delimiter = delimiter

    def parse(self, text: str, label2idx: dict) -> ParseResult:
        """Parse CoT output by splitting reasoning and answer.

        Args:
            text: Raw output text from LLM (with reasoning + final answer)
            label2idx: Dictionary mapping valid labels to indices

        Returns:
            ParseResult with extracted label and reasoning
        """
        if self.delimiter in text:
            # Split on the last occurrence of delimiter
            reasoning, answer = text.rsplit(self.delimiter, 1)
            reasoning = reasoning.strip()
            answer = answer.strip()
        else:
            # Fallback: no delimiter found, parse entire text as answer
            logger.warning(
                f"CoT delimiter '{self.delimiter}' not found in output. "
                f"Parsing entire text as answer."
            )
            reasoning = ""
            answer = text

        # Parse the answer portion
        result = self.answer_parser.parse(answer, label2idx)

        # Add reasoning to result
        result.reasoning = reasoning
        result.raw_text = text

        return result
