"""
Mock vLLM engine for debugging purposes.

This module provides mock implementations of vLLM classes that return random
predictions without requiring GPU resources or model loading. Useful for
rapid development and debugging of inference pipelines.

Supports both JSON and text output formats with optional chain-of-thought reasoning.
"""

from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import SamplingParams

from falldet.data.video_dataset import label2idx

logger = logging.getLogger(__name__)

# Valid labels from the action recognition taxonomy (excluding "no_fall")
VALID_LABELS = list(label2idx.keys())

# Templates for chain-of-thought reasoning
REASON_TEMPLATES = [
    "The person appears to be performing a {label} action based on body position and movement.",
    "Analysis of the movement pattern indicates a {label} activity.",
    "The subject's posture and motion suggest they are {label}.",
    "Based on the visual cues, this appears to be a {label} action.",
    "The observed behavior is consistent with a {label} classification.",
]


class MockCompletionOutput:
    """Mock version of vLLM's CompletionOutput class."""

    def __init__(self, text: str):
        """
        Initialize mock completion output.

        Args:
            text: Generated text (JSON string or plain text with label and optional reason)
        """
        self.text = text


class MockRequestOutput:
    """Mock version of vLLM's RequestOutput class."""

    def __init__(self, outputs: list[MockCompletionOutput]):
        """
        Initialize mock request output.

        Args:
            outputs: List of completion outputs (typically contains one)
        """
        self.outputs = outputs


class MockLLM:
    """
    Mock version of vLLM's LLM class for debugging.

    Generates random predictions from the valid label set without requiring
    GPU resources or model loading. Supports chain-of-thought reasoning and
    seed-based reproducibility.
    """

    def __init__(
        self,
        model: str,
        seed: int = 0,
        cot: bool = False,
        output_format: str = "json",
        **kwargs,
    ):
        """
        Initialize mock LLM.

        Args:
            model: Model path (ignored in mock, kept for compatibility)
            seed: Random seed for reproducible predictions
            cot: Whether to include chain-of-thought reasoning in outputs
            output_format: Output format - "json" or "text"
            **kwargs: Other vLLM parameters (accepted but ignored for compatibility)
        """
        self.model = model
        self.seed = seed
        self.cot = cot
        self.output_format = output_format
        self.rng = random.Random(seed)

        # Log ignored parameters for transparency
        ignored_params = list(kwargs.keys())
        if ignored_params:
            logger.debug(f"MockLLM ignoring these vLLM parameters: {ignored_params}")

        logger.info(f"MockLLM initialized (seed={seed}, cot={cot}, output_format={output_format})")
        logger.info("Mock mode active - no GPU required, generating random predictions")

    def generate(
        self,
        inputs: list[dict[str, Any]],
        sampling_params: SamplingParams | None = None,
    ) -> list[MockRequestOutput]:
        """
        Generate mock predictions for a batch of inputs.

        Args:
            inputs: List of input dictionaries (same format as vLLM expects)
                   Each dict contains: "prompt", "multi_modal_data", "mm_processor_kwargs"
            sampling_params: Sampling parameters (only seed is used if provided)

        Returns:
            List of MockRequestOutput objects matching vLLM's output structure.
            Each output contains either:
            - JSON format: JSON string with randomly selected label and optional reason
            - Text format: Plain text with optional reasoning followed by label
        """
        # Use sampling_params seed if provided, otherwise use constructor seed
        if sampling_params and sampling_params.seed is not None:
            self.rng = random.Random(sampling_params.seed)

        outputs = []

        for i, input_dict in enumerate(inputs):
            # Randomly select a valid label
            label = self.rng.choice(VALID_LABELS)

            # Generate output based on format
            if self.output_format == "json":
                # Generate JSON output with optional chain-of-thought
                if self.cot:
                    reason_template = self.rng.choice(REASON_TEMPLATES)
                    reason = reason_template.format(label=label)
                    json_output = {"reason": reason, "label": label}
                else:
                    json_output = {"label": label}

                # Convert to JSON string
                text = json.dumps(json_output)

            else:  # text format
                # Generate text output with optional chain-of-thought
                if self.cot:
                    reason_template = self.rng.choice(REASON_TEMPLATES)
                    reason = reason_template.format(label=label)
                    text = f"{reason}\n{label}"
                else:
                    text = label

            # Wrap in mock output structures matching vLLM's interface
            completion_output = MockCompletionOutput(text=text)
            request_output = MockRequestOutput(outputs=[completion_output])
            outputs.append(request_output)

        return outputs
