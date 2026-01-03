"""
Mock vLLM engine for debugging purposes.

This module provides mock implementations of vLLM classes that return random
predictions without requiring GPU resources or model loading. Useful for
rapid development and debugging of inference pipelines.
"""

import json
import logging
import random
from typing import Any

from infreqact.data.video_dataset import label2idx

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
            text: Generated text (JSON string with label and optional reason)
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


class MockSamplingParams:
    """Mock version of vLLM's SamplingParams class."""

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_k: int = -1,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        stop_token_ids: list[int] | None = None,
        **kwargs,
    ):
        """
        Initialize mock sampling parameters.

        Accepts all vLLM SamplingParams arguments for compatibility, but only
        uses the seed parameter. All other parameters are stored but ignored.

        Args:
            temperature: Sampling temperature (ignored in mock)
            max_tokens: Maximum tokens to generate (ignored in mock)
            top_k: Top-k sampling parameter (ignored in mock)
            top_p: Top-p sampling parameter (ignored in mock)
            presence_penalty: Presence penalty (ignored in mock)
            frequency_penalty: Frequency penalty (ignored in mock)
            repetition_penalty: Repetition penalty (ignored in mock)
            seed: Random seed for reproducibility (used by mock)
            stop_token_ids: Stop token IDs (ignored in mock)
            **kwargs: Additional parameters (ignored in mock)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.stop_token_ids = stop_token_ids


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
        **kwargs,
    ):
        """
        Initialize mock LLM.

        Args:
            model: Model path (ignored in mock, kept for compatibility)
            seed: Random seed for reproducible predictions
            cot: Whether to include chain-of-thought reasoning in outputs
            **kwargs: Other vLLM parameters (accepted but ignored for compatibility)
        """
        self.model = model
        self.seed = seed
        self.cot = cot
        self.rng = random.Random(seed)

        # Log ignored parameters for transparency
        ignored_params = list(kwargs.keys())
        if ignored_params:
            logger.debug(f"MockLLM ignoring these vLLM parameters: {ignored_params}")

        logger.info(f"MockLLM initialized (seed={seed}, cot={cot})")
        logger.info("Mock mode active - no GPU required, generating random predictions")

    def generate(
        self,
        inputs: list[dict[str, Any]],
        sampling_params: MockSamplingParams | None = None,
    ) -> list[MockRequestOutput]:
        """
        Generate mock predictions for a batch of inputs.

        Args:
            inputs: List of input dictionaries (same format as vLLM expects)
                   Each dict contains: "prompt", "multi_modal_data", "mm_processor_kwargs"
            sampling_params: Sampling parameters (only seed is used if provided)

        Returns:
            List of MockRequestOutput objects matching vLLM's output structure.
            Each output contains a JSON string with randomly selected label and
            optional chain-of-thought reasoning.
        """
        # Use sampling_params seed if provided, otherwise use constructor seed
        if sampling_params and sampling_params.seed is not None:
            self.rng = random.Random(sampling_params.seed)

        outputs = []

        for i, input_dict in enumerate(inputs):
            # Randomly select a valid label
            label = self.rng.choice(VALID_LABELS)

            # Generate JSON output with optional chain-of-thought
            if self.cot:
                reason_template = self.rng.choice(REASON_TEMPLATES)
                reason = reason_template.format(label=label)
                json_output = {"reason": reason, "label": label}
            else:
                json_output = {"label": label}

            # Convert to JSON string
            text = json.dumps(json_output)

            # Wrap in mock output structures matching vLLM's interface
            completion_output = MockCompletionOutput(text=text)
            request_output = MockRequestOutput(outputs=[completion_output])
            outputs.append(request_output)

        return outputs
