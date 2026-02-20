"""Conversation builder for zero-shot and few-shot inference."""

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch

from falldet.schemas import InferenceConfig, PromptConfig

from .prompts import PromptBuilder
from .prompts.components import EXEMPLAR_USER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class VideoWithMetadata:
    """Video frames with associated metadata for vLLM."""

    frames: torch.Tensor
    metadata: dict  # {total_num_frames, fps, frames_indices}


@dataclass
class ConversationData:
    """Intermediate representation before vLLM preparation."""

    messages: list[dict]
    videos: list[VideoWithMetadata]


class ConversationBuilder:
    """Builds conversations for zero-shot and few-shot inference.

    Encapsulates:
    - Exemplar caching (frames + metadata)
    - Message template building
    - vLLM input preparation

    Works identically for zero-shot (num_shots=0) and few-shot (num_shots>0).
    """

    def __init__(
        self,
        config: PromptConfig,
        label2idx: dict,
        exemplars: list[dict] | None = None,
        model_fps: float = 8.0,
        needs_video_metadata: bool = True,
    ):
        """Initialize the conversation builder.

        Args:
            config: Prompt configuration
            label2idx: Label to index mapping
            exemplars: Pre-sampled exemplars (empty list or None for zero-shot)
            model_fps: Frame rate for video metadata
            needs_video_metadata: Whether model requires video metadata
        """
        self.config = config
        self.label2idx = label2idx
        self.model_fps = model_fps
        self.needs_video_metadata = needs_video_metadata

        self._prompt_builder = PromptBuilder(config, label2idx)
        self._exemplars = exemplars or []

        # Cache at initialization
        self._template_cache: list[dict] = self._build_template()
        self._videos_cache: list[VideoWithMetadata] = self._build_video_cache()
        self._user_prompt: str = self._prompt_builder.build_prompt()

        logger.info(
            f"ConversationBuilder initialized: {len(self._exemplars)} exemplars, "
            f"{self.num_videos} videos/request"
        )
        self._log_conversation()

    def _build_template(self) -> list[dict]:
        """Build the static message template (cached)."""
        messages = []

        # System message if needed (e.g., InternVL CoT)
        if system_msg := self._prompt_builder.get_system_message():
            messages.append(system_msg)

        # Exemplar turns (empty for zero-shot)
        for exemplar in self._exemplars:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": exemplar["video"]},
                        {"type": "text", "text": EXEMPLAR_USER_PROMPT},
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": self._format_answer(exemplar["label_str"]),
                        },
                    ],
                }
            )

        return messages

    def _build_video_cache(self) -> list[VideoWithMetadata]:
        """Cache exemplar videos with their metadata."""
        return [
            VideoWithMetadata(
                frames=exemplar["video"],
                metadata=self._build_video_metadata(exemplar["video"]),
            )
            for exemplar in self._exemplars
        ]

    def _build_video_metadata(self, frames: torch.Tensor) -> dict:
        """Build metadata dict for a video."""
        return dict(
            total_num_frames=frames.shape[0],
            fps=self.model_fps,
            frames_indices=list(range(frames.shape[0])),
        )

    def build(self, target_video: torch.Tensor) -> ConversationData:
        """Build conversation data for a target video.

        Args:
            target_video: Target video frames

        Returns:
            ConversationData with messages and videos
        """
        # Shallow copy template and append target message
        messages = self._template_cache.copy()
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": target_video},
                    {"type": "text", "text": self._user_prompt},
                ],
            }
        )

        # Combine cached videos with target
        target_with_meta = VideoWithMetadata(
            frames=target_video,
            metadata=self._build_video_metadata(target_video),
        )
        videos = [*self._videos_cache, target_with_meta]

        return ConversationData(messages=messages, videos=videos)

    def build_vllm_inputs(
        self,
        target_video: torch.Tensor,
        processor,
    ) -> dict:
        """Build ready-to-use vLLM inputs for a target video.

        Args:
            target_video: Target video frames
            processor: AutoProcessor instance

        Returns:
            Dict ready for llm.generate()
        """
        conv_data = self.build(target_video)

        # Apply chat template
        text = processor.apply_chat_template(
            conv_data.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build multi-modal data with list of (frames, metadata) tuples
        if self.needs_video_metadata:
            mm_data = dict(video=[(v.frames, v.metadata) for v in conv_data.videos])
        else:
            mm_data = dict(video=[v.frames for v in conv_data.videos])

        return dict(
            prompt=text,
            multi_modal_data=mm_data,
            mm_processor_kwargs=dict(do_sample_frames=False),
        )

    def _format_answer(self, label: str) -> str:
        """Format exemplar answer based on output format."""
        if self.config.output_format == "json":
            return f'{{"label": "{label}"}}'
        return f"The best answer is: {label}"

    def _format_content_for_logging(self, content: list[dict], max_text_len: int = 200) -> str:
        """Format message content for logging, replacing video tensors with shape info.

        Args:
            content: List of content items (video/text dicts)
            max_text_len: Maximum text length before truncation

        Returns:
            Formatted string representation of the content
        """
        parts = []
        for item in content:
            if item["type"] == "video":
                video = item.get("video")
                if isinstance(video, torch.Tensor):
                    shape_str = "x".join(str(d) for d in video.shape)
                    parts.append(f"<video: [{shape_str}]>")
                else:
                    parts.append("<video>")
            elif item["type"] == "text":
                text = item.get("text", "")
                if len(text) > max_text_len:
                    text = text[:max_text_len] + "..."
                parts.append(text)
        return " ".join(parts)

    def _log_conversation(self) -> None:
        """Log the conversation structure at initialization."""
        # Build a preview including template + target placeholder
        lines = ["Conversation structure:"]
        for i, msg in enumerate(self._template_cache):
            role = msg["role"]
            content_str = self._format_content_for_logging(msg["content"])
            lines.append(f"  [{i}] {role}: {content_str}")

        # Add placeholder for target message
        target_idx = len(self._template_cache)
        target_prompt_preview = (
            self._user_prompt[:200] + "..." if len(self._user_prompt) > 200 else self._user_prompt
        )
        lines.append(f"  [{target_idx}] user: <video: [target]> {target_prompt_preview}")

        logger.info("\n".join(lines))

    @property
    def num_videos(self) -> int:
        """Number of videos per request (for vLLM limit config)."""
        return len(self._exemplars) + 1

    @property
    def user_prompt(self) -> str:
        """Get the user prompt text."""
        return self._user_prompt

    @property
    def parser(self):
        """Get the output parser."""
        return self._prompt_builder.get_parser()


def create_conversation_builder(
    config: InferenceConfig,
    label2idx: dict,
) -> ConversationBuilder:
    """Factory function to create and initialize a ConversationBuilder.

    Handles exemplar sampling if num_shots > 0, keeping this logic
    outside the main inference script.

    Args:
        config: Validated inference configuration
        label2idx: Label to index mapping

    Returns:
        Initialized ConversationBuilder with cached exemplars
    """
    from falldet.data.exemplar_sampler import ExemplarSampler
    from falldet.data.video_dataset_factory import get_video_datasets

    prompt_config = config.prompt.model_copy(update={"labels": list(label2idx.keys())})

    # Sample exemplars if few-shot mode
    exemplars = []
    if config.prompt.num_shots > 0:
        logger.info(f"Setting up {config.prompt.num_shots}-shot prompting...")
        train_datasets = get_video_datasets(
            config=config,
            mode="train",
            split=config.data.split,
            size=config.data.size,
            seed=config.data.seed,
            return_individual=True,
        )
        train_datasets = cast(dict[str, Any], train_datasets)
        train_dataset = list(train_datasets["individual"].values())[0]

        sampler = ExemplarSampler(
            dataset=train_dataset,
            num_shots=config.prompt.num_shots,
            strategy=config.prompt.shot_selection,
            seed=config.prompt.exemplar_seed,
        )
        exemplars = sampler.sample()

    return ConversationBuilder(
        config=prompt_config,
        label2idx=label2idx,
        exemplars=exemplars,
        model_fps=config.model_fps,
        needs_video_metadata=config.model.needs_video_metadata,
    )
