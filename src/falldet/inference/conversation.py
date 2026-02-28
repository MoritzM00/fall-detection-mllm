"""Conversation builder for zero-shot and few-shot inference."""

import logging
from dataclasses import dataclass

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

    Exemplars are supplied per-call via ``build()`` / ``build_vllm_inputs()``,
    allowing each query to receive different exemplars (e.g. similarity-based
    retrieval, or fresh random samples).

    Works identically for zero-shot (no exemplars) and few-shot (exemplars
    passed at build time).
    """

    def __init__(
        self,
        config: PromptConfig,
        label2idx: dict,
        model_fps: float = 8.0,
        needs_video_metadata: bool = True,
    ):
        """Initialize the conversation builder.

        Args:
            config: Prompt configuration
            label2idx: Label to index mapping
            model_fps: Frame rate for video metadata
            needs_video_metadata: Whether model requires video metadata
        """
        self.config = config
        self.label2idx = label2idx
        self.model_fps = model_fps
        self.needs_video_metadata = needs_video_metadata

        self._prompt_builder = PromptBuilder(config, label2idx)
        self._user_prompt: str = self._prompt_builder.build_prompt()

        # Cache the system message (None when not needed)
        self._system_msg: dict | None = self._prompt_builder.get_system_message()

        logger.info(f"ConversationBuilder initialized: {self.num_videos} videos/request")
        self._log_conversation()

    def _build_video_metadata(self, frames: torch.Tensor) -> dict:
        """Build metadata dict for a video."""
        return dict(
            total_num_frames=frames.shape[0],
            fps=self.model_fps,
            frames_indices=list(range(frames.shape[0])),
        )

    def _build_exemplar_messages(
        self, exemplars: list[dict]
    ) -> tuple[list[dict], list[VideoWithMetadata]]:
        """Build message turns and video list for exemplars.

        Args:
            exemplars: List of exemplar dicts, each with 'video' and 'label_str'.

        Returns:
            Tuple of (messages, videos) for the exemplar turns.
        """
        messages: list[dict] = []
        videos: list[VideoWithMetadata] = []

        for exemplar in exemplars:
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
            videos.append(
                VideoWithMetadata(
                    frames=exemplar["video"],
                    metadata=self._build_video_metadata(exemplar["video"]),
                )
            )

        return messages, videos

    def build(
        self,
        target_video: torch.Tensor,
        exemplars: list[dict] | None = None,
    ) -> ConversationData:
        """Build conversation data for a target video.

        Args:
            target_video: Target video frames
            exemplars: Per-query exemplar dicts (None for zero-shot)

        Returns:
            ConversationData with messages and videos
        """
        messages: list[dict] = []
        videos: list[VideoWithMetadata] = []

        # System message
        if self._system_msg is not None:
            messages.append(self._system_msg)

        # Exemplar turns
        if exemplars:
            ex_messages, ex_videos = self._build_exemplar_messages(exemplars)
            messages.extend(ex_messages)
            videos.extend(ex_videos)

        # Target turn
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": target_video},
                    {"type": "text", "text": self._user_prompt},
                ],
            }
        )
        target_with_meta = VideoWithMetadata(
            frames=target_video,
            metadata=self._build_video_metadata(target_video),
        )
        videos.append(target_with_meta)

        return ConversationData(messages=messages, videos=videos)

    def build_vllm_inputs(
        self,
        target_video: torch.Tensor,
        processor,
        exemplars: list[dict] | None = None,
    ) -> dict:
        """Build ready-to-use vLLM inputs for a target video.

        Args:
            target_video: Target video frames
            processor: AutoProcessor instance
            exemplars: Per-query exemplar dicts (None for zero-shot)

        Returns:
            Dict ready for llm.generate()
        """
        conv_data = self.build(target_video, exemplars=exemplars)

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
        """Format message content for logging, replacing video tensors with shape info."""
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
        lines = ["Conversation structure:"]

        idx = 0
        if self._system_msg is not None:
            lines.append(f"  [{idx}] system: ...")
            idx += 1

        if self.config.num_shots > 0:
            lines.append(
                f"  [{idx}..{idx + 2 * self.config.num_shots - 1}] "
                f"{self.config.num_shots} exemplar turns (dynamic per query)"
            )
            idx += 2 * self.config.num_shots

        target_prompt_preview = (
            self._user_prompt[:200] + "..." if len(self._user_prompt) > 200 else self._user_prompt
        )
        lines.append(f"  [{idx}] user: <video: [target]> {target_prompt_preview}")

        logger.info("\n".join(lines))

    @property
    def num_videos(self) -> int:
        """Number of videos per request (for vLLM limit config)."""
        return self.config.num_shots + 1

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

    Exemplar sampling is handled externally by the inference loop; this
    factory only builds the conversation template.

    Args:
        config: Validated inference configuration
        label2idx: Label to index mapping

    Returns:
        Initialized ConversationBuilder
    """
    prompt_config = config.prompt.model_copy(update={"labels": list(label2idx.keys())})

    return ConversationBuilder(
        config=prompt_config,
        label2idx=label2idx,
        model_fps=config.model_fps,
        needs_video_metadata=config.model.needs_video_metadata,
    )
