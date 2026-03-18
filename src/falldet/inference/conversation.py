"""Conversation builder for zero-shot and few-shot inference."""

import logging
from dataclasses import dataclass

import torch

from falldet.schemas import InferenceConfig, PromptConfig

from .prompts import PromptBuilder
from .prompts.components import (
    DEMONSTRATION_DELIMITER,
    FEWSHOT_SHORT_INSTRUCTION,
    QUERY_DELIMITER,
    REQUEST_DELIMITER,
    RESPONSE_DELIMITER,
)

logger = logging.getLogger(__name__)

_MAX_LOG_TEXT_LEN = 2000


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
        self._sample_logged = False

        self._system_msg: dict | None
        if config.num_shots > 0:
            self._preamble: str = self._prompt_builder.build_fewshot_system_instruction()
            # System message: explicit override takes priority, otherwise auto-generated preamble
            system_text = (
                config.system_instruction
                if config.system_instruction is not None
                else self._preamble
            )
            self._system_msg = {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            }
        else:
            self._user_prompt: str = self._prompt_builder.build_prompt()
            self._system_msg = self._prompt_builder.get_system_message()

        logger.info(f"ConversationBuilder initialized: {self.num_videos} videos/request")
        self._log_conversation()

    def _build_video_metadata(self, frames: torch.Tensor) -> dict:
        """Build metadata dict for a video."""
        n = frames.shape[0]
        return dict(total_num_frames=n, fps=self.model_fps, frames_indices=list(range(n)))

    def _make_video(self, frames: torch.Tensor) -> VideoWithMetadata:
        """Wrap frames with computed metadata."""
        return VideoWithMetadata(frames=frames, metadata=self._build_video_metadata(frames))

    def _build_fewshot_messages(
        self, exemplars: list[dict], target_video: torch.Tensor
    ) -> tuple[list[dict], list[VideoWithMetadata]]:
        """Build a single user message containing all exemplars and the target query.

        Structure:
          [system: preamble]
          [user: [DEMONSTRATIONS] [REQUEST] <video> prompt [RESPONSE] answer
                 [REQUEST] <video> prompt [RESPONSE] answer  (repeated for each exemplar)
                 [QUERY] [REQUEST] <video> prompt]           (target)

        Args:
            exemplars: List of exemplar dicts, each with 'video' and 'label_str'.
            target_video: Target video frames.

        Returns:
            Tuple of (messages, videos).
        """
        content: list[dict] = []
        videos: list[VideoWithMetadata] = []

        for i, exemplar in enumerate(exemplars):
            answer = self._format_answer(exemplar["label_str"])
            if i == 0:
                prefix = f"{DEMONSTRATION_DELIMITER}\n{REQUEST_DELIMITER}\n"
            else:
                prefix = f"{REQUEST_DELIMITER}\n"
            content.append({"type": "text", "text": prefix})
            content.append({"type": "video", "video": exemplar["video"]})
            content.append(
                {
                    "type": "text",
                    "text": f"\n{FEWSHOT_SHORT_INSTRUCTION}\n{RESPONSE_DELIMITER}\n{answer}\n\n",
                }
            )
            videos.append(self._make_video(exemplar["video"]))

        # Target query
        content.append({"type": "text", "text": f"{QUERY_DELIMITER}\n{REQUEST_DELIMITER}\n"})
        content.append({"type": "video", "video": target_video})
        content.append({"type": "text", "text": f"\n{FEWSHOT_SHORT_INSTRUCTION}"})
        videos.append(self._make_video(target_video))

        messages = [{"role": "user", "content": content}]
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

        if self._system_msg is not None:
            messages.append(self._system_msg)

        if exemplars is None:
            exemplars = []

        if self.config.num_shots > 0:
            if len(exemplars) != self.config.num_shots:
                raise ValueError(
                    f"Expected {self.config.num_shots} exemplars, but got {len(exemplars)}"
                )
            fewshot_messages, fewshot_videos = self._build_fewshot_messages(exemplars, target_video)
            messages.extend(fewshot_messages)
            videos.extend(fewshot_videos)

        else:
            # Zero-shot target turn
            target_content = [
                {"type": "video", "video": target_video},
                {"type": "text", "text": self._user_prompt},
            ]
            messages.append({"role": "user", "content": target_content})
            videos.append(self._make_video(target_video))

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

        if not self._sample_logged:
            logger.debug("First formatted prompt (chat template applied):\n%s", text)
            self._sample_logged = True

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

    @staticmethod
    def _truncate_preview(text: str) -> str:
        """Truncate text for logging, appending ellipsis if trimmed."""
        return text[:_MAX_LOG_TEXT_LEN] + "..." if len(text) > _MAX_LOG_TEXT_LEN else text

    def _log_conversation(self) -> None:
        """Log the conversation structure at initialization."""
        lines = ["Conversation structure:"]

        idx = 0
        if self._system_msg is not None:
            for content_item in self._system_msg["content"]:
                if content_item["type"] == "text":
                    lines.append(
                        f"  [{idx}] system: {self._truncate_preview(content_item['text'])}"
                    )
                    idx += 1

        if self.config.num_shots > 0:
            lines.append(
                f"  [{idx}] user: 1 message with {self.config.num_shots} exemplar(s) + "
                f"target (dynamic per query)"
            )
        else:
            lines.append(
                f"  [{idx}] user: <video: [target]> {self._truncate_preview(self._user_prompt)}"
            )

        logger.info("\n".join(lines))

    @property
    def num_videos(self) -> int:
        """Number of videos per request (for vLLM limit config)."""
        return self.config.num_shots + 1

    @property
    def user_prompt(self) -> str:
        """Get the user prompt text."""
        if self.config.num_shots > 0:
            return self._preamble
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
