"""Tests for the ConversationBuilder class."""

import pytest
import torch

from falldet.inference.conversation import (
    ConversationBuilder,
    ConversationData,
    VideoWithMetadata,
)
from falldet.inference.prompts.components import FEWSHOT_SHORT_INSTRUCTION
from falldet.inference.prompts.parsers import JSONOutputParser, KeywordOutputParser
from falldet.schemas import FewshotPreamble, FewshotResponse, PromptConfig

# Test data
LABEL2IDX = {
    "walk": 0,
    "fall": 1,
    "fallen": 2,
    "sit_down": 3,
    "sitting": 4,
    "other": 5,
}


def create_mock_video(num_frames: int = 16) -> torch.Tensor:
    """Create a mock video tensor."""
    return torch.randn(num_frames, 3, 224, 224)


def create_mock_exemplars(num_exemplars: int) -> list[dict]:
    """Create mock exemplars for testing."""
    labels = ["walk", "fall", "sitting", "fallen", "sit_down"]
    return [
        {
            "video": create_mock_video(),
            "label_str": labels[i % len(labels)],
            "label": i % len(labels),
        }
        for i in range(num_exemplars)
    ]


class MockProcessor:
    """Mock processor for testing."""

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Mock chat template application."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content_parts = []
            for item in msg["content"]:
                if item["type"] == "text":
                    content_parts.append(item["text"])
                elif item["type"] == "video":
                    content_parts.append("<video>")
            parts.append(f"[{role}]: {' '.join(content_parts)}")

        if add_generation_prompt:
            parts.append("[assistant]:")

        return "\n".join(parts)


class TestVideoWithMetadata:
    """Tests for VideoWithMetadata dataclass."""

    def test_creation(self):
        """Test creating VideoWithMetadata."""
        frames = create_mock_video(16)
        metadata = {"total_num_frames": 16, "fps": 8.0, "frames_indices": list(range(16))}
        video = VideoWithMetadata(frames=frames, metadata=metadata)

        assert video.frames is frames
        assert video.metadata == metadata

    def test_metadata_fields(self):
        """Test that metadata contains expected fields."""
        frames = create_mock_video(8)
        metadata = {"total_num_frames": 8, "fps": 4.0, "frames_indices": [0, 2, 4, 6]}
        video = VideoWithMetadata(frames=frames, metadata=metadata)

        assert video.metadata["total_num_frames"] == 8
        assert video.metadata["fps"] == 4.0
        assert len(video.metadata["frames_indices"]) == 4


class TestConversationData:
    """Tests for ConversationData dataclass."""

    def test_creation(self):
        """Test creating ConversationData."""
        messages = [{"role": "user", "content": []}]
        videos = [
            VideoWithMetadata(
                frames=create_mock_video(),
                metadata={"total_num_frames": 16, "fps": 8.0, "frames_indices": []},
            )
        ]
        data = ConversationData(messages=messages, videos=videos)

        assert data.messages == messages
        assert data.videos == videos


class TestConversationBuilder:
    """Tests for ConversationBuilder."""

    def test_zero_shot_messages_structure(self):
        """Test message structure with zero exemplars."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video)

        # Zero-shot should have only the target user message (no system for Qwen non-CoT)
        assert len(conv_data.messages) == 1
        assert conv_data.messages[0]["role"] == "user"

    def test_zero_shot_single_video(self):
        """Test that zero-shot has exactly one video."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video)

        assert len(conv_data.videos) == 1
        assert conv_data.videos[0].frames is target_video

    def test_few_shot_messages_structure(self):
        """Test message structure: system + one user message per exemplar + target user."""
        num_exemplars = 2
        exemplars = create_mock_exemplars(num_exemplars)
        # Default: preamble=system, response=inline
        config = PromptConfig(num_shots=num_exemplars)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # system + exemplar1 user + exemplar2 user + target user
        assert len(conv_data.messages) == num_exemplars + 2
        assert conv_data.messages[0]["role"] == "system"
        for msg in conv_data.messages[1:]:
            assert msg["role"] == "user"

    def test_few_shot_system_message_contains_introduction(self):
        """Test that system message is present in few-shot and contains preamble."""
        exemplars = create_mock_exemplars(1)
        config = PromptConfig(num_shots=1, role_variant="standard")
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        assert conv_data.messages[0]["role"] == "system"
        system_text = conv_data.messages[0]["content"][0]["text"]
        assert "Role:" in system_text
        assert "Task:" in system_text
        assert "Allowed Labels:" in system_text

    def test_few_shot_video_count(self):
        """Test that few-shot has correct number of videos."""
        num_exemplars = 3
        exemplars = create_mock_exemplars(num_exemplars)
        config = PromptConfig(num_shots=num_exemplars)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        assert len(conv_data.videos) == num_exemplars + 1

    def test_video_metadata_structure(self):
        """Test that video metadata has correct structure."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX, model_fps=10.0)
        target_video = create_mock_video(24)

        conv_data = builder.build(target_video)

        video_meta = conv_data.videos[0].metadata
        assert video_meta["total_num_frames"] == 24
        assert video_meta["fps"] == 10.0
        assert video_meta["frames_indices"] == list(range(24))

    def test_build_vllm_inputs_format(self):
        """Test that build_vllm_inputs returns correct dict structure."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()
        processor = MockProcessor()

        inputs = builder.build_vllm_inputs(target_video, processor)

        assert "prompt" in inputs
        assert "multi_modal_data" in inputs
        assert "mm_processor_kwargs" in inputs

        assert isinstance(inputs["prompt"], str)
        assert "video" in inputs["multi_modal_data"]

    def test_build_vllm_inputs_with_exemplars(self):
        """Test that build_vllm_inputs works with per-query exemplars."""
        num_exemplars = 2
        exemplars = create_mock_exemplars(num_exemplars)
        config = PromptConfig(num_shots=num_exemplars)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()
        processor = MockProcessor()

        inputs = builder.build_vllm_inputs(target_video, processor, exemplars=exemplars)

        assert isinstance(inputs["prompt"], str)
        video_data = inputs["multi_modal_data"]["video"]
        assert len(video_data) == num_exemplars + 1

    def test_build_vllm_inputs_with_video_metadata(self):
        """Test that vLLM inputs include video metadata when required."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX, needs_video_metadata=True)
        target_video = create_mock_video()
        processor = MockProcessor()

        inputs = builder.build_vllm_inputs(target_video, processor)

        # Should be list of (frames, metadata) tuples
        video_data = inputs["multi_modal_data"]["video"]
        assert len(video_data) == 1
        assert isinstance(video_data[0], tuple)
        assert len(video_data[0]) == 2  # (frames, metadata)

    def test_build_vllm_inputs_without_video_metadata(self):
        """Test that vLLM inputs exclude metadata when not required."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX, needs_video_metadata=False)
        target_video = create_mock_video()
        processor = MockProcessor()

        inputs = builder.build_vllm_inputs(target_video, processor)

        # Should be list of just frames tensors
        video_data = inputs["multi_modal_data"]["video"]
        assert len(video_data) == 1
        assert isinstance(video_data[0], torch.Tensor)

    def test_num_videos_property_zero_shot(self):
        """Test num_videos property for zero-shot."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX)

        assert builder.num_videos == 1

    def test_num_videos_property_few_shot(self):
        """Test num_videos property for few-shot."""
        num_exemplars = 4
        config = PromptConfig(num_shots=num_exemplars)
        builder = ConversationBuilder(config, LABEL2IDX)

        assert builder.num_videos == num_exemplars + 1

    def test_format_answer_json(self):
        """Test JSON format answer for exemplars."""
        config = PromptConfig(output_format="json")
        builder = ConversationBuilder(config, LABEL2IDX)

        answer = builder._format_answer("fall")
        assert answer == '{"label": "fall"}'

    def test_format_answer_text(self):
        """Test text format answer for exemplars."""
        config = PromptConfig(output_format="text")
        builder = ConversationBuilder(config, LABEL2IDX)

        answer = builder._format_answer("walk")
        assert answer == "The best answer is: walk"

    def test_parser_property_json(self):
        """Test parser property returns correct type for JSON."""
        config = PromptConfig(output_format="json", cot=False)
        builder = ConversationBuilder(config, LABEL2IDX)

        parser = builder.parser
        assert isinstance(parser, JSONOutputParser)

    def test_parser_property_text(self):
        """Test parser property returns correct type for text."""
        config = PromptConfig(output_format="text", cot=False)
        builder = ConversationBuilder(config, LABEL2IDX)

        parser = builder.parser
        assert isinstance(parser, KeywordOutputParser)

    def test_user_prompt_property(self):
        """Test user_prompt property returns the prompt text."""
        config = PromptConfig()
        builder = ConversationBuilder(config, LABEL2IDX)

        prompt = builder.user_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_different_exemplars_produce_different_conversations(self):
        """Test that passing different exemplars produces different output."""
        config = PromptConfig(num_shots=2)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()
        processor = MockProcessor()

        exemplars_a = create_mock_exemplars(2)
        exemplars_b = create_mock_exemplars(2)
        # Make labels different
        exemplars_b[0]["label_str"] = "other"

        inputs_a = builder.build_vllm_inputs(target_video, processor, exemplars=exemplars_a)
        inputs_b = builder.build_vllm_inputs(target_video, processor, exemplars=exemplars_b)

        assert inputs_a["prompt"] != inputs_b["prompt"]

    def test_exemplar_user_message_contains_prompt(self):
        """Test that FEWSHOT_SHORT_INSTRUCTION appears in the exemplar user message."""
        exemplars = create_mock_exemplars(1)
        config = PromptConfig(num_shots=1)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # With default (system, inline): system + exemplar user + target user
        exemplar_msg = conv_data.messages[1]
        text_items = [item["text"] for item in exemplar_msg["content"] if item["type"] == "text"]
        combined = "\n".join(text_items)
        assert FEWSHOT_SHORT_INSTRUCTION in combined

    def test_exemplar_inline_contains_response_delimiter_and_answer(self):
        """Test that inline user message contains [RESPONSE 1] and the formatted answer."""
        exemplars = create_mock_exemplars(1)
        exemplars[0]["label_str"] = "fall"
        config = PromptConfig(num_shots=1, output_format="json")
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # With default (system, inline): exemplar is at index 1
        exemplar_msg = conv_data.messages[1]
        text_items = [item["text"] for item in exemplar_msg["content"] if item["type"] == "text"]
        combined = "\n".join(text_items)
        assert "[RESPONSE 1]" in combined
        assert '"label": "fall"' in combined

    def test_fewshot_section_markers(self):
        """Test that numbered [REQUEST N], [RESPONSE N], and [QUERY] markers appear."""
        exemplars = create_mock_exemplars(2)
        config = PromptConfig(num_shots=2)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        all_text = "\n".join(
            item["text"]
            for msg in conv_data.messages
            for item in msg["content"]
            if item.get("type") == "text"
        )
        assert "[REQUEST 1]" in all_text
        assert "[RESPONSE 1]" in all_text
        assert "[REQUEST 2]" in all_text
        assert "[RESPONSE 2]" in all_text
        assert "[QUERY]" in all_text

    def test_target_message_has_full_prompt(self):
        """Test that target message contains full prompt (not exemplar prompt)."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video)

        target_msg = conv_data.messages[-1]
        content = target_msg["content"]

        # Should have video and text
        assert content[0]["type"] == "video"
        assert content[1]["type"] == "text"

        # Should contain full prompt components
        prompt_text = content[1]["text"]
        assert "Role:" in prompt_text or "Allowed Labels:" in prompt_text

    def test_none_exemplars_treated_as_zero_shot(self):
        """Test that None exemplars produce zero-shot conversation."""
        config = PromptConfig(num_shots=0)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=None)

        # Zero-shot: just the target message
        assert len(conv_data.messages) == 1
        assert len(conv_data.videos) == 1

    def test_system_instruction_produces_system_message(self):
        """Test that system_instruction adds a system message to the conversation."""
        config = PromptConfig(
            num_shots=0,
            system_instruction="Represent the user's input",
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video)

        # Should have: system + user = 2 messages
        assert len(conv_data.messages) == 2
        assert conv_data.messages[0]["role"] == "system"
        assert conv_data.messages[0]["content"][0]["text"] == "Represent the user's input"
        assert conv_data.messages[1]["role"] == "user"

    def test_system_instruction_with_few_shot(self):
        """Test that system_instruction overrides the auto-generated preamble in few-shot."""
        num_exemplars = 1
        exemplars = create_mock_exemplars(num_exemplars)
        config = PromptConfig(
            num_shots=num_exemplars,
            system_instruction="Custom system instruction",
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # system + exemplar user + target user
        assert len(conv_data.messages) == 3
        assert conv_data.messages[0]["role"] == "system"
        assert conv_data.messages[0]["content"][0]["text"] == "Custom system instruction"


# ---------------------------------------------------------------------------
# Parametrized tests for fewshot_preamble × fewshot_response
# ---------------------------------------------------------------------------

_NUM_SHOTS = 2

# (preamble, response, expected_roles) with _NUM_SHOTS=2
_FORMAT_ROLES = [
    (
        FewshotPreamble.SYSTEM,
        FewshotResponse.INLINE,
        # system preamble + exemplar1 user + exemplar2 user + target user
        ["system", "user", "user", "user"],
    ),
    (
        FewshotPreamble.USER,
        FewshotResponse.INLINE,
        # user preamble + exemplar1 user + exemplar2 user + target user
        ["user", "user", "user", "user"],
    ),
    (
        FewshotPreamble.SYSTEM,
        FewshotResponse.ASSISTANT,
        # system preamble + (user + assistant) * 2 + target user
        ["system", "user", "assistant", "user", "assistant", "user"],
    ),
    (
        FewshotPreamble.USER,
        FewshotResponse.ASSISTANT,
        # user preamble + (user + assistant) * 2 + target user
        ["user", "user", "assistant", "user", "assistant", "user"],
    ),
]


class TestFewshotFormats:
    """Parametrized tests for all fewshot_preamble × fewshot_response combinations."""

    @pytest.mark.parametrize("preamble,response,expected_roles", _FORMAT_ROLES)
    def test_role_sequence(
        self,
        preamble: FewshotPreamble,
        response: FewshotResponse,
        expected_roles: list[str],
    ):
        """Message roles must match the expected sequence for each combination."""
        config = PromptConfig(
            num_shots=_NUM_SHOTS, fewshot_preamble=preamble, fewshot_response=response
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        exemplars = create_mock_exemplars(_NUM_SHOTS)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        roles = [msg["role"] for msg in conv_data.messages]
        assert roles == expected_roles

    @pytest.mark.parametrize("preamble,response,_", _FORMAT_ROLES)
    def test_video_count(self, preamble: FewshotPreamble, response: FewshotResponse, _):
        """Total videos must equal num_shots + 1 for every combination."""
        config = PromptConfig(
            num_shots=_NUM_SHOTS, fewshot_preamble=preamble, fewshot_response=response
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        exemplars = create_mock_exemplars(_NUM_SHOTS)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        assert len(conv_data.videos) == _NUM_SHOTS + 1

    def test_user_preamble_no_system_message(self):
        """fewshot_preamble='user' must not produce a system message."""
        for response in FewshotResponse:
            config = PromptConfig(
                num_shots=_NUM_SHOTS,
                fewshot_preamble=FewshotPreamble.USER,
                fewshot_response=response,
            )
            builder = ConversationBuilder(config, LABEL2IDX)
            conv_data = builder.build(
                create_mock_video(), exemplars=create_mock_exemplars(_NUM_SHOTS)
            )
            roles = [msg["role"] for msg in conv_data.messages]
            assert "system" not in roles

    def test_system_preamble_has_system_message(self):
        """fewshot_preamble='system' must produce exactly one system message."""
        for response in FewshotResponse:
            config = PromptConfig(
                num_shots=_NUM_SHOTS,
                fewshot_preamble=FewshotPreamble.SYSTEM,
                fewshot_response=response,
            )
            builder = ConversationBuilder(config, LABEL2IDX)
            conv_data = builder.build(
                create_mock_video(), exemplars=create_mock_exemplars(_NUM_SHOTS)
            )
            system_msgs = [msg for msg in conv_data.messages if msg["role"] == "system"]
            assert len(system_msgs) == 1

    def test_assistant_response_has_assistant_turns(self):
        """fewshot_response='assistant' must produce one assistant turn per exemplar."""
        for preamble in FewshotPreamble:
            config = PromptConfig(
                num_shots=_NUM_SHOTS,
                fewshot_preamble=preamble,
                fewshot_response=FewshotResponse.ASSISTANT,
            )
            builder = ConversationBuilder(config, LABEL2IDX)
            conv_data = builder.build(
                create_mock_video(), exemplars=create_mock_exemplars(_NUM_SHOTS)
            )
            asst_msgs = [msg for msg in conv_data.messages if msg["role"] == "assistant"]
            assert len(asst_msgs) == _NUM_SHOTS

    def test_inline_response_no_assistant_turns(self):
        """fewshot_response='inline' must not produce any assistant turns."""
        for preamble in FewshotPreamble:
            config = PromptConfig(
                num_shots=_NUM_SHOTS,
                fewshot_preamble=preamble,
                fewshot_response=FewshotResponse.INLINE,
            )
            builder = ConversationBuilder(config, LABEL2IDX)
            conv_data = builder.build(
                create_mock_video(), exemplars=create_mock_exemplars(_NUM_SHOTS)
            )
            asst_msgs = [msg for msg in conv_data.messages if msg["role"] == "assistant"]
            assert len(asst_msgs) == 0

    def test_numbered_request_delimiters_inline(self):
        """Inline turns must use numbered [REQUEST N] and [RESPONSE N] delimiters."""
        config = PromptConfig(
            num_shots=_NUM_SHOTS,
            fewshot_preamble=FewshotPreamble.SYSTEM,
            fewshot_response=FewshotResponse.INLINE,
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        conv_data = builder.build(create_mock_video(), exemplars=create_mock_exemplars(_NUM_SHOTS))

        all_text = "\n".join(
            item["text"]
            for msg in conv_data.messages
            for item in msg["content"]
            if item.get("type") == "text"
        )
        for n in range(1, _NUM_SHOTS + 1):
            assert f"[REQUEST {n}]" in all_text
            assert f"[RESPONSE {n}]" in all_text
        assert "[QUERY]" in all_text

    def test_demonstration_delimiters_in_assistant_variant(self):
        """Assistant variant must use [DEMONSTRATION N] in user turns and bare answers in assistant turns."""
        config = PromptConfig(
            num_shots=_NUM_SHOTS,
            fewshot_preamble=FewshotPreamble.SYSTEM,
            fewshot_response=FewshotResponse.ASSISTANT,
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        conv_data = builder.build(create_mock_video(), exemplars=create_mock_exemplars(_NUM_SHOTS))

        # Check [DEMONSTRATION N] in exemplar user turns
        user_msgs = [msg for msg in conv_data.messages if msg["role"] == "user"]
        for i in range(_NUM_SHOTS):
            n = i + 1
            all_text = " ".join(
                item["text"] for item in user_msgs[i]["content"] if item["type"] == "text"
            )
            assert f"[DEMONSTRATION {n}]" in all_text

        # Assistant turns must not contain any bracket delimiters
        for msg in conv_data.messages:
            if msg["role"] == "assistant":
                text = msg["content"][0]["text"]
                assert "[" not in text

    @pytest.mark.parametrize("preamble,response,_", _FORMAT_ROLES)
    def test_build_vllm_inputs_returns_valid_dict(
        self, preamble: FewshotPreamble, response: FewshotResponse, _
    ):
        """build_vllm_inputs must return the expected keys with correct video count."""
        config = PromptConfig(
            num_shots=_NUM_SHOTS, fewshot_preamble=preamble, fewshot_response=response
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        processor = MockProcessor()

        inputs = builder.build_vllm_inputs(
            create_mock_video(), processor, exemplars=create_mock_exemplars(_NUM_SHOTS)
        )

        assert "prompt" in inputs
        assert "multi_modal_data" in inputs
        assert "mm_processor_kwargs" in inputs
        assert isinstance(inputs["prompt"], str)
        assert len(inputs["multi_modal_data"]["video"]) == _NUM_SHOTS + 1

    @pytest.mark.parametrize("preamble,response,_", _FORMAT_ROLES)
    def test_videos_always_in_user_turns(
        self, preamble: FewshotPreamble, response: FewshotResponse, _
    ):
        """All videos in exemplar turns must be in user messages (never system)."""
        config = PromptConfig(
            num_shots=_NUM_SHOTS, fewshot_preamble=preamble, fewshot_response=response
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        conv_data = builder.build(create_mock_video(), exemplars=create_mock_exemplars(_NUM_SHOTS))

        for msg in conv_data.messages:
            if msg["role"] == "system":
                video_items = [item for item in msg["content"] if item["type"] == "video"]
                assert len(video_items) == 0, "System messages must not contain videos"
