"""Tests for the ConversationBuilder class."""

import torch

from falldet.inference.conversation import (
    ConversationBuilder,
    ConversationData,
    VideoWithMetadata,
)
from falldet.inference.prompts.parsers import JSONOutputParser, KeywordOutputParser
from falldet.schemas import PromptConfig

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
        """Test message structure with exemplars passed at build time."""
        num_exemplars = 2
        exemplars = create_mock_exemplars(num_exemplars)
        config = PromptConfig(num_shots=num_exemplars)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # Should have: 2 user-assistant pairs + 1 target user = 5 messages
        assert len(conv_data.messages) == 5

        # Check message order: user, assistant, user, assistant, user
        assert conv_data.messages[0]["role"] == "user"
        assert conv_data.messages[1]["role"] == "assistant"
        assert conv_data.messages[2]["role"] == "user"
        assert conv_data.messages[3]["role"] == "assistant"
        assert conv_data.messages[4]["role"] == "user"

    def test_few_shot_message_order_with_system(self):
        """Test message order includes system message when needed (InternVL CoT)."""
        exemplars = create_mock_exemplars(1)
        config = PromptConfig(num_shots=1, cot=True, model_family="InternVL")
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # Should have: system, user, assistant, user = 4 messages
        assert len(conv_data.messages) == 4
        assert conv_data.messages[0]["role"] == "system"
        assert conv_data.messages[1]["role"] == "user"
        assert conv_data.messages[2]["role"] == "assistant"
        assert conv_data.messages[3]["role"] == "user"

    def test_few_shot_video_count(self):
        """Test that few-shot has correct number of videos."""
        num_exemplars = 3
        exemplars = create_mock_exemplars(num_exemplars)
        config = PromptConfig(num_shots=num_exemplars)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # Should have exemplar videos + target video
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

    def test_exemplar_user_message_content(self):
        """Test that exemplar user messages have correct content."""
        exemplars = create_mock_exemplars(1)
        config = PromptConfig(num_shots=1)
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        user_msg = conv_data.messages[0]
        assert user_msg["role"] == "user"

        content = user_msg["content"]
        assert len(content) == 2
        assert content[0]["type"] == "video"
        assert content[1]["type"] == "text"
        assert "Classify" in content[1]["text"]

    def test_exemplar_assistant_message_content(self):
        """Test that exemplar assistant messages have correct content."""
        exemplars = create_mock_exemplars(1)
        exemplars[0]["label_str"] = "fall"
        config = PromptConfig(num_shots=1, output_format="json")
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        asst_msg = conv_data.messages[1]
        assert asst_msg["role"] == "assistant"

        content = asst_msg["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert '"label": "fall"' in content[0]["text"]

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
        """Test that system_instruction works with few-shot exemplars."""
        exemplars = create_mock_exemplars(1)
        config = PromptConfig(
            num_shots=1,
            system_instruction="Custom system instruction",
        )
        builder = ConversationBuilder(config, LABEL2IDX)
        target_video = create_mock_video()

        conv_data = builder.build(target_video, exemplars=exemplars)

        # Should have: system, user, assistant, user = 4 messages
        assert len(conv_data.messages) == 4
        assert conv_data.messages[0]["role"] == "system"
        assert conv_data.messages[0]["content"][0]["text"] == "Custom system instruction"
        assert conv_data.messages[1]["role"] == "user"
        assert conv_data.messages[2]["role"] == "assistant"
        assert conv_data.messages[3]["role"] == "user"
