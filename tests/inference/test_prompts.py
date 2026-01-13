"""Tests for the prompts module."""

from infreqact.inference.prompts import (
    CoTOutputParser,
    JSONOutputParser,
    KeywordOutputParser,
    PromptBuilder,
    PromptConfig,
)

# Test data
LABEL2IDX = {
    "walk": 0,
    "fall": 1,
    "fallen": 2,
    "sit_down": 3,
    "sitting": 4,
    "lie_down": 5,
    "lying": 6,
    "stand_up": 7,
    "standing": 8,
    "other": 9,
}


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()
        assert config.output_format == "json"
        assert config.include_role is True
        assert config.include_definitions is True
        assert config.cot is False
        assert config.cot_start_tag == "<think>"
        assert config.cot_end_tag == "</think>"
        assert config.model_family == "qwen"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PromptConfig(
            output_format="text",
            include_role=False,
            cot=True,
            model_family="InternVL",
        )
        assert config.output_format == "text"
        assert config.include_role is False
        assert config.cot is True
        assert config.model_family == "InternVL"


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_default_prompt(self):
        """Test building default prompt with all components."""
        config = PromptConfig()
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        # Check that all expected components are present
        assert "Role:" in prompt
        assert "Allowed Labels:" in prompt
        assert "Definitions & Constraints:" in prompt
        assert "Output Format:" in prompt
        assert '"label": "<class_label>"' in prompt

    def test_baseline_prompt(self):
        """Test building baseline prompt without role and definitions."""
        config = PromptConfig(
            include_role=False,
            include_definitions=False,
        )
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        # Check that role and definitions are excluded
        assert "Role:" not in prompt
        assert "Definitions & Constraints:" not in prompt
        assert "Sequence Rules:" not in prompt

        # But labels and output format should still be present
        assert "Allowed Labels:" in prompt
        assert "Output Format:" in prompt

    def test_cot_prompt(self):
        """Test building CoT prompt with reasoning instruction."""
        config = PromptConfig(cot=True)
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        # Check that CoT instruction is present
        assert "Please reason step-by-step" in prompt
        assert "identify relevant visual content" in prompt

    def test_text_output_format(self):
        """Test building prompt with text output format."""
        config = PromptConfig(output_format="text")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        # Check that text output format is used
        assert "The best answer is:" in prompt
        assert '{"label"' not in prompt

    def test_no_system_prefix_in_prompt_without_cot(self):
        """Test that no system-level prompts are in user prompt without CoT."""
        config = PromptConfig(cot=False, model_family="InternVL")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        # Check that no system-level instructions are in the prompt
        assert "You are a helpful assistant" not in prompt
        assert "<think>" not in prompt

    def test_get_parser_json(self):
        """Test getting JSON parser."""
        config = PromptConfig(output_format="json", cot=False)
        builder = PromptBuilder(config, LABEL2IDX)
        parser = builder.get_parser()

        assert isinstance(parser, JSONOutputParser)

    def test_get_parser_keyword(self):
        """Test getting keyword parser."""
        config = PromptConfig(output_format="text", cot=False)
        builder = PromptBuilder(config, LABEL2IDX)
        parser = builder.get_parser()

        assert isinstance(parser, KeywordOutputParser)

    def test_get_parser_cot(self):
        """Test getting CoT parser."""
        config = PromptConfig(output_format="json", cot=True)
        builder = PromptBuilder(config, LABEL2IDX)
        parser = builder.get_parser()

        assert isinstance(parser, CoTOutputParser)

    def test_get_system_message_internvl_cot(self):
        """Test getting system message for InternVL with CoT."""
        config = PromptConfig(cot=True, model_family="InternVL")
        builder = PromptBuilder(config, LABEL2IDX)
        system_msg = builder.get_system_message()

        assert system_msg is not None
        assert system_msg["role"] == "system"
        assert len(system_msg["content"]) == 1
        assert system_msg["content"][0]["type"] == "text"
        assert "think" in system_msg["content"][0]["text"].lower()

    def test_get_system_message_qwen_cot(self):
        """Test that Qwen CoT doesn't need system message."""
        config = PromptConfig(cot=True, model_family="Qwen")
        builder = PromptBuilder(config, LABEL2IDX)
        system_msg = builder.get_system_message()

        assert system_msg is None

    def test_get_system_message_no_cot(self):
        """Test that system message is None when CoT is disabled."""
        config = PromptConfig(cot=False, model_family="InternVL")
        builder = PromptBuilder(config, LABEL2IDX)
        system_msg = builder.get_system_message()

        assert system_msg is None


class TestJSONOutputParser:
    """Tests for JSONOutputParser."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON output."""
        parser = JSONOutputParser(LABEL2IDX)
        text = '{"label": "fall"}'
        result = parser.parse(text)

        assert result.label == "fall"
        assert result.reasoning is None
        assert result.raw_text == text

    def test_parse_invalid_label(self):
        """Test parsing JSON with invalid label."""
        parser = JSONOutputParser(LABEL2IDX)
        text = '{"label": "invalid_label"}'
        result = parser.parse(text)

        # Should default to "other"
        assert result.label == "other"

    def test_parse_malformed_json(self):
        """Test parsing malformed JSON."""
        parser = JSONOutputParser(LABEL2IDX)
        text = "not valid json"
        result = parser.parse(text)

        # Should default to "other"
        assert result.label == "other"

    def test_parse_missing_label(self):
        """Test parsing JSON without label field."""
        parser = JSONOutputParser(LABEL2IDX)
        text = '{"prediction": "fall"}'
        result = parser.parse(text)

        # Should default to "other" when label is missing
        assert result.label == "other"


class TestKeywordOutputParser:
    """Tests for KeywordOutputParser."""

    def test_parse_simple_label(self):
        """Test parsing text with simple label."""
        parser = KeywordOutputParser(LABEL2IDX)
        text = "The person is walking."
        result = parser.parse(text)

        assert result.label == "walk"
        assert result.raw_text == text

    def test_parse_label_at_start(self):
        """Test parsing text with label at start."""
        parser = KeywordOutputParser(LABEL2IDX)
        text = "fall is what I observed"
        result = parser.parse(text)

        assert result.label == "fall"

    def test_parse_compound_label(self):
        """Test parsing text with compound label (sit_down)."""
        parser = KeywordOutputParser(LABEL2IDX)
        text = "The person sits down on the chair."
        result = parser.parse(text)

        # Should match "sit_down" when phrase uses "sit down" or "sits down"
        assert result.label == "sit_down"

    def test_parse_no_match(self):
        """Test parsing text with no valid label."""
        parser = KeywordOutputParser(LABEL2IDX)
        text = "I don't know what this is"
        result = parser.parse(text)

        # Should default to "other"
        assert result.label == "other"

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        parser = KeywordOutputParser(LABEL2IDX)
        text = "The person is WALKING."
        result = parser.parse(text)

        assert result.label == "walk"

    def test_parse_multiple_labels(self):
        """Test parsing text with multiple possible labels."""
        parser = KeywordOutputParser(LABEL2IDX)
        text = "walk\nfall"
        result = parser.parse(text)

        # Should match the first label found ("walk")
        assert result.label == "walk"

    def test_parse_exact_match(self):
        """Test parsing various labels using parameterization."""
        parser = KeywordOutputParser(LABEL2IDX)
        for label in LABEL2IDX:
            result = parser.parse(label)
            assert result.label == label


class TestCoTOutputParser:
    """Tests for CoTOutputParser."""

    def test_parse_with_think_tags(self):
        """Test parsing CoT output with think tags."""
        base_parser = JSONOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        text = """<think>
        The person appears to lose balance and falls rapidly.
        This is clearly an uncontrolled descent.
        </think>
        {"label": "fall"}
        """

        result = parser.parse(text)

        assert result.label == "fall"
        assert "lose balance" in result.reasoning
        assert result.raw_text == text

    def test_parse_without_tags(self):
        """Test parsing CoT output without tags (fallback)."""
        base_parser = JSONOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        text = '{"label": "walk"}'
        result = parser.parse(text)

        # Should parse entire text as answer
        assert result.label == "walk"
        assert result.reasoning is None

    def test_parse_only_end_tag(self):
        """Test parsing when tokenizer added start tag (Qwen behavior)."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        # Simulate Qwen output where <think> is added by tokenizer
        # Model only generates: "reasoning...</think>answer"
        text = "The person appears to be walking steadily forward.</think>walk"
        result = parser.parse(text)

        assert result.label == "walk"
        assert result.reasoning is not None
        assert "walking steadily" in result.reasoning
        assert "<think>" not in result.reasoning  # Start tag not in reasoning

    def test_parse_only_end_tag_with_json(self):
        """Test parsing only end tag with JSON output."""
        base_parser = JSONOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        text = 'The video shows a person falling down.</think>{"label": "fall"}'
        result = parser.parse(text)

        assert result.label == "fall"
        assert result.reasoning is not None
        assert "falling down" in result.reasoning

    def test_parse_with_keyword_base_parser(self):
        """Test CoT parser wrapping keyword parser."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        text = """<think>
        The person is moving forward steadily.
        </think>
        walk
        """

        result = parser.parse(text)

        assert result.label == "walk"
        assert "moving forward" in result.reasoning

    def test_custom_tags(self):
        """Test CoT parser with custom tags."""
        base_parser = JSONOutputParser(LABEL2IDX)
        parser = CoTOutputParser(
            LABEL2IDX, base_parser, start_tag="<reasoning>", end_tag="</reasoning>"
        )

        text = """<reasoning>
        Analyzing the video.
        </reasoning>
        {"label": "sitting"}
        """

        result = parser.parse(text)

        assert result.label == "sitting"
        assert "Analyzing" in result.reasoning

    def test_extract_reasoning(self):
        """Test extract_reasoning method."""
        base_parser = JSONOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        text = "<think>reasoning here</think>content here"
        reasoning, content = parser.extract_reasoning(text)

        assert reasoning == "reasoning here"
        assert content == "content here"

    def test_extract_reasoning_no_end_tag(self):
        """Test extract_reasoning when end tag is missing."""
        base_parser = JSONOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        text = "<think>reasoning here"
        reasoning, content = parser.extract_reasoning(text)

        assert reasoning == "reasoning here"
        assert content is None

    def test_extract_reasoning_no_start_tag(self):
        """Test extract_reasoning when start tag is missing."""
        base_parser = JSONOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        text = "just content here"
        reasoning, content = parser.extract_reasoning(text)

        assert reasoning is None
        assert content == "just content here"
