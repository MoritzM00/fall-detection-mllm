"""Tests for the prompts module."""

import pytest
from pydantic import ValidationError

from falldet.inference.prompts import (
    CoTOutputParser,
    JSONOutputParser,
    KeywordOutputParser,
    PromptBuilder,
)
from falldet.schemas import PromptConfig

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
    """Tests for PromptConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()
        assert config.system_instruction is None
        assert config.output_format == "json"
        assert config.cot is False
        assert config.cot_start_tag == "<think>"
        assert config.cot_end_tag == "</think>"
        assert config.model_family == "qwen"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PromptConfig(
            output_format="text",
            cot=True,
            model_family="InternVL",
        )
        assert config.output_format == "text"
        assert config.cot is True
        assert config.model_family == "InternVL"

    def test_default_few_shot_fields(self):
        """Test default values for few-shot fields."""
        config = PromptConfig()
        assert config.num_shots == 0
        assert config.shot_selection == "balanced"
        assert config.exemplar_seed == 42

    def test_custom_few_shot_fields(self):
        """Test custom values for few-shot fields."""
        config = PromptConfig(
            num_shots=4,
            shot_selection="random",
            exemplar_seed=123,
        )
        assert config.num_shots == 4
        assert config.shot_selection == "random"
        assert config.exemplar_seed == 123

    def test_few_shot_fields_types(self):
        """Test that few-shot fields have correct types."""
        config = PromptConfig(num_shots=8, shot_selection="balanced", exemplar_seed=0)
        assert isinstance(config.num_shots, int)
        assert isinstance(config.shot_selection, str)
        assert isinstance(config.exemplar_seed, int)

    def test_default_variant_fields(self):
        """Test default values for variant selector fields."""
        config = PromptConfig()
        assert config.role_variant == "standard"
        assert config.task_variant == "standard"
        assert config.labels_variant == "bulleted"
        assert config.definitions_variant is None

    def test_custom_variant_fields(self):
        """Test custom values for variant selector fields."""
        config = PromptConfig(
            role_variant="specialized",
            task_variant="extended",
            labels_variant="comma",
            definitions_variant="extended",
        )
        assert config.role_variant == "specialized"
        assert config.task_variant == "extended"
        assert config.labels_variant == "comma"
        assert config.definitions_variant == "extended"

    def test_variant_fields_none_values(self):
        """Test that variant fields can be set to None."""
        config = PromptConfig(
            role_variant=None,
            definitions_variant=None,
        )
        assert config.role_variant is None
        assert config.definitions_variant is None

    # ========================================================================
    # Validation Tests (Pydantic)
    # ========================================================================

    def test_invalid_role_variant_raises_validation_error(self):
        """Test that invalid role_variant value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(role_variant="invalid")

        error = exc_info.value
        assert "role_variant" in str(error)
        # Check that error message lists allowed values
        assert "standard" in str(error) or "Input should be" in str(error)

    def test_invalid_output_format_raises_validation_error(self):
        """Test that invalid output_format value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(output_format="invalid")

        error = exc_info.value
        assert "output_format" in str(error)

    def test_invalid_task_variant_raises_validation_error(self):
        """Test that invalid task_variant value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(task_variant="invalid")

        error = exc_info.value
        assert "task_variant" in str(error)

    def test_invalid_labels_variant_raises_validation_error(self):
        """Test that invalid labels_variant value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(labels_variant="invalid")

        error = exc_info.value
        assert "labels_variant" in str(error)

    def test_invalid_definitions_variant_raises_validation_error(self):
        """Test that invalid definitions_variant value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(definitions_variant="invalid")

        error = exc_info.value
        assert "definitions_variant" in str(error)

    def test_invalid_shot_selection_raises_validation_error(self):
        """Test that invalid shot_selection value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(shot_selection="invalid")

        error = exc_info.value
        assert "shot_selection" in str(error)

    def test_unknown_field_raises_validation_error(self):
        """Test that unknown field raises ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(unknown_field="value")

        error = exc_info.value
        assert "unknown_field" in str(error)
        assert "Extra inputs are not permitted" in str(error)

    def test_invalid_type_raises_validation_error(self):
        """Test that invalid type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(cot=["invalid"])  # Should be bool, not list

        error = exc_info.value
        assert "cot" in str(error)

    def test_invalid_num_shots_type_raises_validation_error(self):
        """Test that invalid num_shots type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptConfig(num_shots="four")  # Should be int

        error = exc_info.value
        assert "num_shots" in str(error)


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_default_prompt(self):
        """Test building default prompt with standard components."""
        config = PromptConfig()
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        # Check that expected components are present (role yes, definitions no by default)
        assert "Role:" in prompt
        assert "Allowed Labels:" in prompt
        assert "Output Format:" in prompt
        assert '"label": "<class_label>"' in prompt
        # Definitions not included by default
        assert "Definitions:" not in prompt

    def test_baseline_prompt(self):
        """Test building baseline prompt without role and definitions."""
        config = PromptConfig(
            role_variant=None,
            definitions_variant=None,
        )
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        # Check that role and definitions are excluded
        assert "Role:" not in prompt
        assert "Definitions:" not in prompt

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

    def test_get_system_message_explicit_instruction(self):
        """Test that explicit system_instruction is returned as system message."""
        config = PromptConfig(system_instruction="Represent the user's input")
        builder = PromptBuilder(config, LABEL2IDX)
        system_msg = builder.get_system_message()

        assert system_msg is not None
        assert system_msg["role"] == "system"
        assert system_msg["content"][0]["text"] == "Represent the user's input"

    def test_system_instruction_takes_priority_over_internvl_r1(self):
        """Test that system_instruction takes priority over InternVL R1 auto-detect."""
        config = PromptConfig(
            system_instruction="Custom instruction",
            cot=True,
            model_family="InternVL",
        )
        builder = PromptBuilder(config, LABEL2IDX)
        system_msg = builder.get_system_message()

        assert system_msg is not None
        assert system_msg["content"][0]["text"] == "Custom instruction"
        # Should NOT contain the R1 system prompt
        assert "think" not in system_msg["content"][0]["text"].lower()

    def test_no_system_instruction_falls_back_to_internvl_r1(self):
        """Test that None system_instruction falls back to InternVL R1 when applicable."""
        config = PromptConfig(
            system_instruction=None,
            cot=True,
            model_family="InternVL",
        )
        builder = PromptBuilder(config, LABEL2IDX)
        system_msg = builder.get_system_message()

        assert system_msg is not None
        assert "think" in system_msg["content"][0]["text"].lower()

    # ========================================================================
    # Variant Selection Tests
    # ========================================================================

    def test_role_variant_standard(self):
        """Test standard role variant."""
        config = PromptConfig(role_variant="standard")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "You are an expert Human Activity Recognition (HAR) specialist." in prompt

    def test_role_variant_specialized(self):
        """Test specialized role variant."""
        config = PromptConfig(role_variant="specialized")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "fall detection and post-fall assessment" in prompt

    def test_role_variant_video_specialized(self):
        """Test video_specialized role variant."""
        config = PromptConfig(role_variant="video_specialized")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "video analyst specializing in" in prompt

    def test_role_variant_none(self):
        """Test that role_variant=None omits role section."""
        config = PromptConfig(role_variant=None)
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "Role:" not in prompt

    def test_task_variant_standard(self):
        """Test standard task variant."""
        config = PromptConfig(task_variant="standard")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "Analyze the video clip and classify the primary action" in prompt

    def test_task_variant_extended(self):
        """Test extended task variant."""
        config = PromptConfig(task_variant="extended")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "Carefully consider the context, body posture" in prompt

    def test_definitions_variant_standard(self):
        """Test standard definitions variant."""
        config = PromptConfig(definitions_variant="standard")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "Definitions & Decision Rules:" in prompt
        assert "FALL DETECTION" in prompt
        assert "DISTINGUISHING FALLEN vs. LYING vs. SITTING" in prompt

    def test_definitions_variant_extended(self):
        """Test extended definitions variant."""
        config = PromptConfig(definitions_variant="extended")
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "Definitions & Decision Rules:" in prompt
        assert "FALL DETECTION (highest priority" in prompt

    def test_definitions_variant_none(self):
        """Test that definitions_variant=None omits definitions section."""
        config = PromptConfig(definitions_variant=None)
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "Definitions:" not in prompt
        assert "Definitions & Decision Rules:" not in prompt

    def test_labels_variant_bulleted(self):
        """Test bulleted labels variant."""
        config = PromptConfig(labels_variant="bulleted", labels=["walk", "fall", "sit"])
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "Allowed Labels:" in prompt
        assert "- walk" in prompt
        assert "- fall" in prompt
        assert "- sit" in prompt

    def test_labels_variant_comma(self):
        """Test comma-separated labels variant."""
        config = PromptConfig(labels_variant="comma", labels=["walk", "fall", "sit"])
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "walk, fall, sit" in prompt

    def test_labels_variant_numbered(self):
        """Test numbered labels variant."""
        config = PromptConfig(labels_variant="numbered", labels=["walk", "fall", "sit"])
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "1. walk" in prompt
        assert "2. fall" in prompt
        assert "3. sit" in prompt

    def test_labels_variant_grouped(self):
        """Test grouped labels variant."""
        config = PromptConfig(
            labels_variant="grouped", labels=["walk", "fall", "sit_down", "kneel_down", "crawl"]
        )
        builder = PromptBuilder(config, LABEL2IDX)
        prompt = builder.build_prompt()

        assert "* Core:" in prompt
        assert "* Extended (Rare):" in prompt


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

    def test_extract_final_answer_best_answer_pattern(self):
        """Test extraction of 'The best answer is:' pattern (primary)."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        content = "Some analysis here mentioning standing and walking. The best answer is: fall"
        answer = parser.extract_final_answer(content)

        assert answer == "fall"

    def test_extract_final_answer_final_answer_pattern(self):
        """Test extraction of 'Final Answer:' pattern."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        content = "The person is not standing but falling. Final Answer: fall"
        answer = parser.extract_final_answer(content)

        assert answer == "fall"

    def test_extract_final_answer_the_answer_is_pattern(self):
        """Test extraction of 'The answer is' pattern."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        content = "Based on the analysis of standing posture, the answer is sitting"
        answer = parser.extract_final_answer(content)

        assert answer == "sitting"

    def test_extract_final_answer_case_insensitive(self):
        """Test that final answer extraction is case-insensitive."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        content = "Analysis complete. THE BEST ANSWER IS: walk"
        answer = parser.extract_final_answer(content)

        assert answer == "walk"

    def test_extract_final_answer_no_marker(self):
        """Test fallback when no answer marker is present."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        content = "The person is walking steadily"
        answer = parser.extract_final_answer(content)

        # Should return original content
        assert answer == content

    def test_parse_with_final_answer_marker(self):
        """Test full parsing with final answer marker extracts correct label."""
        base_parser = KeywordOutputParser(LABEL2IDX)
        parser = CoTOutputParser(LABEL2IDX, base_parser)

        # Simulates the problematic case: content mentions "standing" but final answer is "fall"
        text = """Reasoning about the video.</think>

To determine the primary action, we analyze:
- The person is not actively moving or standing
- They appear to have lost balance

The best answer is: fall"""

        result = parser.parse(text)

        assert result.label == "fall"
        assert "Reasoning about the video" in result.reasoning
