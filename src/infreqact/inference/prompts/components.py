"""Prompt text components for building activity recognition prompts."""

import textwrap

# Role description (expert persona)
ROLE_COMPONENT = textwrap.dedent("""
    Role:
    You are an expert Human Activity Recognition (HAR) specialist.
""").strip()

# Task instruction (always included)
TASK_INSTRUCTION = textwrap.dedent("""
    Task:
    Analyze the video clip and classify the primary action being performed. Only use one of the allowed labels provided below.
""").strip()

# Allowed labels
LABELS_COMPONENT = textwrap.dedent("""
    Allowed Labels:
    * Core: walk, fall, fallen, sit_down, sitting, lie_down, lying, stand_up, standing, other
    * Extended (Rare): kneel_down, kneeling, squat_down, squatting, crawl, jump
""").strip()

# Label definitions and constraints
DEFINITIONS_COMPONENT = textwrap.dedent("""
    Definitions & Constraints:
    - Fall vs. Lie/Sit:
        - fall: Uncontrolled, rapid descent (accidental).
        - lie_down / sit_down: Intentional, controlled descent.
    - Dynamic vs. Static:
        - Dynamic (Actions): e.g. walk, fall: subject is moving.
        - Static (States): e.g. fallen, sitting: subject remains still.
""").strip()

# Chain-of-thought instruction
COT_INSTRUCTION = textwrap.dedent("""
    Please reason step-by-step, identify relevant visual content,
    analyze key timestamps and clues, and then provide the final answer.
""").strip()

# Output format instructions
JSON_OUTPUT_FORMAT = textwrap.dedent("""
    Output Format:
    Return a strictly valid JSON object where <class_label> is one of the allowed labels.
    {
      "label": "<class_label>"
    }
""").strip()

TEXT_OUTPUT_FORMAT = textwrap.dedent("""
    Output Format:
    Respond with 'The best answer is: <class_label>' where <class_label> is one of the allowed labels.
""").strip()

# Model-specific components
# R1 system prompt for CoT reasoning (thinking mode)
R1_SYSTEM_PROMPT = textwrap.dedent("""
    You are a helpful assistant. Before providing your final answer, conduct a detailed analysis of the question. Enclose your entire thinking process within <think> and </think> tags. After your analysis, provide your final answer separately.
""").strip()

# Few-shot exemplar prompt (used for exemplar user turns)
EXEMPLAR_USER_PROMPT = "Classify the action shown in this video."
