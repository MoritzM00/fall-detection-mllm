"""Prompt text components for building activity recognition prompts."""

import textwrap

# Role description
ROLE_COMPONENT = textwrap.dedent("""
    Role:
    You are an expert Human Activity Recognition (HAR) specialist. Analyze the video clip to classify the action or posture of the main subject.
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
    * Walk vs. Other: walk includes jogging, drunk walking, and carrying small items. Pushing large objects (chairs, carts) must be labeled other.
    * Fall vs. Lie/Sit:
        * fall: Uncontrolled, rapid descent (accidental).
        * lie_down / sit_down: Intentional, controlled descent.
    * Dynamic vs. Static:
        * Dynamic (Actions): walk, fall, sit_down, lie_down, stand_up. Label starts at first frame of motion.
        * Static (States): fallen, sitting, lying, standing. Label starts when subject comes to complete rest.
""").strip()

# Sequence rules
CONSTRAINTS_COMPONENT = textwrap.dedent("""
    Sequence Rules:
    * Moment of Rest: If a person transitions Sit -> Lie without pausing, use only the destination label.
    * Fall Termination: fall ends when inertia stops. fallen begins only when the person is on the ground in a resting state.
""").strip()

# Chain-of-thought instruction
COT_INSTRUCTION = textwrap.dedent("""
    Please reason step-by-step, identify relevant visual content,
    analyze key timestamps and clues, and then provide the final answer.
""").strip()

# Output format instructions
JSON_OUTPUT_FORMAT = textwrap.dedent("""
    Output Format:
    Return a JSON object:
    {
      "label": "<class_label>"
    }
""").strip()

TEXT_OUTPUT_FORMAT = textwrap.dedent("""
    Output Format:
    State only the label from the allowed labels list.
""").strip()

# Final instruction for strict adherence
ADHERENCE_INSTRUCTION = textwrap.dedent("""
    Only use the allowed labels and stick exactly to the output format. If uncertain between two labels,
    choose the one that best fits the definitions above. Don't invent new labels.
""").strip()

# Model-specific components
# InternVL R1 system prompt for CoT reasoning (thinking mode)
INTERNVL_R1_SYSTEM_PROMPT = textwrap.dedent("""
    You are a helpful assistant. Before providing your final answer, conduct a detailed analysis of the question. Enclose your entire thinking process within <think> and </think> tags. After your analysis, provide your final answer separately.
""").strip()
