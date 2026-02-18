"""Prompt text components for building activity recognition prompts."""

import textwrap
from collections.abc import Callable

from falldet.schemas import DefinitionsVariant, LabelsVariant, RoleVariant, TaskVariant

# Role description (expert persona)
ROLE_COMPONENT = textwrap.dedent("""
    Role:
    You are an expert Human Activity Recognition (HAR) specialist.
""").strip()

ROLE_COMPONENT_SPECIALIZED = textwrap.dedent("""
    Role:
    You are an expert Human Activity Recognition (HAR) specialist focused on fall detection and post-fall assessment.
""").strip()

ROLE_COMPONENT_VIDEO_SPECIALIZED = textwrap.dedent("""
    Role:
    You are an expert video analyst specializing in Human Activity Recognition (HAR) focused on fall detection and post-fall assessment.
""").strip()


TASK_INSTRUCTION = textwrap.dedent("""
    Task:
    Analyze the video clip and classify the primary action being performed. Assign exactly one label from the allowed list below.
""").strip()

TASK_INSTRUCTION_EXTENDED = textwrap.dedent("""
    Task:
    You are given a short video clip depicting one or several human subjects. Your task is to analyze the clip and classify the primary action being performed by the main subject. Assign exactly one label from the allowed list below, focusing on fall detection and post-fall assessment. Carefully consider the context, body posture, movement dynamics, and any environmental cues present in the video to make an accurate classification.
""").strip()

LABELS_COMPONENT = textwrap.dedent("""
    Allowed Labels:
    * Core: walk, fall, fallen, sit_down, sitting, lie_down, lying, stand_up, standing, other
    * Extended (Rare): kneel_down, kneeling, squat_down, squatting, crawl, jump
""").strip()

# from OPUS 4.6.
DEFINITIONS_COMPONENT_EXTENDED = textwrap.dedent("""
    Definitions & Decision Rules:

    1. FALL DETECTION (highest priority — classify these first):
    - fall: Rapid, uncontrolled descent to a lower position. Key indicators: sudden loss of balance, accelerating downward motion, no bracing or controlled lowering, body hits ground/furniture abruptly. Duration is typically < 1-2 seconds.
    - fallen: Person is on the ground AND context suggests a prior fall (not an intentional action). Key indicators: awkward/unnatural body position, limbs splayed asymmetrically, person appears unable or struggling to get up, position is inconsistent with intentional resting or relaxation. If a person is on the ground and the posture looks unintentional or distressed, prefer "fallen" over "lying" or "sitting".

    2. DISTINGUISHING FALLEN vs. LYING vs. SITTING (critical distinction):
    - fallen: On the ground after an uncontrolled event. Look for: unusual location (middle of hallway, next to overturned furniture), awkward posture (twisted limbs, face-down in unnatural position), signs of distress or immobility, no pillow/blanket/mat.
    - lying: Intentionally reclining in a controlled manner. Look for: relaxed posture, natural environment for lying (bed, couch, mat), symmetric body alignment, calm demeanor.
    - lie_down: The active transition from upright to a lying position, performed deliberately and smoothly.
    - sitting: Intentionally seated on a surface designed for sitting (chair, bench, floor with controlled posture), stable and relaxed.
    - sit_down: The active, controlled transition from standing/lying to a seated position.

    3. STANDING & TRANSITIONS:
    - standing: Upright posture, stationary or with minimal weight shifting. No significant transition occurring.
    - stand_up: Active transition from a lower position (sitting, lying, kneeling) to upright. The clip captures the rising motion.

    4. LOCOMOTION & DYNAMIC ACTIONS:
    - walk: Forward/backward movement with alternating foot contact on the ground.
    - crawl: Movement on hands and knees (or hands and feet), torso roughly parallel to ground.
    - jump: Both feet intentionally leave the ground simultaneously.
    - other: Any action not covered above (e.g., running, reaching, turning in place, bending to pick something up).

    5. LOW POSTURES (intentional & controlled):
    - kneel_down / kneeling: One or both knees on the ground, torso upright. Transition (kneel_down) vs. sustained posture (kneeling).
    - squat_down / squatting: Feet on the ground, knees deeply bent, hips lowered. Transition (squat_down) vs. sustained posture (squatting).
    """).strip()

DEFINITIONS_COMPONENT = textwrap.dedent("""
    Definitions:
    - Fall vs. Lie/Sit:
        - fall: Uncontrolled, rapid descent (accidental).
        - lie_down / sit_down: Intentional, controlled descent.
    - Fallen: A person who has already experienced a fall and is now on the ground, regardless of their current posture (e.g., lying, sitting, kneeling).
    - Stand_up vs. Standing:
        - stand_up: Transition from e.g. sitting to standing.
        - standing: Upright posture without recent transition.
    - 'other' is for actions that do not fit the above categories, such as running etc.
    - Jump: A person who is intentionally airborne, with both feet off the ground, regardless of the height or distance of the jump.
    - kneeling, squatting, crawling: A person who is intentionally close to the ground, with one or both knees on the ground (kneeling), a person who is intentionally in a low squatting position (squatting), or a person who is intentionally moving on hands and knees (crawling). Note that these actions are distinct from falling, as they are intentional and controlled movements.
    - Only one label should be assigned per video clip, even if multiple actions are present. Focus on the primary action.
    - When in doubt, choose the label that best fits the primary action being performed, even if it is not a perfect match.
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


# ============================================================================
# Label Formatter Functions
# ============================================================================


def _format_labels_bulleted(labels: list[str]) -> str:
    """Format labels as bulleted list (current default behavior)."""
    lines = ["Allowed Labels:"]
    for label in labels:
        lines.append(f"- {label}")
    return "\n".join(lines)


def _format_labels_comma(labels: list[str]) -> str:
    """Format labels as comma-separated list."""
    return f"Allowed Labels:\n{', '.join(labels)}"


def _format_labels_grouped(labels: list[str]) -> str:
    """Format labels with core/extended grouping."""
    # This assumes the default label structure - for custom labels, just show them all
    core_labels = [
        "walk",
        "fall",
        "fallen",
        "sit_down",
        "sitting",
        "lie_down",
        "lying",
        "stand_up",
        "standing",
        "other",
    ]
    extended_labels = ["kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump"]

    # Filter to only include labels that are in the provided list
    core_present = [label for label in core_labels if label in labels]
    extended_present = [label for label in extended_labels if label in labels]
    other_present = [
        label for label in labels if label not in core_labels and label not in extended_labels
    ]

    lines = ["Allowed Labels:"]
    if core_present:
        lines.append(f"* Core: {', '.join(core_present)}")
    if extended_present:
        lines.append(f"* Extended (Rare): {', '.join(extended_present)}")
    if other_present:
        lines.append(f"* Other: {', '.join(other_present)}")

    return "\n".join(lines)


def _format_labels_numbered(labels: list[str]) -> str:
    """Format labels as numbered list."""
    lines = ["Allowed Labels:"]
    for i, label in enumerate(labels, 1):
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


# ============================================================================
# Variant Registries
# ============================================================================

ROLE_VARIANTS: dict[str, str] = {
    RoleVariant.STANDARD: ROLE_COMPONENT,
    RoleVariant.SPECIALIZED: ROLE_COMPONENT_SPECIALIZED,
    RoleVariant.VIDEO_SPECIALIZED: ROLE_COMPONENT_VIDEO_SPECIALIZED,
}

TASK_VARIANTS: dict[str, str] = {
    TaskVariant.STANDARD: TASK_INSTRUCTION,
    TaskVariant.EXTENDED: TASK_INSTRUCTION_EXTENDED,
}

DEFINITIONS_VARIANTS: dict[str, str] = {
    DefinitionsVariant.STANDARD: DEFINITIONS_COMPONENT,
    DefinitionsVariant.EXTENDED: DEFINITIONS_COMPONENT_EXTENDED,
}

OUTPUT_FORMAT_VARIANTS: dict[str, str] = {
    "json": JSON_OUTPUT_FORMAT,
    "text": TEXT_OUTPUT_FORMAT,
}

LABEL_FORMAT_VARIANTS: dict[LabelsVariant, Callable[[list[str]], str]] = {
    LabelsVariant.BULLETED: _format_labels_bulleted,
    LabelsVariant.COMMA: _format_labels_comma,
    LabelsVariant.GROUPED: _format_labels_grouped,
    LabelsVariant.NUMBERED: _format_labels_numbered,
}
