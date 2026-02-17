"""Configuration model for prompt building.

Re-exported from falldet.schemas where all config models are defined.
"""

from falldet.schemas import (
    DefinitionsVariant,
    LabelsVariant,
    PromptConfig,
    RoleVariant,
    TaskVariant,
)

__all__ = [
    "PromptConfig",
    "RoleVariant",
    "TaskVariant",
    "LabelsVariant",
    "DefinitionsVariant",
]
