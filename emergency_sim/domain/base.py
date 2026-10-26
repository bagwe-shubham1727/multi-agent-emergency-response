"""Pydantic base classes for domain value objects and LLM-facing DTOs."""

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable value object; unknown fields rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AgentIOModel(BaseModel):
    """LLM-facing DTO.

    Output subclasses must be strict-schema compatible: no dict fields, no
    defaults, no numeric/length constraints. Ranges are checked in semantic
    validation by the coordinator instead.
    """

    model_config = ConfigDict(extra="forbid")
