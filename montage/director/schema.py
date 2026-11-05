"""Structured-output schema for the LLM: EditPlan without `source` (set by code); every field required."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from montage.types import SegmentDirective


class EditPlanLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vibe: str
    vibe_queries: list[str]
    avoid_queries: list[str]
    pacing: float
    energy_bias: float
    chronological: bool
    target_length_s: float
    excerpt_hint: Literal["auto", "start", "peak", "custom"]
    excerpt_start_s: float
    opening_query: str
    closing_query: str
    segments: list[SegmentDirective]
    must_include: list[str]
    exclude: list[str]
    rationale: str


ENUM_FIELDS = ("excerpt_hint",)
SEGMENT_ENUM_FIELDS = ("anchor", "mood")


def json_schema() -> dict:
    """Schema for the configured provider: Anthropic's transformed schema, or plain JSON Schema for an
    OpenAI-compatible server (vLLM / Ollama constrain decoding to it)."""
    from montage.config import get_config
    if get_config().director.provider == "openai_compat":
        return EditPlanLLM.model_json_schema()
    import anthropic
    return anthropic.transform_schema(EditPlanLLM)
