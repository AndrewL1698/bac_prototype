"""Director: the user's brief/notes -> EditPlan (LLM when available, rules otherwise)."""

from __future__ import annotations

import os

from montage.config import get_config
from montage.director.rules import plan_from_rules
from montage.log import get_logger
from montage.types import EditPlan

log = get_logger("director")


def llm_enabled() -> bool:
    """Claude needs an API key; an open-weight model only needs a server URL (if it is down, calls fail
    fast and the rules take over)."""
    d = get_config().director
    if not d.use_llm:
        return False
    if d.provider == "openai_compat":
        return bool(d.openai_compat.base_url)
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_label() -> str:
    """Short name of the active director for the UI and logs."""
    d = get_config().director
    if not llm_enabled():
        return "rules"
    return d.openai_compat.model if d.provider == "openai_compat" else "Claude"


def make_plan(brief: str, ctx: dict | None = None) -> tuple[EditPlan, list[str]]:
    """Returns (plan, fallbacks). Any LLM failure degrades to the rule-based plan."""
    if llm_enabled():
        try:
            from montage.director.llm import plan_from_llm
            return plan_from_llm(brief, ctx or {}), []
        except Exception as e:  # noqa: BLE001  optional component
            log.warning("LLM director failed (%s); using rules", e)
    return plan_from_rules(brief), ["director:rules"]
