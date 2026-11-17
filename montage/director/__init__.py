"""Director: the user's brief/notes -> EditPlan (LLM when available, rules otherwise)."""

from __future__ import annotations

import os
import urllib.request

from montage.config import get_config
from montage.director.rules import plan_from_rules
from montage.log import get_logger
from montage.types import EditPlan

log = get_logger("director")


def server_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=2):
            return True
    except OSError:
        return False


def llm_enabled() -> bool:
    """The open-weight model needs its server to answer (start it with scripts/serve_llm.py); Claude needs an
    API key. Otherwise the rules direct."""
    d = get_config().director
    if not d.use_llm:
        return False
    if d.provider == "openai_compat":
        return server_up(d.openai_compat.base_url)
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
            return keep_explicit_avoids(plan_from_llm(brief, ctx or {}), plan_from_rules(brief)), []
        except Exception as e:  # noqa: BLE001  optional component
            log.warning("LLM director failed (%s); using rules", e)
    return plan_from_rules(brief), ["director:rules"]


def keep_explicit_avoids(plan: EditPlan, rules: EditPlan) -> EditPlan:
    """The literal "no X" phrases the rules parser finds always reach the plan's avoid list: models sometimes
    drop them (Qwen3-8B missed "no people" in the director comparison)."""
    plan.avoid_queries = list(dict.fromkeys(rules.avoid_queries + plan.avoid_queries))[: get_config().director.max_queries]
    return plan
