"""Revision notes -> updated EditPlan. Rules fallback here; the LLM path arrives in phase 9A."""

from __future__ import annotations

import re

from montage.config import get_config
from montage.types import EDL, EditPlan

FASTER = re.compile(r"\b(faster|quicker|more energy|speed (it )?up|too slow)\b", re.I)
SLOWER = re.compile(r"\b(slower|calmer|slow (it )?down|too fast|less frantic)\b", re.I)
MORE = re.compile(r"\bmore (?:of )?(?:the )?([^,.;:!?\n]+)", re.I)
LESS = re.compile(r"\b(?:less|no|lose|drop|remove|without) (?:of )?(?:the )?([^,.;:!?\n]+)", re.I)
TIME = re.compile(r"\bat (\d{1,2}):(\d{2})\b")
PARTS = {"beginning": "start", "start": "start", "intro": "start", "opening": "start", "middle": "middle",
         "end": "end", "ending": "end", "outro": "end"}


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _shot_at(edl: EDL, t: float):
    return next((s for s in edl.shots if s.start <= t < s.end), None)


def revise_rules(prev: EditPlan, edl: EDL | None, note: str) -> EditPlan:
    from montage.types import SegmentDirective
    cfg = get_config().director
    p = prev.model_copy(deep=True)
    text = note.strip()
    delta = 0.4 if FASTER.search(text) else -0.4 if SLOWER.search(text) else 0.0
    part = next((PARTS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in PARTS), None)
    tm = TIME.search(text)
    if tm and edl is not None and re.search(r"\b(lose|drop|remove|cut|no|skip|without)\b", text, re.I):
        shot = _shot_at(edl, int(tm.group(1)) * 60 + int(tm.group(2)))
        if shot and shot.asset_id not in p.exclude:
            p.exclude.append(shot.asset_id)
    elif delta:
        if part is None:
            p.pacing = _clamp(p.pacing + delta)
        else:
            anchor = {"start": "intro", "end": "outro", "middle": "build"}[part]
            seg = next((s for s in p.segments if s.anchor == anchor), None)
            if seg is None:
                p.segments.append(SegmentDirective(anchor=anchor, section_index=-1, content_queries=[],
                                                   mood="match_music", pacing=_clamp(p.pacing + delta)))
            else:
                seg.pacing = _clamp((seg.pacing or p.pacing) + delta)
            if part == "middle":  # middle = build + peak
                p.segments.append(SegmentDirective(anchor="peak", section_index=-1, content_queries=[],
                                                   mood="match_music", pacing=_clamp(p.pacing + delta)))
    if not tm:
        for m in MORE.finditer(text):
            q = " ".join(m.group(1).split()[: cfg.max_query_words])
            if q and q not in p.vibe_queries and not FASTER.search("more " + q):
                p.vibe_queries = (p.vibe_queries + [q])[-cfg.max_queries:]
        for m in LESS.finditer(text):
            q = " ".join(m.group(1).split()[: cfg.max_query_words])
            if q and q not in p.avoid_queries and not SLOWER.search(q):
                p.avoid_queries = (p.avoid_queries + [q])[-cfg.max_queries:]
    p.rationale = f"Revised by rules for the note: {text[:120]}"
    p.source = "rules"
    return p


def revise_plan(prev: EditPlan, edl: EDL | None, note: str, ctx: dict | None = None) -> tuple[EditPlan, list[str]]:
    """Returns (plan, fallbacks)."""
    from montage.director import llm_enabled
    if llm_enabled():
        try:
            from montage.director.llm import revise_from_llm
            return revise_from_llm(prev, edl, note, ctx or {}), []
        except Exception:  # noqa: BLE001  optional component
            pass
    return revise_rules(prev, edl, note), ["director:rules"]
