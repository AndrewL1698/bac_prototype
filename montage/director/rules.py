"""Rule-based director: brief -> EditPlan, deterministic and offline (always available)."""

from __future__ import annotations

import re

from montage.config import get_config
from montage.types import EditPlan

SLOWER = ("slow", "chill", "calm", "gentle", "relaxed", "mellow", "cinematic")
FASTER = ("fast", "hype", "energetic", "quick", "intense", "upbeat", "punchy")
CHRONO = ("chronological", "in order", "timeline", "as it happened")
FILLER = {"anything", "any", "all", "the", "of", "shots", "shot", "photos", "photo", "clips", "clip", "videos", "video"}
_PUNCT = r"[,.;:!?\n]"
AVOID_RE = re.compile(r"\b(?:skip|avoid|without|don't use|dont use|do not use|no)\b\s+([^,.;:!?\n]+)", re.I)
OPEN_RE = re.compile(r"\b(?:open(?:ing)? (?:on|with)|start (?:on|with)|begin (?:on|with))\s+([^,.;:!?\n]+)", re.I)
CLOSE_RE = re.compile(r"\b(?:end(?:ing)? (?:on|with)|finish (?:on|with)|close (?:on|with))\s+([^,.;:!?\n]+)", re.I)
LEN_RE = re.compile(r"\b(\d{2,3})\s*(?:s\b|sec\b|secs\b|seconds?\b)", re.I)


def _has(text: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in words)


def _avoid_query(phrase: str) -> str:
    p = phrase.strip().lower()
    if "blurry" in p or "blurred" in p or "out of focus" in p:
        return "blurry photo"
    if "screenshot" in p:
        return "screenshot"
    words = [w for w in p.split() if w not in FILLER]
    return " ".join(words) or p


def _cap_words(q: str) -> str:
    return " ".join(q.split()[: get_config().director.max_query_words])


def plan_from_rules(brief: str) -> EditPlan:
    cfg = get_config()
    brief = (brief or "").strip()
    text = brief.lower()
    pacing = (-0.5 if _has(text, SLOWER) else 0.0) + (0.5 if _has(text, FASTER) else 0.0)
    avoid = []
    for m in AVOID_RE.finditer(brief):
        q = _avoid_query(m.group(1))
        if q and q not in avoid:
            avoid.append(q)
    opening = (OPEN_RE.search(brief).group(1).strip() if OPEN_RE.search(brief) else "")
    closing = (CLOSE_RE.search(brief).group(1).strip() if CLOSE_RE.search(brief) else "")
    m = LEN_RE.search(brief)
    target = float(min(cfg.music.excerpt.max_length_s, max(cfg.music.excerpt.min_length_s, int(m.group(1))))) if m else 0.0

    # Vibe query: the brief minus directive clauses (avoid / open / close / length), so CLIP is not
    # pulled toward things the director asked to skip.
    vibe_text = brief
    for rx in (AVOID_RE, LEN_RE):
        vibe_text = rx.sub("", vibe_text)
    vibe_text = re.sub(r"\s*,\s*(,\s*)+", ", ", vibe_text).strip(" ,.;")
    queries = [_cap_words(vibe_text)] if vibe_text else []
    from montage.features.tags import vocabulary
    for tag in vocabulary():
        if tag not in ("blurry photo", "dark photo", "screenshot", "text document", "meme", "receipt") \
                and re.search(rf"\b{re.escape(tag)}\b", text) and tag not in queries and tag not in avoid:
            queries.append(tag)
    queries = queries[: cfg.director.max_queries]
    return EditPlan(
        vibe=brief[:200], vibe_queries=queries, avoid_queries=avoid[: cfg.director.max_queries],
        pacing=max(-1.0, min(1.0, pacing)), energy_bias=0.0, chronological=_has(text, CHRONO),
        target_length_s=target, excerpt_hint="auto", excerpt_start_s=0.0,
        opening_query=_cap_words(opening), closing_query=_cap_words(closing), segments=[], must_include=[],
        exclude=[], rationale="Rule-based plan (no LLM).", source="rules",
    )
