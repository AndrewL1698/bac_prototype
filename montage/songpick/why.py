"""'Why this song' (phase 12.10): one plain sentence per suggested song from the director's LLM.

What is sent (CLAUDE.md section 11): the photos' mood word and what they show (zero-shot photo tags), and per song
its artist, track, tags and mood word (heard in its audio, or read from its tags). Never play counts, dates, ranks
or other listening data, never audio or images. The factual reasons from the ranking stay in `TrackCandidate.reason`;
this fills `TrackCandidate.why`. Any failure, no API key, or `songpick.why.enabled: false` leaves `why` unset and the
app shows the factual reason. Logs carry counts only.
"""

from __future__ import annotations

import json

from montage.config import get_config
from montage.log import get_logger
from montage.songpick import vibe
from montage.types import SongPick, TrackCandidate

log = get_logger("songpick.why")

SYSTEM = """You write the line shown under each song a photo-montage app suggests: one sentence per song saying why \
it suits the user's photos. Use only the facts given: what the photos show and their mood, and each song's tags and \
mood (heard in its audio, or read from its tags). Never mention lyrics, the artist's life, release dates, popularity, \
or anything else not given. Speak to the user ("your photos"). At most {max_words} words per line, no preamble. \
Return one line per song, in the order given."""

SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
          "required": ["lines"], "additionalProperties": False}


def _song_mood(c: TrackCandidate) -> str:
    if c.valence is None or c.arousal is None:
        return ""
    word = vibe.mood_word(c.valence, c.arousal) or "neutral"
    return f"; mood: {word} ({'heard in its audio' if c.mood_source == 'audio' else 'from its tags'})"


def facts(pick: SongPick, photo_tags: list[str]) -> str:
    """The whole user message: only the fields listed in the module doc."""
    mood = (vibe.mood_word(pick.album_valence, pick.album_arousal) or "neutral"
            if pick.album_valence is not None and pick.album_arousal is not None else "unknown")
    lines = [f"Photos: mood {mood}" + (f"; they show {', '.join(photo_tags)}" if photo_tags else "") + ".", "Songs:"]
    for i, c in enumerate(pick.candidates, 1):
        tags = f"; tags: {', '.join(c.tags[:get_config().songpick.vibe.top_tags])}" if c.tags else ""
        lines.append(f"{i}. {c.artist} - {c.track}{tags}{_song_mood(c)}")
    return "\n".join(lines)


def _ask(text: str, n: int) -> list[str]:
    from montage.director.llm import LLMFailure, _text, get_client
    cfg, d = get_config().songpick.why, get_config().director
    resp = get_client().messages.create(
        model=d.model, max_tokens=cfg.max_tokens, system=SYSTEM.format(max_words=cfg.max_words),
        messages=[{"role": "user", "content": text}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"}, timeout=d.timeout_s)
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise LLMFailure(f"stop_reason={resp.stop_reason}")
    lines = [str(x).strip() for x in json.loads(_text(resp)).get("lines") or []]
    if len(lines) != n or not all(lines):
        raise LLMFailure(f"got {len(lines)} lines for {n} songs")
    return lines


def explain(pick: SongPick, photo_tags: list[str] | None = None) -> bool:
    """Fill `why` for every shortlisted song, in place. True if the LLM wrote them."""
    from montage.director import llm_enabled
    if not get_config().songpick.why.enabled or not pick.candidates or not llm_enabled():
        return False
    try:
        lines = _ask(facts(pick, photo_tags or []), len(pick.candidates))
    except Exception as e:  # noqa: BLE001  optional component: the factual reasons stay
        log.warning("songpick: why-this-song failed (%s); showing the ranking reasons", type(e).__name__)
        return False
    for c, line in zip(pick.candidates, lines):
        c.why = line
    log.info("songpick: why-this-song lines for %d songs", len(lines))
    return True
