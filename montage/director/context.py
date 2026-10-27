"""Text context for the director (only text goes to the API by default)."""

from __future__ import annotations

from collections import Counter

from montage.music.profile import mmss, summarize
from montage.types import EDL, Asset, Candidate, MusicProfile


def song_summary(profile: MusicProfile) -> str:
    return summarize(profile)


def media_inventory(assets: list[Asset], candidates: list[Candidate], tags: dict[str, list[str]],
                    max_lines: int = 100) -> str:
    """One line per asset: `a012 video 0:24 2026-07-14 | beach, ocean, ...` (video tags = most frequent)."""
    by_asset: dict[str, list[str]] = {}
    for c in candidates:
        by_asset.setdefault(c.asset_id, []).extend(tags.get(c.id, []))
    lines = []
    for a in assets[:max_lines]:
        t = [w for w, _ in Counter(by_asset.get(a.id, [])).most_common(5)]
        dur = f" {mmss(a.duration)}" if a.kind == "video" and a.duration else ""
        date = f" {a.taken_at:%Y-%m-%d}" if a.taken_at else ""
        lines.append(f"{a.id} {a.kind}{dur}{date} | {', '.join(t)}")
    if len(assets) > max_lines:
        lines.append(f"... and {len(assets) - max_lines} more")
    return "\n".join(lines)


def edl_summary(edl: EDL, tags: dict[str, list[str]] | None = None) -> str:
    tags = tags or {}
    return "\n".join(f"{mmss(s.start)}-{mmss(s.end)} {s.asset_id} {s.kind} | {', '.join(tags.get(s.candidate_id, [])[:3])}"
                     for s in edl.shots)


def allowed_queries_hint() -> str:
    from montage.features.tags import vocabulary
    return ", ".join(vocabulary())
