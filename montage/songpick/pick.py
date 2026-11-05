"""Shortlist songs for an album from the user's listening data (phase 12).

Sources, each scored on its own and max-normalized, then summed per track with `w_sources`:
- export: the full Spotify data export (optional; the only source with dates for old photos).
  Period = capture dates padded by `pad_days`; widened by `widen_factor` up to `max_pad_days` when
  fewer than `min_window_events` plays fall inside; else the whole history with a recency decay.
- recent: Spotify sign-in "recently played" (last 50 plays, with dates), scored inside the padded period.
- top: Spotify sign-in top tracks (about 4 weeks / 6 months / 1 year, ranks only). A list counts when
  its span overlaps the period; narrower lists weigh more (`spotify.range_weight`). Photos older than
  every list fall back to the 1-year list with a note.
No network, no weights: the caller fetches the data (montage.songpick.spotify).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

from montage.config import get_config
from montage.songpick.affinity import in_period, score_tracks, track_key
from montage.songpick.period import album_span, padded
from montage.types import ListeningEvent, SongPick, TopTrack, TrackCandidate, UserError

log = logging.getLogger(__name__)

RANGE_LABEL = {"short_term": "the last 4 weeks", "medium_term": "the last 6 months", "long_term": "the last year"}


def choose_period(events: list[ListeningEvent], taken_at: Iterable[datetime | None]
                  ) -> tuple[tuple[datetime, datetime] | None, str, list[str]]:
    """Period for the export source: (period or None, period_source, notes)."""
    cfg = get_config().songpick
    span = album_span(taken_at)
    if span is None:
        return None, "history", ["The photos have no capture dates, so songs are ranked by your overall "
                                 "listening (recent plays count more)."]
    pad = cfg.pad_days
    first = True
    while pad <= cfg.max_pad_days:
        period = padded(span, pad)
        n = sum(in_period(e, period) for e in events)
        if n >= cfg.min_window_events:
            notes = [] if first else [f"Few plays right around the photo dates; widened the period to "
                                      f"{pad:g} days either side."]
            return period, "album" if first else "album_widened", notes
        pad *= cfg.widen_factor
        first = False
    return None, "history", ["Your history has too few plays around the photo dates, so songs are ranked "
                             "by your overall listening (recent plays count more)."]


def _export_candidates(events: list[ListeningEvent], taken_at: list[datetime | None]):
    period, source, notes = choose_period(events, taken_at)
    cands = score_tracks(events, period)
    if not cands and period is not None:  # many songs, each played once: accept single plays
        cands = score_tracks(events, period, min_plays=1)
        if cands:
            notes.append("No song was played more than once in the period; single plays included.")
    if not cands and period is not None:
        period, source = None, "history"
        notes.append("No full plays in the period, so songs are ranked by your overall listening.")
        cands = score_tracks(events, None, min_plays=1)
    for c in cands:
        c.sources = ["export"]
    n_win = len(events) if period is None else sum(in_period(e, period) for e in events)
    return cands, period, source, notes, n_win


def _recent_candidates(recent: list[ListeningEvent], period: tuple[datetime, datetime] | None):
    cands = score_tracks(recent, period, min_plays=1)
    for c in cands:
        c.sources = ["recent"]
        if c.reason.startswith("You played it"):
            c.reason = c.reason.replace("You played it", "You recently played it", 1)
    n_win = len(recent) if period is None else sum(in_period(e, period) for e in recent)
    return cands, n_win


def ranges_for_period(period: tuple[datetime, datetime] | None, available: Iterable[str],
                      now: datetime) -> tuple[list[str], bool]:
    """Top-track lists whose span (ending now) overlaps the period; (ranges, photos_older_than_all)."""
    days = get_config().songpick.spotify.range_days
    avail = [r for r in RANGE_LABEL if r in set(available)]
    if period is None:
        return avail, False
    hit = [r for r in avail if period[1] >= now - timedelta(days=days[r]) and period[0] <= now]
    if not hit and "long_term" in avail:
        return ["long_term"], True
    return hit, False


def _top_candidates(top: dict[str, list[TopTrack]], ranges: list[str]) -> list[TrackCandidate]:
    weights = get_config().songpick.spotify.range_weight
    best: dict[str, tuple[float, TopTrack]] = {}
    seen: dict[str, list[str]] = {}
    for r in ranges:
        lst = top.get(r) or []
        for t in lst:
            v = (len(lst) + 1 - t.rank) / len(lst) * weights[r]
            k = track_key(t.artist, t.track)
            seen.setdefault(k, []).append(r)
            if k not in best or v > best[k][0]:
                best[k] = (v, t)
    if not best:
        return []
    top_v = max(v for v, _ in best.values())
    out = []
    for k, (v, t) in best.items():
        aff = round(v / top_v, 6)
        out.append(TrackCandidate(
            key=k, track=t.track, artist=t.artist, album=t.album, uri=t.uri, plays_window=0.0, plays_total=0.0,
            n_plays_window=0, n_skips_window=0, lift=1.0, affinity=aff, score=aff, duration_ms=t.duration_ms,
            reason=f"#{t.rank} in your top songs of {RANGE_LABEL[t.time_range]}.",
            sources=[f"top:{r}" for r in seen[k]]))
    out.sort(key=lambda c: (-c.affinity, c.key))
    return out


def merge_sources(groups: list[tuple[str, list[TrackCandidate]]]) -> list[TrackCandidate]:
    """Sum per-source affinities per track (weights `w_sources`), re-normalize, keep every reason."""
    w = get_config().songpick.w_sources
    merged: dict[str, TrackCandidate] = {}
    for name, cands in groups:
        for c in cands:
            add = w.get(name, 1.0) * c.affinity
            m = merged.get(c.key)
            if m is None:
                m = merged[c.key] = c.model_copy(deep=True)
                m.affinity = add
                continue
            m.affinity += add
            # export and recent can list the same plays, so counts take the max instead of a sum
            m.plays_window = max(m.plays_window, c.plays_window)
            m.plays_total = max(m.plays_total, c.plays_total)
            m.n_plays_window = max(m.n_plays_window, c.n_plays_window)
            m.n_skips_window = max(m.n_skips_window, c.n_skips_window)
            m.lift = max(m.lift, c.lift)
            m.uri = m.uri or c.uri
            m.album = m.album or c.album
            m.duration_ms = m.duration_ms or c.duration_ms
            m.sources += [s for s in c.sources if s not in m.sources]
            m.reason = f"{m.reason} {c.reason}"
    out = list(merged.values())
    top = max((c.affinity for c in out), default=0.0)
    for c in out:
        c.affinity = round(c.affinity / top, 6) if top > 0 else 0.0
        c.score = c.affinity
    return out


def suggest(taken_at: Iterable[datetime | None], *, export: list[ListeningEvent] | None = None,
            recent: list[ListeningEvent] | None = None, top: dict[str, list[TopTrack]] | None = None,
            now: datetime | None = None, k: int | None = None) -> SongPick:
    """Best `k` (default `shortlist_size`) songs for an album from whatever listening data is given."""
    cfg = get_config().songpick
    k = cfg.shortlist_size if k is None else k
    if export is not None and not export:
        raise UserError("The listening history is empty.")
    if not export and not recent and not any((top or {}).values()):
        raise UserError("Connect Spotify (or upload your Spotify history) to get song suggestions.")
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    taken_at = list(taken_at)
    span = album_span(taken_at)
    album_period = padded(span, cfg.pad_days) if span else None
    groups: list[tuple[str, list[TrackCandidate]]] = []
    notes: list[str] = []
    sources: list[str] = []
    period, period_source = album_period, "album" if span else "history"
    n_events = n_win = 0

    if export:
        cands, period, period_source, ex_notes, ex_win = _export_candidates(export, taken_at)
        groups.append(("export", cands))
        notes += ex_notes
        sources.append("export")
        n_events, n_win = len(export), ex_win
    elif span is None:
        notes.append("The photos have no capture dates, so the picks come from your top songs and recent plays.")
    if recent:
        cands, r_win = _recent_candidates(recent, album_period)
        if cands:
            groups.append(("recent", cands))
            sources.append("recent")
        n_events += len(recent)
        n_win += r_win
    if top and any(top.values()):
        ranges, too_old = ranges_for_period(album_period, [r for r, v in top.items() if v], now)
        if too_old and not export:
            notes.append("These photos are older than the year of listening Spotify sign-in shows, so the picks "
                         "are your top songs of the last year. Upload your full Spotify history (data export) "
                         "for songs from that time.")
        if ranges:
            groups.append(("top", _top_candidates(top, ranges)))
            sources += [f"top:{r}" for r in ranges]

    cands = merge_sources(groups)
    if not cands:
        raise UserError("No songs found in your listening data for these photos.")
    for c in cands:  # no vibe stage yet: vibe is None and score = affinity (alpha only applies with a vibe)
        if c.vibe is not None:
            c.score = round(cfg.alpha * c.affinity + (1 - cfg.alpha) * c.vibe, 6)
    cands.sort(key=lambda c: (-c.score, -c.n_plays_window, c.key))
    log.info("songpick: sources %s, period %s, %d of %d plays inside, %d candidates, shortlist %d",
             sources, period_source, n_win, n_events, len(cands), min(k, len(cands)))
    return SongPick(period_start=period[0] if period else None, period_end=period[1] if period else None,
                    period_source=period_source, n_events=n_events, n_events_window=n_win,
                    candidates=cands[:k], vibe_source="none", notes=notes, sources=sources)


def shortlist(events: list[ListeningEvent], taken_at: Iterable[datetime | None],
              k: int | None = None) -> SongPick:
    """Export-only shortlist (the data-export path)."""
    return suggest(taken_at, export=events, k=k)
