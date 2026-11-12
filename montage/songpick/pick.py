"""Shortlist songs for an album from the user's listening data (phase 12), in one of three modes.

Listening sources, each scored on its own and max-normalized, then summed per track with `w_sources`:
- export: the full Spotify data export (optional; the only source with play dates for old photos).
- recent: Spotify sign-in "recently played" (last 50 plays, with dates); counts inside the padded album period.
- top: Spotify sign-in top tracks (about 4 weeks / 6 months / 1 year, ranks only; `spotify.range_weight`).

Modes (`songpick.default_mode`, chosen in the app):
- release ("songs from back then"): songs from the past year of listening whose album release date falls in
  an age-scaled window around the photos' median date (`period.release_window`), plus songs actually played
  during the album (recent plays, or the 4-week list for photos that recent). The window widens by
  `widen_factor` until `release.min_candidates` songs qualify; if none ever do, it falls back to "vibe".
  Compilation/remaster dates are skipped (`release.reliable`). Ties go to the release closest to the photos.
- vibe ("best vibe fit"): dates ignored; the whole pool ranked mostly by vibe.
- export ("played back then"): the data export inside an age-scaled listening window
  (`period.listening_window`), widened up to `max_window_days`, else the whole history with a recency decay.
Ranking: score = alpha * affinity + (1 - alpha) * vibe (`vibe.alpha_by_mode`). The vibe comes from an optional
`vibe_scorer` hook (montage.songpick.vibe.Scorer: song tags vs the photos), run on the best `vibe.max_tracks`
songs by affinity; without it (or when no song has tags) the ranking is affinity alone.
No network, no weights here: the caller fetches the data (montage.songpick.spotify) and builds the scorer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from montage.config import get_config
from montage.songpick import release as rel
from montage.songpick.affinity import in_period, release_fields, score_tracks, track_key
from montage.songpick.period import (album_span, listening_window, median_date, padded, release_before_days,
                                     release_window)
from montage.types import ListeningEvent, SongPick, TopTrack, TrackCandidate, UserError

log = logging.getLogger(__name__)

MODES = ("release", "vibe", "export")
RANGE_LABEL = {"short_term": "the last 4 weeks", "medium_term": "the last 6 months", "long_term": "the last year"}
NO_DATES = "The photos have no capture dates, so songs are ranked by your overall listening."
VIBE_OFF = "Couldn't compare songs with your photos' mood, so these are the songs you listen to most."
NO_TAGS = "Found no genre or mood tags for these songs, so they are ranked by your listening."
VibeScorer = Callable[[list[TrackCandidate]], bool]   # fills TrackCandidate.vibe in place; True if any song had tags


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------- sources ----------

def choose_period(events: list[ListeningEvent], taken_at: Iterable[datetime | None], now: datetime | None = None
                  ) -> tuple[tuple[datetime, datetime] | None, str, list[str]]:
    """Listening window for the export source: (period or None, period_source, notes)."""
    cfg = get_config().songpick
    now = now or _now()
    taken_at = list(taken_at)
    if album_span(taken_at) is None:
        return None, "history", ["The photos have no capture dates, so songs are ranked by your overall "
                                 "listening (recent plays count more)."]
    scale, first = 1.0, True
    while True:
        period = listening_window(taken_at, now, scale)
        if sum(in_period(e, period) for e in events) >= cfg.min_window_events:
            days = (period[1] - period[0]).days
            notes = [] if first else [f"Few plays right around the photo dates; widened the window to {days} days."]
            return period, "album" if first else "album_widened", notes
        if (period[1] - period[0]).days >= cfg.max_window_days:
            break
        scale *= cfg.widen_factor
        first = False
    return None, "history", ["Your history has too few plays around the photo dates, so songs are ranked "
                             "by your overall listening (recent plays count more)."]


def _export_candidates(events: list[ListeningEvent], taken_at: list[datetime | None], now: datetime):
    period, source, notes = choose_period(events, taken_at, now)
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
            sources=[f"top:{r}" for r in seen[k]], **release_fields(t)))
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
            if c.release_date is not None and (m.release_date is None or (c.release_reliable and not m.release_reliable)):
                m.release_date, m.release_precision, m.release_reliable = (
                    c.release_date, c.release_precision, c.release_reliable)
            m.sources += [s for s in c.sources if s not in m.sources]
            m.reason = f"{m.reason} {c.reason}"
    out = list(merged.values())
    top = max((c.affinity for c in out), default=0.0)
    for c in out:
        c.affinity = round(c.affinity / top, 6) if top > 0 else 0.0
        c.score = c.affinity
    return out


def _signin_groups(recent, top, album_period):
    """Groups, source names, events, events inside the album period, and the keys played during the album."""
    groups: list[tuple[str, list[TrackCandidate]]] = []
    sources: list[str] = []
    played_then: set[str] = set()
    n_win = 0
    if recent and album_period is not None:
        cands, n_win = _recent_candidates(recent, album_period)
        if cands:
            groups.append(("recent", cands))
            sources.append("recent")
            played_then |= {c.key for c in cands}
    if top and any(top.values()):
        ranges = [r for r in RANGE_LABEL if top.get(r)]
        cands = _top_candidates(top, ranges)
        groups.append(("top", cands))
        sources += [f"top:{r}" for r in ranges]
    return groups, sources, n_win, played_then


def _short_term_keys(top, album_period, now) -> set[str]:
    """Songs of the 4-week list when the whole album falls inside those 4 weeks (they were played then)."""
    days = get_config().songpick.spotify.range_days["short_term"]
    if not top or album_period is None or album_period[0] < now - timedelta(days=days):
        return set()
    return {track_key(t.artist, t.track) for t in top.get("short_term") or []}


# ---------- ranking ----------

def _blend(cands: list[TrackCandidate], mode: str) -> None:
    """score = alpha * affinity + (1 - alpha) * vibe (`vibe.alpha_by_mode`); affinity alone when vibe is None."""
    alpha = get_config().songpick.vibe.alpha_by_mode[mode]
    for c in cands:
        c.score = c.affinity if c.vibe is None else round(alpha * c.affinity + (1 - alpha) * c.vibe, 6)


def _vibe_pool(cands: list[TrackCandidate], scorer: VibeScorer | None) -> tuple[list[TrackCandidate], bool]:
    """Run the scorer on the best `vibe.max_tracks` songs by affinity; (pool to rank, vibe used)."""
    if scorer is None:
        return cands, False
    pool = sorted(cands, key=lambda c: (-c.affinity, -c.n_plays_window, c.key))[:get_config().songpick.vibe.max_tracks]
    if not scorer(pool):
        for c in pool:
            c.vibe = None
        return cands, False
    return pool, True


def _rank(cands: list[TrackCandidate], k: int, mode: str, scorer: VibeScorer | None = None
          ) -> tuple[list[TrackCandidate], bool]:
    pool, used = _vibe_pool(cands, scorer)
    _blend(pool, mode)
    pool.sort(key=lambda c: (-c.score, -c.n_plays_window, c.key))
    return pool[:k], used


def _release_pick(cands, taken_at, now, played_then, k, scorer=None):
    """(shortlist, window, widened, notes, vibe used); empty shortlist when nothing came out around the photos."""
    cfg = get_config().songpick.release
    widen = get_config().songpick.widen_factor
    med = median_date(taken_at)
    base = release_before_days(taken_at, now) or 0.0
    dated = [c for c in cands if c.release_date is not None and c.release_reliable]
    n_unreliable = sum(1 for c in cands if c.release_date is not None and not c.release_reliable)

    def inside(win):
        return [c for c in cands if c.key in played_then or (
            c.release_date is not None and c.release_reliable
            and _overlaps(rel.interval(c.release_date, c.release_precision), win))]

    want = min(cfg.min_candidates, len({c.key for c in dated} | played_then))  # widening cannot find more
    scale = 1.0
    win = release_window(taken_at, now, scale)
    elig = inside(win)
    while len(elig) < want and base * scale * widen <= cfg.max_before_days:
        scale *= widen
        win = release_window(taken_at, now, scale)
        elig = inside(win)
    notes = []
    if n_unreliable:
        notes.append(f"{n_unreliable} song{'s' if n_unreliable != 1 else ''} skipped: their Spotify date is a "
                     "remaster or compilation date, not when the song came out.")
    if not elig:
        return [], win, scale > 1.0, notes, False
    if scale > 1.0:
        notes.append(f"Few of your songs came out right around the photos; widened the window to "
                     f"{win[0]:%b %Y} - {win[1]:%b %Y}.")
    top = max(c.affinity for c in elig)
    for c in elig:
        c.affinity = round(c.affinity / top, 6) if top > 0 else 0.0
        c.score = c.affinity
        if c.release_date is not None and c.release_reliable and _overlaps(rel.interval(c.release_date, c.release_precision), win):
            c.reason = (f"Came out {rel.describe(c.release_date, c.release_precision)}, near your photos "
                        f"({med:%b %Y}). {c.reason}")

    def dist(c):
        if c.release_date is None:
            return 0.0
        a, b = rel.interval(c.release_date, c.release_precision)
        return 0.0 if a <= med <= b else min(abs((a - med).total_seconds()), abs((b - med).total_seconds()))

    pool, used = _vibe_pool(elig, scorer)
    _blend(pool, "release")
    pool.sort(key=lambda c: (-c.score, dist(c), c.key))
    if len(elig) < k:
        notes.append(f"Only {len(elig)} of your songs from the past year came out around then.")
    return pool[:k], win, scale > 1.0, notes, used


def _overlaps(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def suggest(taken_at: Iterable[datetime | None], *, mode: str | None = None,
            export: list[ListeningEvent] | None = None, recent: list[ListeningEvent] | None = None,
            top: dict[str, list[TopTrack]] | None = None, now: datetime | None = None,
            k: int | None = None, vibe_scorer: VibeScorer | None = None) -> SongPick:
    """Best `k` (default `shortlist_size`) songs for an album in the given mode (default `default_mode`).
    `vibe_scorer` (optional) fills the vibe of the songs in the ranking pool (see the module doc)."""
    cfg = get_config().songpick
    mode = mode or cfg.default_mode
    if mode not in MODES:
        raise ValueError(f"unknown song-pick mode {mode!r}")
    k = cfg.shortlist_size if k is None else k
    if export is not None and not export:
        raise UserError("The listening history is empty.")
    if mode == "export" and not export:
        raise UserError("Upload your Spotify history files (data export) for this mode, or pick another mode.")
    if not export and not recent and not any((top or {}).values()):
        raise UserError("Connect Spotify (or upload your Spotify history) to get song suggestions.")
    now = now or _now()
    taken_at = list(taken_at)
    span = album_span(taken_at)
    album_period = padded(span, cfg.pad_days) if span else None
    notes: list[str] = []

    if mode == "export":
        cands, period, period_source, notes, n_win = _export_candidates(export, taken_at, now)
        groups, sources = [("export", cands)], ["export"]
        if recent and period is not None:
            rc, r_win = _recent_candidates(recent, period)
            if rc:
                groups.append(("recent", rc))
                sources.append("recent")
            n_win += r_win
        merged = merge_sources(groups)
        if not merged:
            raise UserError("No songs found in your listening data for these photos.")
        out, used = _rank(merged, k, mode, vibe_scorer)
        return _done(out, mode, period, period_source, len(export) + len(recent or []), n_win,
                     notes + _vibe_notes(vibe_scorer, used, mode), sources, used)

    groups, sources, n_win, played_then = _signin_groups(recent, top, album_period)
    if export:  # the export enlarges the pool (whole history, recent plays count more)
        ex = score_tracks(export, None)
        for c in ex:
            c.sources = ["export"]
        groups.insert(0, ("export", ex))
        sources.insert(0, "export")
    n_events = len(export or []) + len(recent or [])
    merged = merge_sources(groups)
    if not merged:
        raise UserError("No songs found in your listening data for these photos.")

    if mode == "release":
        if span is None:
            notes.append(NO_DATES)
        else:
            played_then |= _short_term_keys(top, album_period, now)
            out, win, widened, rnotes, used = _release_pick([c.model_copy(deep=True) for c in merged], taken_at,
                                                            now, played_then, k, vibe_scorer)
            notes += rnotes
            if out:
                return _done(out, "release", win, "release_widened" if widened else "release", n_events, n_win,
                             notes + _vibe_notes(vibe_scorer, used, "release"), sources, used)
            med = median_date(taken_at)
            notes.append(f"None of the songs you played this year came out around {med:%b %Y}; showing your best "
                         "overall picks. Upload your full Spotify history for songs you actually played back then.")
    out, used = _rank(merged, k, "vibe", vibe_scorer)
    return _done(out, "vibe", None, "history", n_events, n_win, notes + _vibe_notes(vibe_scorer, used, "vibe"),
                 sources, used)


def _vibe_notes(scorer: VibeScorer | None, used: bool, mode: str) -> list[str]:
    if used:
        return []
    if scorer is not None:
        return [NO_TAGS]
    return [VIBE_OFF] if mode == "vibe" else []


def _done(out, mode, period, period_source, n_events, n_win, notes, sources, vibe_used=False) -> SongPick:
    log.info("songpick: mode %s, sources %s, period %s, %d of %d plays inside, shortlist %d, vibe %s",
             mode, sources, period_source, n_win, n_events, len(out), "tags" if vibe_used else "none")
    return SongPick(period_start=period[0] if period else None, period_end=period[1] if period else None,
                    period_source=period_source, n_events=n_events, n_events_window=n_win, candidates=out,
                    vibe_source="tags" if vibe_used else "none", notes=notes, sources=sources, mode=mode)


def shortlist(events: list[ListeningEvent], taken_at: Iterable[datetime | None],
              k: int | None = None, now: datetime | None = None) -> SongPick:
    """Export-only shortlist (the data-export path)."""
    return suggest(taken_at, mode="export", export=events, k=k, now=now)
