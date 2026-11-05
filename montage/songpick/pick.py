"""Shortlist songs for an album from the user's listening history (phase 12, fallback stage).

Period: the album's capture dates padded by `pad_days`; if fewer than `min_window_events` plays
fall inside, the padding grows by `widen_factor` up to `max_pad_days`; if that is still too few,
or the album has no dates, the whole history is used with a recency decay. No network, no weights.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from montage.config import get_config
from montage.songpick.affinity import in_period, score_tracks
from montage.songpick.period import album_span, padded
from montage.types import ListeningEvent, SongPick, UserError

log = logging.getLogger(__name__)


def choose_period(events: list[ListeningEvent], taken_at: Iterable[datetime | None]
                  ) -> tuple[tuple[datetime, datetime] | None, str, list[str]]:
    """(period or None, period_source, notes)."""
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


def shortlist(events: list[ListeningEvent], taken_at: Iterable[datetime | None],
              k: int | None = None) -> SongPick:
    """Best `k` (default `shortlist_size`) songs for an album, with a reason each."""
    cfg = get_config().songpick
    k = cfg.shortlist_size if k is None else k
    if not events:
        raise UserError("The listening history is empty.")
    taken_at = list(taken_at)
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
    if not cands:
        raise UserError("No full song plays found in the listening history.")
    for c in cands:  # no vibe stage yet: vibe is None and score = affinity (alpha only applies with a vibe)
        if c.vibe is not None:
            c.score = round(cfg.alpha * c.affinity + (1 - cfg.alpha) * c.vibe, 6)
    cands.sort(key=lambda c: (-c.score, -c.n_plays_window, c.key))
    n_win = len(events) if period is None else sum(in_period(e, period) for e in events)
    log.info("songpick: period %s, %d of %d plays inside, %d candidates, shortlist %d",
             source, n_win, len(events), len(cands), min(k, len(cands)))
    return SongPick(period_start=period[0] if period else None, period_end=period[1] if period else None,
                    period_source=source, n_events=len(events), n_events_window=n_win,
                    candidates=cands[:k], vibe_source="none", notes=notes)
