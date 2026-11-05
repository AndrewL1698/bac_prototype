"""Period affinity: which songs did you play during the album's period, more than usual?

Per play: weight = min(ms_played / full_play_ms, 1) for plays of at least min_play_ms (else 0),
minus skip_penalty if skipped. Per track (key "artist|track", lowercased):
- plays_window = sum of weights inside the period, plays_total = sum over the whole history.
- expected = plays_total * (all window plays / all plays): the period share the track would get
  if you played it as evenly as everything else.
- lift = (plays_window + lift_prior) / (expected + lift_prior): above 1 = "more than usual".
- affinity = plays_window * lift ** sig_weight, max-normalized. The lift surfaces "your song of
  that summer" instead of your all-time favourites.
Without a period, every play counts with a recency decay and lift = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from montage.config import get_config
from montage.types import ListeningEvent, TrackCandidate


def track_key(artist: str, track: str) -> str:
    return f"{' '.join(artist.lower().split())}|{' '.join(track.lower().split())}"


def play_weight(e: ListeningEvent) -> float:
    cfg = get_config().songpick
    w = min(e.ms_played / cfg.full_play_ms, 1.0) if e.ms_played >= cfg.min_play_ms else 0.0
    return w - (cfg.skip_penalty if e.skipped else 0.0)


def in_period(e: ListeningEvent, period: tuple[datetime, datetime]) -> bool:
    return period[0] <= e.ts <= period[1]


@dataclass
class _Acc:
    event: ListeningEvent                  # most recent play (display names, uri)
    window: float = 0.0
    total: float = 0.0
    n_window: int = 0
    n_skips: int = 0
    last_window: datetime | None = None
    uri: str | None = None
    album: str = ""
    duration_ms: int | None = None


def _fmt_day(d: datetime) -> str:
    return f"{d:%b} {d.day}, {d.year}"


def _reason(a: _Acc, lift: float, period: tuple[datetime, datetime] | None) -> str:
    n = a.n_window
    times = "once" if n == 1 else f"{n} times"
    if period is None:
        last = f", most recently {_fmt_day(a.last_window)}" if a.last_window else ""
        text = f"One of your most played songs ({times}{last})"
    else:
        text = f"You played it {times} between {_fmt_day(period[0])} and {_fmt_day(period[1])}"
        if lift >= 1.5:
            text += f", {lift:.1f}x more than usual for you"
    if a.n_skips:
        text += f"; skipped {a.n_skips}x"
    return text + "."


def score_tracks(events: list[ListeningEvent], period: tuple[datetime, datetime] | None,
                 min_plays: int | None = None) -> list[TrackCandidate]:
    """Every eligible track with its affinity, best first. Deterministic for a given history."""
    cfg = get_config().songpick
    min_plays = cfg.min_window_plays if min_plays is None else min_plays
    latest = max((e.ts for e in events), default=None)
    acc: dict[str, _Acc] = {}
    for e in events:  # sorted by time, so the last event per key is the most recent
        k = track_key(e.artist, e.track)
        a = acc.get(k)
        if a is None:
            a = acc[k] = _Acc(event=e)
        a.event = e
        a.uri = e.uri or a.uri
        a.album = e.album or a.album
        a.duration_ms = e.duration_ms or a.duration_ms
        w = play_weight(e)
        a.total += w
        inside = in_period(e, period) if period is not None else True
        if not inside:
            continue
        if period is None and latest is not None:
            age_days = (latest - e.ts).total_seconds() / 86400.0
            if w > 0:  # decay plays; skip penalties keep full weight
                w *= 0.5 ** (age_days / cfg.recency_half_life_days)
        a.window += w
        if e.ms_played >= cfg.min_play_ms:
            a.n_window += 1
            a.last_window = e.ts
        if e.skipped:
            a.n_skips += 1

    t_win = sum(max(a.window, 0.0) for a in acc.values())
    t_all = sum(max(a.total, 0.0) for a in acc.values())
    rows: list[tuple[float, _Acc, float]] = []
    for a in acc.values():
        pw = max(a.window, 0.0)
        if a.n_window < min_plays or pw <= 0:
            continue
        if period is None or t_all <= 0:
            lift = 1.0
        else:
            expected = max(a.total, 0.0) * t_win / t_all
            lift = (pw + cfg.lift_prior) / (expected + cfg.lift_prior)
        rows.append((pw * lift ** cfg.sig_weight, a, lift))
    if not rows:
        return []
    top = max(r[0] for r in rows)
    out = []
    for raw, a, lift in rows:
        aff = raw / top
        out.append(TrackCandidate(
            key=track_key(a.event.artist, a.event.track), track=a.event.track, artist=a.event.artist,
            album=a.album, uri=a.uri, plays_window=round(a.window, 4), plays_total=round(a.total, 4),
            n_plays_window=a.n_window, n_skips_window=a.n_skips, lift=round(lift, 4),
            affinity=round(aff, 6), score=round(aff, 6), reason=_reason(a, lift, period),
            duration_ms=a.duration_ms))
    out.sort(key=lambda c: (-c.affinity, -c.n_plays_window, c.key))
    return out
