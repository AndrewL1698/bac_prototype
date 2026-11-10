"""The album's time period: when the photos and videos were taken.

Capture dates (`Asset.taken_at`) are the local wall clock while history times are UTC; the
difference (at most about a day) is far smaller than the padding, so they are compared as is.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from montage.config import get_config


def _naive(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo is not None else dt


def _trimmed(taken_at: Iterable[datetime | None]) -> list[datetime]:
    """Sorted capture dates without the earliest/latest `period_trim_frac` (one stray old photo must
    not stretch the period over years)."""
    dates = sorted(_naive(d) for d in taken_at if d is not None)
    k = int(get_config().songpick.period_trim_frac * len(dates))
    return dates[k:len(dates) - k] if dates else []


def album_span(taken_at: Iterable[datetime | None]) -> tuple[datetime, datetime] | None:
    """Robust (start, end) of the capture dates. None when no item has a date."""
    dates = _trimmed(taken_at)
    return (dates[0], dates[-1]) if dates else None


def median_date(taken_at: Iterable[datetime | None]) -> datetime | None:
    """Median of the trimmed capture dates (the album's "when"). None when no item has a date."""
    dates = _trimmed(taken_at)
    if not dates:
        return None
    a, b = dates[(len(dates) - 1) // 2], dates[len(dates) // 2]
    return a + (b - a) / 2


def age_scaled_days(age_years: float, table: list[list[float]]) -> float:
    """Piecewise-linear lookup in [[age_years, days], ...], clamped at both ends."""
    pts = sorted((float(a), float(d)) for a, d in table)
    if age_years <= pts[0][0]:
        return pts[0][1]
    for (a0, d0), (a1, d1) in zip(pts, pts[1:]):
        if age_years <= a1:
            return d0 + (d1 - d0) * (age_years - a0) / (a1 - a0) if a1 > a0 else d1
    return pts[-1][1]


def age_years(when: datetime, now: datetime) -> float:
    """Photo age in years, counted in whole days so a pick does not change within a day."""
    return max((_naive(now).date() - when.date()).days / 365.25, 0.0)


def release_before_days(taken_at: Iterable[datetime | None], now: datetime) -> float | None:
    """Base number of days before the median a song may have come out (by photo age)."""
    med = median_date(taken_at)
    if med is None:
        return None
    return age_scaled_days(age_years(med, now), get_config().songpick.release.before_days_by_age)


def release_window(taken_at: Iterable[datetime | None], now: datetime, scale: float = 1.0
                   ) -> tuple[datetime, datetime] | None:
    """Mode "release": when a song must have come out to count as "from back then". Asymmetric (songs are
    mostly heard after release): median - before .. median + after_frac * before; always covers the album."""
    taken_at = list(taken_at)
    med, span, before = median_date(taken_at), album_span(taken_at), release_before_days(taken_at, now)
    if med is None or span is None or before is None:
        return None
    before *= scale
    after = get_config().songpick.release.after_frac * before
    return min(med - timedelta(days=before), span[0]), max(med + timedelta(days=after), span[1])


def listening_window(taken_at: Iterable[datetime | None], now: datetime, scale: float = 1.0
                     ) -> tuple[datetime, datetime] | None:
    """Mode "export": when the user must have played a song. Centered on the median, total width by photo
    age (`window_by_age_days`) times `scale`; always covers the album span (at least `min_window_days`)."""
    taken_at = list(taken_at)
    med, span = median_date(taken_at), album_span(taken_at)
    if med is None or span is None:
        return None
    half = timedelta(days=age_scaled_days(age_years(med, now), get_config().songpick.window_by_age_days) * scale / 2)
    album = padded(span, 0.0)
    return min(med - half, album[0]), max(med + half, album[1])


def padded(span: tuple[datetime, datetime], pad_days: float) -> tuple[datetime, datetime]:
    """Pad both sides, then stretch symmetrically to at least `min_window_days`."""
    cfg = get_config().songpick
    start, end = span[0] - timedelta(days=pad_days), span[1] + timedelta(days=pad_days)
    short = timedelta(days=cfg.min_window_days) - (end - start)
    if short > timedelta(0):
        start, end = start - short / 2, end + short / 2
    return start, end
