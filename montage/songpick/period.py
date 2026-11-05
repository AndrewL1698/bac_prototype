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


def album_span(taken_at: Iterable[datetime | None]) -> tuple[datetime, datetime] | None:
    """Robust (start, end) of the capture dates: the earliest/latest `period_trim_frac` are ignored
    so one stray old photo does not stretch the period over years. None when no item has a date."""
    dates = sorted(_naive(d) for d in taken_at if d is not None)
    if not dates:
        return None
    k = int(get_config().songpick.period_trim_frac * len(dates))
    return dates[k], dates[len(dates) - 1 - k]


def padded(span: tuple[datetime, datetime], pad_days: float) -> tuple[datetime, datetime]:
    """Pad both sides, then stretch symmetrically to at least `min_window_days`."""
    cfg = get_config().songpick
    start, end = span[0] - timedelta(days=pad_days), span[1] + timedelta(days=pad_days)
    short = timedelta(days=cfg.min_window_days) - (end - start)
    if short > timedelta(0):
        start, end = start - short / 2, end + short / 2
    return start, end


def album_period(taken_at: Iterable[datetime | None], pad_days: float | None = None) -> tuple[datetime, datetime] | None:
    span = album_span(taken_at)
    if span is None:
        return None
    return padded(span, get_config().songpick.pad_days if pad_days is None else pad_days)
