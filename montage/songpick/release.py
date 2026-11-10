"""Song release dates from Spotify album data (mode "release": songs that came out around the photos).

Spotify gives `album.release_date` as "YYYY", "YYYY-MM" or "YYYY-MM-DD" with a `release_date_precision`.
A less precise date stands for the whole year/month (`interval`). Compilations, remasters and deluxe
editions carry the re-release date, not the song's, so they are flagged unreliable (a name heuristic;
a MusicBrainz first-release lookup is a later option, PLAN Decisions 2026-11-08).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time
from typing import Any

from montage.config import get_config

PRECISIONS = ("year", "month", "day")


def parse_release(value: Any, precision: Any = None) -> tuple[date | None, str | None]:
    """("1999-07-01" or "1999-07" or "1999", precision) -> (first day it covers, precision); bad -> (None, None)."""
    if not isinstance(value, str):
        return None, None
    parts = value.strip().split("-")
    want = precision if precision in PRECISIONS else PRECISIONS[min(len(parts), 3) - 1]
    n = PRECISIONS.index(want) + 1
    if len(parts) < n:
        return None, None
    try:
        nums = [int(p) for p in parts[:n]] + [1] * (3 - n)
        d = date(*nums)
    except ValueError:
        return None, None
    if d.year < 1800:  # Spotify uses "0000" for unknown
        return None, None
    return d, want


def interval(d: date, precision: str | None) -> tuple[datetime, datetime]:
    """The time span a release date stands for (a whole year/month when less precise), as naive datetimes."""
    if precision == "year":
        end = date(d.year, 12, 31)
    elif precision == "month":
        end = date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
    else:
        end = d
    return datetime.combine(d, time.min), datetime.combine(end, time.max)


def _words_re() -> re.Pattern:
    words = get_config().songpick.release.unreliable_words
    return re.compile(r"\b(" + "|".join(re.escape(w.lower()) for w in words) + r")\b") if words else re.compile(r"$^")


def reliable(album_name: str, track_name: str, album_type: Any = None) -> bool:
    """False when the album date is likely a re-release date (compilation, remaster, deluxe edition...)."""
    if isinstance(album_type, str) and album_type.lower() == "compilation":
        return False
    rx = _words_re()
    return not (rx.search((album_name or "").lower()) or rx.search((track_name or "").lower()))


def from_album(album: dict | None, track_name: str) -> dict:
    """Release fields for a ListeningEvent/TopTrack from a Spotify track's `album` object."""
    album = album or {}
    d, prec = parse_release(album.get("release_date"), album.get("release_date_precision"))
    return {"release_date": d, "release_precision": prec,
            "release_reliable": reliable(album.get("name") or "", track_name, album.get("album_type"))}


def describe(d: date, precision: str | None) -> str:
    """Short human form: "in 1999", "Jul 1999"."""
    return f"in {d.year}" if precision == "year" else f"{d:%b %Y}"
