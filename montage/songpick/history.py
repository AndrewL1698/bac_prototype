"""Parse Spotify streaming-history exports into ListeningEvents.

Two export formats are accepted (both are JSON lists of plays):
- Extended streaming history (`Streaming_History_Audio_*.json`): `ts` (UTC, end of play),
  `ms_played`, `master_metadata_track_name`, `master_metadata_album_artist_name`,
  `master_metadata_album_album_name`, `spotify_track_uri`, `skipped`, `reason_end`.
- Account data (`StreamingHistory*.json`): `endTime` ("YYYY-MM-DD HH:MM", UTC), `artistName`,
  `trackName`, `msPlayed`.

Podcast episodes, audiobooks and rows without a track name are dropped. Only the fields of
ListeningEvent are kept: IP addresses, user agents, countries and the like are never read.
The history is personal data: never commit it and never log track lists (counts only).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from montage.types import ListeningEvent, UserError

log = logging.getLogger(__name__)

HISTORY_GLOBS = ("Streaming_History_Audio*.json", "endsong*.json", "StreamingHistory_music*.json", "StreamingHistory[0-9]*.json")


def _parse_ts(s: Any) -> datetime | None:
    """ISO 'ts' or 'endTime' -> naive UTC datetime."""
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def parse_row(row: dict) -> ListeningEvent | None:
    """One export row -> event, or None for podcasts, audiobooks, and malformed rows."""
    if not isinstance(row, dict):
        return None
    if "ts" in row:  # extended streaming history
        track = row.get("master_metadata_track_name")
        artist = row.get("master_metadata_album_artist_name")
        if row.get("episode_name") or row.get("spotify_episode_uri") or row.get("audiobook_title"):
            return None
        ts, ms = _parse_ts(row.get("ts")), row.get("ms_played")
        album = row.get("master_metadata_album_album_name") or ""
        uri = row.get("spotify_track_uri") or None
        skipped = bool(row.get("skipped")) or row.get("reason_end") == "fwdbtn"
    else:  # account-data history
        track, artist = row.get("trackName"), row.get("artistName")
        ts, ms = _parse_ts(row.get("endTime")), row.get("msPlayed")
        album, uri, skipped = "", None, False
    if not track or not artist or ts is None or not isinstance(ms, (int, float)) or ms < 0:
        return None
    return ListeningEvent(ts=ts, ms_played=int(ms), track=str(track), artist=str(artist),
                          album=str(album), uri=uri if isinstance(uri, str) else None, skipped=skipped)


def history_files(paths: Iterable[str | Path]) -> list[Path]:
    """Files given directly, plus history files found (recursively) in given folders."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found = {f for g in HISTORY_GLOBS for f in p.rglob(g)}
            out.extend(sorted(found))
        elif p.is_file():
            out.append(p)
    return out


def parse_rows(rows: Iterable[dict]) -> list[ListeningEvent]:
    events = [e for e in map(parse_row, rows) if e is not None]
    events.sort(key=lambda e: (e.ts, e.artist, e.track))
    return events


def load_history(paths: Iterable[str | Path]) -> list[ListeningEvent]:
    """Load and merge every history file; exact duplicate rows (same play in two files) are dropped."""
    files = history_files(paths)
    if not files:
        raise UserError("No Spotify streaming history found. Upload the JSON files from your Spotify "
                        "data export (Streaming_History_Audio_*.json).")
    rows: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise UserError(f"Could not read {f.name} as a Spotify history file ({type(e).__name__}).") from e
        if not isinstance(data, list):
            raise UserError(f"{f.name} is not a Spotify history file (expected a list of plays).")
        rows.extend(data)
    events = parse_rows(rows)
    seen: set[tuple] = set()
    unique = []
    for e in events:
        k = (e.ts, e.ms_played, e.track, e.artist)
        if k not in seen:
            seen.add(k)
            unique.append(e)
    if not unique:
        raise UserError("The history files contain no music plays.")
    log.info("songpick: %d music plays from %d file(s) (%d rows read)", len(unique), len(files), len(rows))
    return unique
