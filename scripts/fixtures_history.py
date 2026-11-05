"""Synthetic Spotify streaming history for tests (made-up artists and tracks, never real data).

Two years of plays (2025-2026) with a known answer for an album taken 2026-07-03..07-10:
- "Summer Anthem" (The Fixtures) is played heavily only during July 1-14 2026: the expected top pick.
- "Forever Favourite" (Old Reliable) is played steadily all the time: popular, but not special then.
- "Skip Me" (Nope Band) is played often in July but always skipped: never shortlisted.
- Background songs are played evenly; podcast episodes and sub-30 s plays are noise.
Written in the extended format (Streaming_History_Audio_*.json) plus one small account-data
format file (StreamingHistory_music_0.json). Pure Python, deterministic for a given seed.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

ALBUM_DATES = ["2026-07-03T10:00:00", "2026-07-05T15:30:00", "2026-07-07T19:10:00", "2026-07-10T12:00:00"]
TOP_PICK = ("The Fixtures", "Summer Anthem")
FAVOURITE = ("Old Reliable", "Forever Favourite")
SKIPPED = ("Nope Band", "Skip Me")
BACKGROUND = [(f"Artist {c}", f"Song {c}{i}") for c in "ABCDEFGH" for i in (1, 2)]
START, END = datetime(2025, 1, 1), datetime(2026, 12, 31)
SUMMER = (datetime(2026, 7, 1), datetime(2026, 7, 14))


def _row(ts: datetime, artist: str, track: str, ms: int, skipped: bool = False) -> dict:
    return {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "platform": "synthetic", "ms_played": ms,
        "conn_country": "ZZ", "ip_addr": "0.0.0.0",
        "master_metadata_track_name": track, "master_metadata_album_artist_name": artist,
        "master_metadata_album_album_name": f"{artist} Album",
        "spotify_track_uri": "spotify:track:" + hashlib.sha1(f"{artist}|{track}".encode()).hexdigest()[:22],
        "episode_name": None, "episode_show_name": None, "spotify_episode_uri": None,
        "reason_start": "trackdone", "reason_end": "fwdbtn" if skipped else "trackdone",
        "shuffle": False, "skipped": skipped, "offline": False, "incognito_mode": False,
    }


def _rand_time(rng: random.Random, a: datetime, b: datetime) -> datetime:
    return a + timedelta(seconds=rng.randrange(int((b - a).total_seconds())))


def synthetic_rows(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for _ in range(1400):  # background, evenly over two years
        a, t = rng.choice(BACKGROUND)
        rows.append(_row(_rand_time(rng, START, END), a, t, rng.randrange(150_000, 240_000)))
    for _ in range(260):  # the all-time favourite
        rows.append(_row(_rand_time(rng, START, END), *FAVOURITE, rng.randrange(170_000, 230_000)))
    for _ in range(22):  # the song of that summer
        rows.append(_row(_rand_time(rng, *SUMMER), *TOP_PICK, rng.randrange(180_000, 210_000)))
    for _ in range(4):
        rows.append(_row(_rand_time(rng, START, SUMMER[0]), *TOP_PICK, rng.randrange(180_000, 210_000)))
    for _ in range(25):  # often started in July, always skipped early
        rows.append(_row(_rand_time(rng, *SUMMER), *SKIPPED, rng.randrange(31_000, 40_000), skipped=True))
    for _ in range(60):  # sub-30 s plays never count
        a, t = rng.choice(BACKGROUND)
        rows.append(_row(_rand_time(rng, START, END), a, t, rng.randrange(1_000, 29_000)))
    for _ in range(40):  # podcast episodes
        r = _row(_rand_time(rng, START, END), "", "", 1_200_000)
        r.update(master_metadata_track_name=None, master_metadata_album_artist_name=None,
                 master_metadata_album_album_name=None, spotify_track_uri=None,
                 episode_name="Episode", episode_show_name="Some Podcast",
                 spotify_episode_uri="spotify:episode:0")
        rows.append(r)
    rows.sort(key=lambda r: r["ts"])
    return rows


def account_rows(seed: int = 0) -> list[dict]:
    """A few plays in the older account-data format (endTime, artistName, trackName, msPlayed)."""
    rng = random.Random(seed + 1)
    out = []
    for _ in range(30):
        a, t = rng.choice(BACKGROUND)
        ts = _rand_time(rng, START, END)
        out.append({"endTime": ts.strftime("%Y-%m-%d %H:%M"), "artistName": a, "trackName": t,
                    "msPlayed": rng.randrange(150_000, 240_000)})
    return out


def write_history(out_dir: Path, seed: int = 0) -> dict:
    """Write the synthetic export into out_dir; returns the truth for tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = synthetic_rows(seed)
    half = len(rows) // 2
    (out_dir / "Streaming_History_Audio_2025_0.json").write_text(json.dumps(rows[:half]))
    (out_dir / "Streaming_History_Audio_2025-2026_1.json").write_text(json.dumps(rows[half:]))
    (out_dir / "StreamingHistory_music_0.json").write_text(json.dumps(account_rows(seed)))
    return {"album_dates": ALBUM_DATES, "top_pick": list(TOP_PICK), "favourite": list(FAVOURITE),
            "skipped": list(SKIPPED), "n_rows": len(rows)}


# ---------- Spotify sign-in API responses (synthetic, same shapes as the Web API) ----------

def _api_track(artist: str, track: str, duration_ms: int = 200_000) -> dict:
    return {"type": "track", "name": track, "duration_ms": duration_ms,
            "uri": "spotify:track:" + hashlib.sha1(f"{artist}|{track}".encode()).hexdigest()[:22],
            "artists": [{"name": artist}], "album": {"name": f"{artist} Album"}}


def spotify_recent_json(now: datetime, seed: int = 0) -> dict:
    """/me/player/recently-played: 50 plays over the last 10 days; "Summer Anthem" dominates the last 3 days."""
    rng = random.Random(seed + 2)
    items = []
    for i in range(50):
        if i < 14:
            artist, track, ago = *TOP_PICK, rng.uniform(0, 3)
        else:
            (artist, track), ago = rng.choice(BACKGROUND + [FAVOURITE]), rng.uniform(0, 10)
        ts = now - timedelta(days=ago)
        items.append({"track": _api_track(artist, track, 195_000 if track == TOP_PICK[1] else 210_000),
                      "played_at": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")})
    items.sort(key=lambda x: x["played_at"], reverse=True)  # the API lists newest first
    return {"items": items, "limit": 50}


def spotify_top_json(time_range: str) -> dict:
    """/me/top/tracks: the favourite tops the 1-year list; "Summer Anthem" tops the 4-week list."""
    order = {
        "short_term": [TOP_PICK] + BACKGROUND[:9] + [FAVOURITE],
        "medium_term": [BACKGROUND[3], TOP_PICK] + BACKGROUND[4:12] + [FAVOURITE],
        "long_term": [FAVOURITE] + BACKGROUND[:12],
    }[time_range]
    return {"items": [_api_track(a, t) for a, t in order], "limit": 50}
