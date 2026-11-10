"""Opt-in: fetch the picked song's audio from YouTube with yt-dlp.

Downloading from YouTube is against YouTube's terms and the songs are copyrighted; the app only does
it when the user ticks the opt-in box and clicks "Use this song" (PLAN Decisions, 2026-11-05). Only the
search text "<artist> - <track>" leaves the machine. Audio is cached on the private Drive at
`cache/youtube_audio/<video_id>.m4a` (media is gitignored and blocked by the hygiene test), so recut
and export keep working. The app never serves that folder: it plays a temp copy (`serve_copy`), which
Gradio moves into its own content-hashed cache. Upload stays the fallback for every failure.

`pick_best` is a pure function (unit-tested): title/artist similarity, a bonus for official
"<Artist> - Topic" uploads, a penalty for live/cover/remix-style versions, and a duration check
against Spotify's track length when it is known.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from montage import paths
from montage.config import get_config
from montage.types import UserError

log = logging.getLogger(__name__)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _norm(s: str, keep_brackets: bool = False) -> str:
    s = s.lower()
    if not keep_brackets:
        s = re.sub(r"[\(\[].*?[\)\]]", " ", s)   # drop "(Official Video)", "[Lyrics]" for name matching
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _sim(needle: str, hay: str) -> float:
    n, h = _norm(needle), _norm(hay)
    if not n:
        return 0.0
    if re.search(rf"\b{re.escape(n)}\b", h):
        return 1.0
    return SequenceMatcher(None, n, h).ratio()


def score_result(r: dict, track: str, artist: str, duration_ms: int | None) -> float | None:
    """Match score for one search result, or None if it cannot be the song (bad id, length out of range)."""
    cfg = get_config().songpick.youtube
    vid, title = r.get("id") or "", r.get("title") or ""
    channel = r.get("channel") or r.get("uploader") or ""
    dur = r.get("duration")
    if not VIDEO_ID_RE.match(vid) or not title:
        return None
    if isinstance(dur, (int, float)) and not (cfg.min_song_s <= dur <= cfg.max_song_s):
        return None
    first_artist = artist.split(",")[0].strip()
    s = 0.6 * _sim(track, title) + 0.4 * max(_sim(first_artist, title), _sim(first_artist, channel))
    if channel.lower().endswith(" - topic") and _sim(first_artist, channel[:-8]) >= 0.8:
        s += cfg.topic_bonus
    t, wanted = _norm(title, keep_brackets=True), _norm(track, keep_brackets=True)  # "(Live at ...)" counts
    for w in cfg.bad_words:
        w = _norm(w)
        if re.search(rf"\b{re.escape(w)}\b", t) and not re.search(rf"\b{re.escape(w)}\b", wanted):
            s -= cfg.bad_penalty
            break
    if duration_ms and isinstance(dur, (int, float)):
        diff = abs(dur - duration_ms / 1000.0)
        s += -cfg.duration_penalty if diff > cfg.duration_tol_s else 0.1 * (1 - diff / cfg.duration_tol_s)
    return s


def pick_best(results: list[dict], track: str, artist: str, duration_ms: int | None = None
              ) -> tuple[dict, float] | None:
    """Best acceptable result (score >= min_match), ties broken by search order."""
    cfg = get_config().songpick.youtube
    best: tuple[dict, float] | None = None
    for r in results:
        s = score_result(r or {}, track, artist, duration_ms)
        if s is not None and s >= cfg.min_match and (best is None or s > best[1]):
            best = (r, s)
    return best


# ---------- yt-dlp (network; one seam each so tests can replace them) ----------

def cookies_path() -> str | None:
    """cookies.txt for yt-dlp: env `cookies_env` (written by colab_bootstrap from the YOUTUBE_COOKIES secret to
    local VM disk), else config `cookies_file`. Never on the shared Drive or in the repo; never logged."""
    cfg = get_config().songpick.youtube
    for p in (os.environ.get(cfg.cookies_env or ""), cfg.cookies_file):
        if p and Path(p).is_file():
            return p
    return None


def _ydl_opts(**extra: Any) -> dict:
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True}
    cookies = cookies_path()
    if cookies:
        opts["cookiefile"] = cookies
    opts.update(extra)
    return opts


def _friendly(e: Exception) -> UserError:
    msg = str(e).lower()
    if "confirm you" in msg and "bot" in msg:
        if cookies_path():
            return UserError("YouTube is blocking downloads from this server even with your YouTube cookies (it "
                             "asks to confirm you're not a bot); they may have expired. Re-export them into the "
                             "YOUTUBE_COOKIES secret (see README), or upload the song file instead.")
        return UserError("YouTube is blocking downloads from this server right now (it asks to confirm you're "
                         "not a bot). Add a YOUTUBE_COOKIES Colab Secret from a spare YouTube account (see README), "
                         "or upload the song file instead.")
    return UserError(f"Could not get the song from YouTube ({type(e).__name__}). Upload the song file instead.")


def search_url(track: str, artist: str) -> str:
    """A YouTube search link for the user to open in their own browser (the manual fallback)."""
    return "https://www.youtube.com/results?search_query=" + quote_plus(f"{artist.split(',')[0].strip()} - {track}")


def search(query: str, n: int) -> list[dict]:
    """Metadata of the first n search results (no download)."""
    try:
        import yt_dlp
    except ImportError as e:
        raise UserError("YouTube fetching is not installed on this server (yt-dlp). Upload the song instead.") from e
    try:
        with yt_dlp.YoutubeDL(_ydl_opts(skip_download=True, extract_flat="in_playlist")) as ydl:
            info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    except Exception as e:  # noqa: BLE001  (yt-dlp raises many types)
        raise _friendly(e) from e
    return [x for x in (info or {}).get("entries") or [] if isinstance(x, dict)]


def _download(video_id: str, out_dir: Path) -> Path:
    import yt_dlp
    opts = _ydl_opts(format="bestaudio[ext=m4a]/bestaudio/best", outtmpl=str(out_dir / "%(id)s.%(ext)s"),
                     postprocessors=[{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
                     ffmpeg_location=shutil.which(get_config().render.ffmpeg_bin) or None)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    files = sorted(out_dir.glob(f"{video_id}.*"))
    m4a = [f for f in files if f.suffix == ".m4a"]
    if not (m4a or files):
        raise RuntimeError("download produced no file")
    return (m4a or files)[0]


def audio_path(video_id: str) -> Path:
    return paths.cache_dir("youtube_audio") / f"{video_id}.m4a"


def serve_copy(path: Path | str) -> Path:
    """Copy a cached song into the system temp dir for the app's Song player. The Drive cache is not an
    allowed Gradio path (anyone with the share link could fetch songs by video id); temp files are."""
    src = Path(path)
    out_dir = Path(tempfile.gettempdir()) / "montage_songpick"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / src.name
    shutil.copy2(src, out)
    return out


def fetch_audio(video_id: str) -> Path:
    """Cached m4a for a YouTube video id (downloads once, writes atomically)."""
    if not VIDEO_ID_RE.match(video_id or ""):
        raise ValueError(f"bad YouTube video id {video_id!r}")
    out = audio_path(video_id)
    if out.exists() and out.stat().st_size > 0:
        return out
    with tempfile.TemporaryDirectory(dir=paths.scratch_dir("youtube")) as tmp:
        try:
            got = _download(video_id, Path(tmp))
        except Exception as e:  # noqa: BLE001
            raise _friendly(e) from e
        paths.atomic_write(out, lambda p: shutil.copyfile(got, p))
    return out


def find_and_fetch(track: str, artist: str, duration_ms: int | None = None) -> tuple[Path, dict]:
    """Search, pick the best match, download. Returns (audio path, chosen result metadata)."""
    cfg = get_config().songpick.youtube
    first_artist = artist.split(",")[0].strip()
    results = search(f"{first_artist} - {track}", cfg.max_results)
    best = pick_best(results, track, artist, duration_ms)
    if best is None:
        raise UserError(f"Could not find a clear match for \"{track}\" on YouTube. Upload the song file instead.")
    r, s = best
    log.info("youtube: fetching a match (score %.2f, %ss)", s, r.get("duration"))  # no titles in logs
    return fetch_audio(r["id"]), r
