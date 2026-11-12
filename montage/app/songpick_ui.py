"""'Pick a song for me' handlers for the Gradio app (plain values in and out, so tests need no Gradio).

Flow: Connect Spotify (redirect to Spotify, back through docs/spotify/callback.html) -> Suggest songs
(capture dates and mood of the chosen photos + listening data + song tags) -> Use this song (opt-in YouTube fetch) -> the
song player is filled and the normal Generate flow runs unchanged.

The per-user state `sp` (a gr.State dict) holds the Spotify token and the fetched listening data. It
lives only in the app process for that browser session: never written to disk, never logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from montage.config import get_config
from montage.log import get_logger
from montage.songpick import pick as pick_mod
from montage.songpick import spotify
from montage.songpick.history import load_history
from montage.types import SongPick, UserError

log = get_logger("songpick.ui")
NO_MOOD = "Couldn't read your photos' mood, so songs are ranked by your listening."


def connect(page_url: str) -> tuple[str, str]:
    """(Spotify authorize URL, status markdown with a fallback link)."""
    url = spotify.begin_auth(page_url or "")
    return url, f"Opening Spotify... If nothing happens, [continue to Spotify]({url})."


def _status(sp: dict) -> str:
    n_top = sum(len(v) for v in (sp.get("top") or {}).values())
    return (f"Spotify connected: {len(sp.get('recent') or [])} recent plays and {n_top} top songs "
            f"(read-only; kept only for this browser session).")


def finish(code: str, state: str) -> tuple[dict, str]:
    token = spotify.finish_auth(code, state)
    recent, top = spotify.fetch_listening(token)
    sp = {"token": token, "recent": recent, "top": top}
    return sp, _status(sp)


def on_page_load(query: dict) -> tuple[dict | None, str]:
    """Called on every page load with the URL query. Returns (new sp state or None to keep, status)."""
    code, state = query.get("spotify_code"), query.get("state")
    if not code:
        if spotify.is_configured():
            return None, ""
        return None, "Spotify sign-in is not set up on this server (no SPOTIFY_CLIENT_ID). You can still upload a song."
    try:
        return finish(code, state or "")
    except UserError as e:
        return None, str(e)


def paste_code(text: str) -> tuple[dict, str]:
    """Fallback when the callback page could not redirect: it shows '<code> <state>' to paste here."""
    parts = (text or "").split()
    if len(parts) != 2:
        raise UserError("Paste the whole code shown on the Spotify page (two parts separated by a space).")
    return finish(parts[0], parts[1])


def _album_dates(media: list[str]):
    from montage.ingest.assets import ingest_with_notes
    if not media:
        raise UserError("Add photos and videos (or pick a demo album) first, so the app knows when they were taken.")
    assets, _ = ingest_with_notes(media)
    return [a.taken_at for a in assets]


def _album(media: list[str], progress: Callable[[float, str], None] | None = None):
    """(capture dates, AlbumVibe or None, notes). The photos' mood uses the editor's cached feature path, so
    Generate is warm afterwards; if it fails, songs are ranked by listening alone."""
    if not media:
        raise UserError("Add photos and videos (or pick a demo album) first, so the app knows when they were taken.")
    if not get_config().songpick.vibe.enabled:
        return _album_dates(media), None, []
    from montage.songpick import vibe
    try:
        av = vibe.album_vibe(media, progress)
        return av.taken_at, av, []
    except UserError:
        raise
    except Exception as e:  # noqa: BLE001  optional component: fall back to listening affinity
        log.warning("songpick: album mood failed (%s: %s); ranking by listening", type(e).__name__, e)
        return _album_dates(media), None, [NO_MOOD]


MODE_CHOICES = [("Songs from back then (release date)", "release"), ("Best vibe fit", "vibe"),
                ("My full Spotify history (data export)", "export")]


def suggest(media: list[str], history_files: list[str] | None, sp: dict | None, mode: str | None = None,
            progress: Callable[[float, str], None] | None = None) -> tuple[list[tuple[str, str]], SongPick, str]:
    """(radio choices [(label, index)], the SongPick, markdown with the window and notes)."""
    sp = sp or {}
    if mode == "export" and not history_files:
        raise UserError("Upload your Spotify history files (data export JSON) for this mode, or pick another mode.")
    export = load_history(history_files) if history_files else None
    if not export and not sp.get("recent") and not any((sp.get("top") or {}).values()):
        raise UserError("Connect Spotify (or upload your Spotify history files) first.")
    taken_at, album, notes = _album(media, progress)
    scorer = None
    if album is not None:
        from montage.songpick import tags, vibe
        scorer = vibe.Scorer(album, tags.lookup)
        if progress:
            progress(0.92, "looking up song tags")
    pick = pick_mod.suggest(taken_at, mode=mode, export=export, recent=sp.get("recent"), top=sp.get("top"),
                            vibe_scorer=scorer)
    if album is not None:
        pick.album_valence, pick.album_arousal = round(album.valence, 4), round(album.arousal, 4)
    pick.notes = notes + pick.notes
    choices = [(f"{c.artist} - {c.track}: {c.reason}", str(i)) for i, c in enumerate(pick.candidates)]
    if pick.mode == "release" and pick.period_start and pick.period_end:
        head = f"Songs that came out around your photos ({pick.period_start:%b %Y} - {pick.period_end:%b %Y})."
    elif pick.mode == "export" and pick.period_start and pick.period_end:
        head = f"Songs you played around {pick.period_start:%b %d, %Y} - {pick.period_end:%b %d, %Y}."
    elif pick.vibe_source != "none":
        head = "Your songs that best fit the photos' vibe."
    else:
        head = "Your best picks overall."
    md = "\n".join([head] + [f"> {n}" for n in pick.notes])
    return choices, pick, md


def _manual(c, why: str) -> str:
    from montage.songpick import youtube
    return (f"{why} [Find \"{c.artist} - {c.track}\" on YouTube]({youtube.search_url(c.track, c.artist)}) "
            "(opens in your browser), get the song file you're allowed to use, then drop it in the **Song** box.")


def use_song(choice: str | None, pick: SongPick | None, youtube_ok: bool) -> tuple[str | None, str]:
    """(audio path for the Song player or None, status markdown). Fetches from YouTube only when opted in;
    when the fetch is off or fails, the status links to a YouTube search so the user can upload the file."""
    if pick is None or not pick.candidates:
        raise UserError("Press Suggest songs first.")
    try:
        c = pick.candidates[int(choice if choice not in (None, "") else 0)]
    except (ValueError, IndexError) as e:
        raise UserError("Pick one of the suggested songs.") from e
    if not youtube_ok:
        return None, _manual(c, "To fetch it automatically, tick \"Fetch the song's audio from YouTube\". Or:")
    from montage.songpick import youtube
    try:
        path, meta = youtube.find_and_fetch(c.track, c.artist, c.duration_ms)
    except UserError as e:
        return None, _manual(c, f"{e}")
    dur = meta.get("duration")
    length = f", {int(dur) // 60}:{int(dur) % 60:02d}" if isinstance(dur, (int, float)) else ""
    who = meta.get("channel") or meta.get("uploader") or "YouTube"
    return str(youtube.serve_copy(path)), f"Using **{c.artist} - {c.track}** (from YouTube: \"{meta.get('title', '')}\", {who}{length})."
