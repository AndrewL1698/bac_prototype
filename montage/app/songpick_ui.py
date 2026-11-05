"""'Pick a song for me' handlers for the Gradio app (plain values in and out, so tests need no Gradio).

Flow: Connect Spotify (redirect to Spotify, back through docs/spotify/callback.html) -> Suggest songs
(capture dates of the chosen photos + listening data) -> Use this song (opt-in YouTube fetch) -> the
song player is filled and the normal Generate flow runs unchanged.

The per-user state `sp` (a gr.State dict) holds the Spotify token and the fetched listening data. It
lives only in the app process for that browser session: never written to disk, never logged.
"""

from __future__ import annotations

from pathlib import Path

from montage.songpick import pick as pick_mod
from montage.songpick import spotify
from montage.songpick.history import load_history
from montage.types import SongPick, UserError


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


def suggest(media: list[str], history_files: list[str] | None, sp: dict | None) -> tuple[list[tuple[str, str]], SongPick, str]:
    """(radio choices [(label, index)], the SongPick, markdown with the period and notes)."""
    sp = sp or {}
    export = load_history(history_files) if history_files else None
    if not export and not sp.get("recent") and not any((sp.get("top") or {}).values()):
        raise UserError("Connect Spotify (or upload your Spotify history files) first.")
    pick = pick_mod.suggest(_album_dates(media), export=export, recent=sp.get("recent"), top=sp.get("top"))
    choices = [(f"{c.artist} - {c.track}: {c.reason}", str(i)) for i, c in enumerate(pick.candidates)]
    if pick.period_start and pick.period_end:
        head = f"Songs from around {pick.period_start:%b %d, %Y} - {pick.period_end:%b %d, %Y}."
    else:
        head = "Songs from your overall listening."
    md = "\n".join([head] + [f"> {n}" for n in pick.notes])
    return choices, pick, md


def use_song(choice: str | None, pick: SongPick | None, youtube_ok: bool) -> tuple[str, str]:
    """(audio path for the Song player, status markdown). Fetches from YouTube only when opted in."""
    if pick is None or not pick.candidates:
        raise UserError("Press Suggest songs first.")
    try:
        c = pick.candidates[int(choice if choice not in (None, "") else 0)]
    except (ValueError, IndexError) as e:
        raise UserError("Pick one of the suggested songs.") from e
    if not youtube_ok:
        raise UserError(f"To fetch \"{c.track}\" automatically, tick \"Fetch the song's audio from YouTube\". "
                        "Or upload the song file yourself in the Song box.")
    from montage.songpick import youtube
    path, meta = youtube.find_and_fetch(c.track, c.artist, c.duration_ms)
    dur = meta.get("duration")
    length = f", {int(dur) // 60}:{int(dur) % 60:02d}" if isinstance(dur, (int, float)) else ""
    who = meta.get("channel") or meta.get("uploader") or "YouTube"
    return str(Path(path)), f"Using **{c.artist} - {c.track}** (from YouTube: \"{meta.get('title', '')}\", {who}{length})."
