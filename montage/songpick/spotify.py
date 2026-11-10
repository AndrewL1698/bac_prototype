"""'Connect Spotify': read-only listening data through the Spotify Web API.

OAuth Authorization Code with PKCE (RFC 7636): no client secret, only the Client ID from the
`SPOTIFY_CLIENT_ID` Colab Secret. Scopes: recently played + top tracks, nothing else.

The redirect URI is a fixed static page (docs/spotify/callback.html on GitHub Pages) because the
app's own share URL changes on every restart. `state` carries the app URL; the page sends the
browser back to it (only to hosts in `songpick.spotify.return_hosts`) with `?spotify_code=...`.

Tokens and listening data live only in the user's app session (gr.State): never written to disk,
never logged. Spotify Development Mode allows 5 allowlisted users per app; others get a 403.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urlsplit

from montage.config import get_config
from montage.songpick import release
from montage.types import ListeningEvent, TopTrack, UserError

log = logging.getLogger(__name__)

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_URL = "https://api.spotify.com/v1"
TIME_RANGES = ("short_term", "medium_term", "long_term")

_pending: dict[str, tuple[str, float]] = {}   # state -> (code_verifier, expires_at)
_lock = threading.Lock()


# ---------- PKCE and state ----------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_verifier() -> str:
    """43-128 characters from the unreserved set (RFC 7636 section 4.1)."""
    return secrets.token_urlsafe(64)[:96]


def code_challenge(verifier: str) -> str:
    """S256: BASE64URL(SHA256(verifier)), no padding."""
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def safe_return_url(url: str) -> bool:
    """Same rule as docs/spotify/callback.html: https on an allowed host suffix (e.g. *.gradio.live),
    or http on a loopback host. Anything else would make the callback page an open redirect."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if not host or parts.username or parts.password:
        return False
    for allowed in get_config().songpick.spotify.return_hosts:
        allowed = allowed.lower()
        if allowed.startswith("."):
            if parts.scheme == "https" and host.endswith(allowed) and len(host) > len(allowed):
                return True
        elif host == allowed and parts.scheme in ("http", "https"):
            return True
    return False


def make_state(app_url: str) -> str:
    """'<nonce>~<base64url(app_url)>': the callback page decodes the part after '~'."""
    return f"{secrets.token_urlsafe(16)}~{_b64url(app_url.encode('utf-8'))}"


def client_id() -> str | None:
    return os.environ.get(get_config().songpick.spotify.client_id_env) or None


def is_configured() -> bool:
    return client_id() is not None


def begin_auth(app_url: str, now: float | None = None) -> str:
    """Spotify authorize URL for this app instance; remembers the PKCE verifier for `auth_ttl_s`."""
    cfg = get_config().songpick.spotify
    cid = client_id()
    if not cid:
        raise UserError("Spotify sign-in is not set up for this app (no SPOTIFY_CLIENT_ID secret). "
                        "Upload a song instead, or ask the app owner to add the Spotify Client ID.")
    app_url = app_url.split("?", 1)[0].split("#", 1)[0]
    if not safe_return_url(app_url):
        raise UserError("Spotify sign-in only works from the app's share link (https://...gradio.live) or "
                        "a local address.")
    now = time.time() if now is None else now
    verifier, state = make_verifier(), make_state(app_url)
    with _lock:
        for k in [k for k, (_, exp) in _pending.items() if exp < now]:
            del _pending[k]
        _pending[state] = (verifier, now + cfg.auth_ttl_s)
    q = {"client_id": cid, "response_type": "code", "redirect_uri": cfg.redirect_uri,
         "scope": " ".join(cfg.scopes), "state": state, "code_challenge_method": "S256",
         "code_challenge": code_challenge(verifier)}
    return f"{AUTH_URL}?{urlencode(q)}"


# ---------- HTTP (one seam, so tests can replace it) ----------

def _request(method: str, url: str, **kw: Any) -> tuple[int, dict, dict]:
    """(status, json body or {}, headers). Network errors become UserError."""
    import requests
    try:
        r = requests.request(method, url, timeout=get_config().songpick.spotify.timeout_s, **kw)
    except requests.RequestException as e:
        raise UserError(f"Could not reach Spotify ({type(e).__name__}). Check the connection and try again.") from e
    try:
        body = r.json() if r.content else {}
    except ValueError:
        body = {}
    return r.status_code, body if isinstance(body, dict) else {}, dict(r.headers)


def _check(status: int, body: dict, what: str) -> dict:
    if status == 200:
        return body
    if status == 401:
        raise UserError("Your Spotify sign-in expired. Connect Spotify again.")
    if status == 403:
        raise UserError("Spotify refused access for this account. While the app is in Spotify's Development Mode, "
                        "only accounts the app owner has added (up to 5) can connect.")
    if status == 429:
        raise UserError("Spotify is limiting requests right now. Try again in a minute.")
    log.warning("spotify: %s failed with HTTP %d", what, status)  # never log the body (may echo tokens)
    raise UserError(f"Spotify could not {what} (HTTP {status}). Try again, or upload a song instead.")


def finish_auth(code: str, state: str, now: float | None = None) -> dict:
    """Exchange the code for a token dict {access_token, expires_at}. The state must be one we issued."""
    cfg = get_config().songpick.spotify
    now = time.time() if now is None else now
    with _lock:
        entry = _pending.pop(state, None)
    if entry is None or entry[1] < now:
        raise UserError("That Spotify sign-in link has expired or was already used. Connect Spotify again.")
    if not code:
        raise UserError("Spotify did not return a sign-in code. Connect Spotify again.")
    status, body, _ = _request("POST", TOKEN_URL, data={
        "grant_type": "authorization_code", "code": code, "redirect_uri": cfg.redirect_uri,
        "client_id": client_id() or "", "code_verifier": entry[0]})
    if status == 400:
        raise UserError("Spotify rejected the sign-in (the code expired or was used). Connect Spotify again.")
    body = _check(status, body, "finish the sign-in")
    if not body.get("access_token"):
        raise UserError("Spotify sign-in returned no access token. Connect Spotify again.")
    return {"access_token": body["access_token"], "expires_at": now + float(body.get("expires_in", 3600))}


def _get(token: dict, path: str, params: dict, what: str) -> dict:
    if not token or not token.get("access_token"):
        raise UserError("Connect Spotify first.")
    if token.get("expires_at", 0) < time.time():
        raise UserError("Your Spotify sign-in expired. Connect Spotify again.")
    status, body, _ = _request("GET", f"{API_URL}{path}", params=params,
                               headers={"Authorization": f"Bearer {token['access_token']}"})
    return _check(status, body, what)


# ---------- parsing (pure) ----------

def _artist(track: dict) -> str:
    artists = track.get("artists") or []
    return ", ".join(a.get("name", "") for a in artists if isinstance(a, dict) and a.get("name"))


def _played_at(s: Any) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def recent_from_json(body: dict) -> list[ListeningEvent]:
    """/me/player/recently-played -> events (a listed play counts as a full play: Spotify only lists
    plays of at least 30 s and gives no play length)."""
    out = []
    for item in body.get("items") or []:
        t = (item or {}).get("track") or {}
        ts, name, artist = _played_at(item.get("played_at")), t.get("name"), _artist(t)
        if not ts or not name or not artist or t.get("type", "track") != "track":
            continue
        dur = t.get("duration_ms")
        dur = int(dur) if isinstance(dur, (int, float)) and dur > 0 else None
        out.append(ListeningEvent(ts=ts, ms_played=dur or get_config().songpick.full_play_ms, track=name,
                                  artist=artist, album=((t.get("album") or {}).get("name") or ""),
                                  uri=t.get("uri"), source="recent", duration_ms=dur,
                                  **release.from_album(t.get("album"), name)))
    out.sort(key=lambda e: (e.ts, e.artist, e.track))
    return out


def top_from_json(body: dict, time_range: str) -> list[TopTrack]:
    out = []
    for t in body.get("items") or []:
        name, artist = (t or {}).get("name"), _artist(t or {})
        if not name or not artist:
            continue
        dur = t.get("duration_ms")
        out.append(TopTrack(track=name, artist=artist, album=((t.get("album") or {}).get("name") or ""),
                            uri=t.get("uri"), duration_ms=int(dur) if isinstance(dur, (int, float)) and dur > 0 else None,
                            time_range=time_range, rank=len(out) + 1, **release.from_album(t.get("album"), name)))
    return out


# ---------- fetch ----------

def recently_played(token: dict) -> list[ListeningEvent]:
    return recent_from_json(_get(token, "/me/player/recently-played", {"limit": 50}, "read your recent plays"))


def top_tracks(token: dict, time_range: str) -> list[TopTrack]:
    if time_range not in TIME_RANGES:
        raise ValueError(f"unknown time range {time_range!r}")
    limit = get_config().songpick.spotify.top_limit
    body = _get(token, "/me/top/tracks", {"time_range": time_range, "limit": limit}, "read your top songs")
    return top_from_json(body, time_range)


def fetch_listening(token: dict) -> tuple[list[ListeningEvent], dict[str, list[TopTrack]]]:
    """Recent plays plus the three top-tracks lists. Logs counts only."""
    recent = recently_played(token)
    top = {r: top_tracks(token, r) for r in TIME_RANGES}
    log.info("spotify: %d recent plays, top lists %s", len(recent), {r: len(v) for r, v in top.items()})
    return recent, top
