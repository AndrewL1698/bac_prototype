"""Song tags for the stage A vibe match (phase 12.8): Last.fm first, MusicBrainz for the gaps.

Only track and artist names leave the machine. Last.fm needs a free API key (env `songpick.vibe.lastfm.api_key_env`,
Colab Secret LASTFM_API_KEY); without it MusicBrainz (no key, 1 request/s, artist tags) is used alone.
Results are cached in process memory only (a tag list hints at what the user listens to), never on disk, and
logs carry counts only. Every failure gives empty tags: the pick then falls back to listening affinity.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from montage.config import get_config
from montage.log import get_logger
from montage.songpick.vibe import Tags, norm_tag

log = get_logger("songpick.tags")
CACHE_MAX = 5000
_cache: OrderedDict[tuple[str, ...], Tags] = OrderedDict()
_cache_lock = threading.Lock()
_mb_lock = threading.Lock()
_mb_last = [0.0]
_sleep = time.sleep          # tests replace these two
_clock = time.monotonic


def _get_json(url: str, params: dict, headers: dict | None = None, timeout: float = 5.0) -> dict | None:
    """GET -> JSON dict, or None on any error (network, HTTP status, bad JSON)."""
    import requests
    try:
        r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        if r.status_code != 200:
            return None
        body = r.json()
        return body if isinstance(body, dict) else None
    except Exception:  # noqa: BLE001  optional component: no tags, never a crash
        return None


def _cached(key: tuple[str, ...], fn: Callable[[], Tags | None]) -> Tags:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    val = fn()
    if val is None:          # a failed request is not cached, so the next Suggest retries it
        return []
    with _cache_lock:
        _cache[key] = val
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)
    return val


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def clean(raw: list[tuple[str, float]], artist: str) -> Tags:
    """Normalize names, rescale weights to 0-100, drop junk and the artist's own name, keep the best `top_tags`."""
    cfg = get_config().songpick.vibe
    junk = {norm_tag(t) for t in cfg.junk_tags}
    names = {norm_tag(artist)} | {norm_tag(a) for a in artist.split(",")}
    raw = [(norm_tag(t), float(w)) for t, w in raw if isinstance(t, str) and _num(w) is not None]
    top = max((w for _, w in raw), default=0.0)
    if top <= 0:
        return []
    best: dict[str, float] = {}
    for t, w in raw:
        w = 100.0 * w / top
        if not (2 <= len(t) <= 40) or t in junk or t in names or w < cfg.min_tag_weight:
            continue
        best[t] = max(best.get(t, 0.0), w)
    return sorted(((t, round(w, 2)) for t, w in best.items()), key=lambda x: (-x[1], x[0]))[: cfg.top_tags]


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def merge(track_tags: Tags, artist_tags: Tags) -> Tags:
    """Track tags first; artist tags fill in at `artist_tag_weight` of their weight."""
    cfg = get_config().songpick.vibe
    best = dict(track_tags)
    for t, w in artist_tags:
        best[t] = max(best.get(t, 0.0), round(w * cfg.artist_tag_weight, 2))
    return sorted(best.items(), key=lambda x: (-x[1], x[0]))[: cfg.top_tags]


# ---------- Last.fm ----------

def lastfm_key() -> str | None:
    return os.environ.get(get_config().songpick.vibe.lastfm.api_key_env) or None


def _lastfm_tags(body: dict | None) -> list[tuple[str, float]] | None:
    if body is None or "error" in body:
        return None if body is None else []
    tags = (body.get("toptags") or {}).get("tag") or []
    if isinstance(tags, dict):  # Last.fm returns a bare object when there is a single tag
        tags = [tags]
    return [(t.get("name"), t.get("count", 0)) for t in tags if isinstance(t, dict)]


def _lastfm(method: str, key: str, **params: str) -> list[tuple[str, float]] | None:
    cfg = get_config().songpick.vibe.lastfm
    body = _get_json(cfg.url, {"method": method, "api_key": key, "format": "json", "autocorrect": "1", **params},
                     timeout=cfg.timeout_s)
    return _lastfm_tags(body)


def lastfm_track(artist: str, track: str, key: str) -> Tags:
    cfg = get_config().songpick.vibe.lastfm
    tt = _cached(("lfm_t", artist.lower(), track.lower()), lambda: _clean_or_none(
        _lastfm("track.gettoptags", key, artist=artist, track=track), artist))
    if len(tt) >= cfg.min_tags:
        return tt
    at = _cached(("lfm_a", artist.lower()), lambda: _clean_or_none(
        _lastfm("artist.gettoptags", key, artist=artist), artist))
    return merge(tt, at)


def _clean_or_none(raw, artist: str) -> Tags | None:
    return None if raw is None else clean(raw, artist)


# ---------- MusicBrainz ----------

def _mb_wait() -> None:
    gap = get_config().songpick.vibe.musicbrainz.interval_s
    with _mb_lock:
        wait = _mb_last[0] + gap - _clock()
        if wait > 0:
            _sleep(wait)
        _mb_last[0] = _clock()


def _lucene(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _mb_artist(artist: str) -> Tags | None:
    cfg = get_config().songpick.vibe.musicbrainz
    _mb_wait()
    body = _get_json(cfg.url + "artist", {"query": f'artist:"{_lucene(artist)}"', "limit": "1", "fmt": "json"},
                     headers={"User-Agent": cfg.user_agent, "Accept": "application/json"}, timeout=cfg.timeout_s)
    if body is None:
        return None
    hits = body.get("artists") or []
    if not hits or _num(hits[0].get("score")) is None or float(hits[0]["score"]) < cfg.min_score:
        return []
    raw = [(t.get("name"), t.get("count", 0)) for t in hits[0].get("tags") or [] if isinstance(t, dict)]
    return clean(raw, artist)


def musicbrainz(artist: str) -> Tags:
    """Artist tags (MusicBrainz has few tags per recording). Multi-artist credits use the first artist."""
    first = artist.split(",")[0].strip() or artist
    return _cached(("mb_a", first.lower()), lambda: _mb_artist(first))


# ---------- lookup ----------

def lookup(tracks: list[tuple[str, str, str]]) -> dict[str, Tags]:
    """{key: tags} for (key, artist, track) triples, best effort, in the given (affinity) order."""
    cfg = get_config().songpick.vibe
    out: dict[str, Tags] = {}
    key = lastfm_key()
    n_lfm = n_mb = 0
    if key and tracks:
        with ThreadPoolExecutor(max_workers=max(1, cfg.lastfm.workers)) as ex:
            res = list(ex.map(lambda t: lastfm_track(t[1], t[2], key), tracks))
        for (k, _, _), tags in zip(tracks, res):
            if tags:
                out[k] = tags
                n_lfm += 1
    todo = [t for t in tracks if t[0] not in out][: cfg.musicbrainz.max_lookups]
    for k, artist, _ in todo:
        tags = musicbrainz(artist)
        if tags:
            out[k] = tags
            n_mb += 1
    log.info("songpick: tags for %d of %d songs (Last.fm %d%s, MusicBrainz %d)", len(out), len(tracks), n_lfm,
             "" if key else " - no key", n_mb)
    return out
