"""Per-shot render cache: an encoded shot is reused whenever the same shot is rendered again.

A recut usually changes a few shots; every other shot has the same key, so only the changed shots are
encoded and the assembly (stream-copy concat + song mux) runs as before. The key covers everything that
changes the encoded pixels: the source bytes, which file is read (HDR proxy or original), the source
in-point, the frame count, framing, Ken Burns, the face center, the output size and CRF, and the render
config. Every shot is encoded with identical settings, so cached and fresh shots concat cleanly.

Files are content-addressed and written atomically, so parallel renders (threads or processes) are safe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from montage import cache, paths
from montage.config import get_config
from montage.ingest.frames import zscale_available
from montage.log import get_logger
from montage.render.shots import render_shot
from montage.types import Asset, Shot

log = get_logger("shot_cache")

SHOT_CACHE_VERSION = 1  # bump when shot rendering code changes the pixels
# render config fields that do not change a shot's pixels (or are keyed separately)
_NOT_IN_KEY = {"workers", "ffmpeg_bin", "ffprobe_bin", "audio_bitrate", "preview", "export", "shot_cache"}


def cache_dir() -> Path:
    c = get_config().render.shot_cache
    return paths.cache_dir("shots") if c.location == "drive" else paths.scratch_dir("shot_cache")


def shot_key(shot: Shot, asset: Asset, W: int, H: int, crf: int,
             face: tuple[float, float] = (-1.0, -1.0)) -> str:
    from montage.render.proxy import render_source
    c = get_config().render
    if shot.kind == "video":
        src, needs_tonemap = render_source(asset)
        source = ["proxy" if src != asset.src_path else "original", bool(needs_tonemap and zscale_available())]
    else:
        source = ["photo", False]
    return cache.text_hash(
        SHOT_CACHE_VERSION, asset.hash, shot.kind, source,
        None if shot.src_in is None else round(shot.src_in, 3),
        shot.frame_end - shot.frame_start, shot.framing,
        shot.kenburns.model_dump() if shot.kenburns else None,
        W, H, crf, [round(face[0], 4), round(face[1], 4)],
        c.model_dump(exclude=_NOT_IN_KEY), n=20)


def audio_key(song_hash: str, song_start: float, length: float, fade: float) -> str:
    """Key of the encoded song excerpt (song bytes, excerpt start and length, fade, audio bitrate)."""
    return "audio_" + cache.text_hash(SHOT_CACHE_VERSION, song_hash, round(song_start, 4), round(length, 4),
                                      round(fade, 4), get_config().render.audio_bitrate, n=20)


def cached_file(key: str, ext: str, write: Callable[[Path], None]) -> tuple[Path, bool]:
    """(path, whether it was reused). On a miss, `write(tmp)` produces the file, stored atomically."""
    p = cache_dir() / f"{key}{ext}"
    if p.exists() and p.stat().st_size > 0:
        try:
            os.utime(p)  # recently used: kept by prune()
        except OSError:
            pass
        return p, True
    paths.atomic_write(p, write)
    return p, False


def cached_shot(key: str, shot: Shot, asset: Asset, W: int, H: int, crf: int,
                face: tuple[float, float] = (-1.0, -1.0)) -> tuple[Path, bool]:
    """(path of the encoded shot, whether it was reused). Encodes and stores it on a miss."""
    return cached_file(key, ".mp4", lambda tmp: render_shot(shot, asset, tmp, W, H, crf, face))


def prune(max_gb: float | None = None, keep: set[Path] | None = None) -> int:
    """Delete least recently used shots until the cache is under max_gb; returns how many were deleted."""
    limit = (get_config().render.shot_cache.max_gb if max_gb is None else max_gb) * 1e9
    keep = keep or set()
    files = []
    for p in [*cache_dir().glob("*.mp4"), *cache_dir().glob("*.m4a")]:
        try:
            st = p.stat()
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, p))
    total = sum(s for _, s, _ in files)
    n = 0
    for _, size, p in sorted(files):
        if total <= limit:
            break
        if p in keep:
            continue
        p.unlink(missing_ok=True)
        total -= size
        n += 1
    return n
