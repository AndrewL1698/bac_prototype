"""SDR render proxies for HDR videos (built once, cached on Drive by content hash).

Tonemapping every rendered frame is the most expensive part of rendering phone HDR (HLG/PQ) clips.
A proxy is the whole clip tonemapped once to SDR H.264 (upright, short side <= 1080, short GOP for fast
seeks). Renders use the proxy when it exists and the original otherwise, so results never depend on it.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from montage import cache, paths
from montage.config import get_config
from montage.ingest.frames import HDR_CHAIN, short_side_scale, zscale_available
from montage.log import get_logger
from montage.types import Asset

log = get_logger("proxy")
PROXY_VERSION = 1


def proxy_path(asset: Asset) -> Path:
    return cache.cache_path("proxies", f"{asset.hash}_v{PROXY_VERSION}", ".mp4")


def render_source(asset: Asset) -> tuple[str, bool]:
    """(path to read for rendering, whether it still needs tonemapping)."""
    if asset.kind == "video" and asset.is_hdr:
        p = proxy_path(asset)
        if p.exists() and p.stat().st_size > 0:
            return str(p), False
    return asset.src_path, asset.is_hdr


def ensure_proxy(asset: Asset) -> Path | None:
    if asset.kind != "video" or not asset.is_hdr or not zscale_available():
        return None
    out = proxy_path(asset)
    if out.exists() and out.stat().st_size > 0:
        return out
    c = get_config().render
    scale = short_side_scale(1080) if min(asset.width, asset.height) > 1080 else "null"

    def write(tmp: Path) -> None:
        r = subprocess.run([c.ffmpeg_bin, "-y", "-v", "error", "-i", asset.src_path, "-vf", f"{scale},{HDR_CHAIN}",
                            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-g", "15",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"proxy failed for {asset.filename}: {r.stderr.strip()[-300:]}")

    paths.atomic_write(out, write)
    return out


def ensure_proxies(assets: list[Asset], workers: int = 4) -> int:
    """Build missing proxies for the HDR videos among `assets`; returns how many exist afterwards."""
    todo = [a for a in assets if a.kind == "video" and a.is_hdr]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for a, res in zip(todo, ex.map(lambda a: _safe(a), todo)):
            if res is None:
                log.warning("no proxy for %s (renders use the original)", a.filename)
    return sum(proxy_path(a).exists() for a in todo)


def _safe(a: Asset) -> Path | None:
    try:
        return ensure_proxy(a)
    except Exception as e:  # noqa: BLE001  proxies are an optimization only
        log.warning("%s", e)
        return None
