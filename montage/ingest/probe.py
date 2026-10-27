"""Probe media metadata: ffprobe for videos, Pillow/EXIF for photos. Sizes are DISPLAY sizes."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from montage.config import get_config

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
EXIF_ROTATION = {3: 180, 6: 90, 8: 270, 5: 90, 7: 270}


def ffprobe_json(path: str | Path) -> dict:
    cmd = [get_config().render.ffprobe_bin, "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {Path(path).name}: {r.stderr.strip()[:200]}")
    return json.loads(r.stdout or "{}")


def _rate(s: str | None) -> float | None:
    if not s or s in ("0/0", "0"):
        return None
    if "/" in s:
        n, d = s.split("/", 1)
        return float(n) / float(d) if float(d) else None
    return float(s)


def parse_time(s: str | None) -> datetime | None:
    """ISO-ish container/EXIF timestamps -> naive datetime (offset dropped: keep the local wall clock)."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    try:  # 2026-07-14T10:00:00-0700
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def rotation_from_stream(vs: dict) -> int:
    """Clockwise degrees needed for display (legacy `rotate` tag convention)."""
    for sd in vs.get("side_data_list") or []:
        if "rotation" in sd:  # display matrix: counter-clockwise
            return int(round(-float(sd["rotation"]))) % 360
    tag = (vs.get("tags") or {}).get("rotate")
    if tag is not None:
        return int(round(float(tag))) % 360
    return 0


def probe_video(path: str | Path) -> dict:
    info = ffprobe_json(path)
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
                and not (s.get("disposition") or {}).get("attached_pic")]
    if not vstreams:
        raise RuntimeError(f"no video stream in {Path(path).name}")
    vs = vstreams[0]
    fmt = info.get("format", {})
    rot = rotation_from_stream(vs)
    w, h = int(vs.get("width", 0)), int(vs.get("height", 0))
    if rot in (90, 270):
        w, h = h, w
    dur = vs.get("duration") or fmt.get("duration")
    tags = {**(fmt.get("tags") or {}), **(vs.get("tags") or {})}
    taken = parse_time(tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time"))
    return {
        "width": w, "height": h, "rotation": rot,
        "duration": float(dur) if dur else None,
        "fps": _rate(vs.get("avg_frame_rate")) or _rate(vs.get("r_frame_rate")),
        "codec": vs.get("codec_name"),
        "is_hdr": vs.get("color_transfer") in HDR_TRANSFERS,
        "taken_at": taken.isoformat() if taken else None,
    }


def probe_photo(path: str | Path) -> dict:
    from PIL import Image
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        exif = im.getexif()
        orientation = int(exif.get(0x0112, 1) or 1)
        w, h = im.size
        stamp = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    taken = parse_time(stamp if isinstance(stamp, str) else None)
    return {"width": w, "height": h, "rotation": EXIF_ROTATION.get(orientation, 0),
            "taken_at": taken.isoformat() if taken else None}
