"""Render one shot to an MP4 with exactly n = frame_end - frame_start frames (identical encode settings)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from montage.config import get_config
from montage.ingest.frames import HDR_CHAIN, zscale_available
from montage.render import kenburns
from montage.types import Asset, Shot

T = 16 / 9


def encode_args(crf: int) -> list[str]:
    c = get_config().render
    return ["-an", "-c:v", "libx264", "-preset", c.preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-r", str(c.fps), "-video_track_timescale", "15360", "-movflags", "+faststart"]


def crop_filter(asset: Asset, face: tuple[float, float]) -> str:
    """crop=w:h:x:y on the DISPLAY frame (FFmpeg autorotates before filters), centered on the face."""
    w, h = asset.width, asset.height
    if w / h > T:
        ch, cw = h, int(round(h * T))
    else:
        cw, ch = w, int(round(w / T))
    cw, ch = cw - cw % 2, ch - ch % 2
    cx = (face[0] if face[0] >= 0 else 0.5) * w
    cy = (face[1] if face[1] >= 0 else 0.5) * h
    x = int(min(max(cx - cw / 2, 0), w - cw))
    y = int(min(max(cy - ch / 2, 0), h - ch))
    return f"crop={cw}:{ch}:{x}:{y}"


def video_filters(asset: Asset, framing: str, W: int, H: int, face: tuple[float, float]) -> str:
    c = get_config().render
    pre = HDR_CHAIN + "," if asset.is_hdr and zscale_available() else ""
    if framing == "crop":
        body = f"{crop_filter(asset, face)},scale={W}:{H}"
    elif framing == "fill_blur":
        body = (f"split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"gblur=sigma={c.blur_sigma},eq=brightness={c.blur_darken}[bg];"
                f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
    else:
        body = f"scale={W}:{H}"
    return f"{pre}{body},fps={c.fps},tpad=stop_mode=clone:stop_duration=1,setsar=1,format=yuv420p"


def render_video_shot(shot: Shot, asset: Asset, out: Path, W: int, H: int, crf: int,
                      face: tuple[float, float] = (-1.0, -1.0)) -> Path:
    c = get_config().render
    n = shot.frame_end - shot.frame_start
    cmd = [c.ffmpeg_bin, "-y", "-v", "error", "-ss", f"{shot.src_in or 0.0:.3f}", "-i", asset.src_path,
           "-t", f"{n / c.fps + 0.25:.3f}", "-vf", video_filters(asset, shot.framing, W, H, face),
           "-frames:v", str(n), *encode_args(crf), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"video shot {shot.slot_idx} failed: {r.stderr.strip()[-400:]}")
    return out


def render_photo_shot(shot: Shot, asset: Asset, out: Path, W: int, H: int, crf: int,
                      face: tuple[float, float] = (-1.0, -1.0)) -> Path:
    c = get_config().render
    n = shot.frame_end - shot.frame_start
    cmd = [c.ffmpeg_bin, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(c.fps), "-i", "-", "-frames:v", str(n), *encode_args(crf), str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fr in kenburns.frames(asset.src_path, shot.framing, shot.kenburns, n, W, H, face):
            p.stdin.write(fr.tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read().decode("utf-8", "replace")
    if p.wait() != 0:
        raise RuntimeError(f"photo shot {shot.slot_idx} failed: {err.strip()[-400:]}")
    return out


def render_shot(shot: Shot, asset: Asset, out: Path, W: int, H: int, crf: int,
                face: tuple[float, float] = (-1.0, -1.0)) -> Path:
    fn = render_photo_shot if shot.kind == "photo" else render_video_shot
    return fn(shot, asset, out, W, H, crf, face)
