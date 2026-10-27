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
    """Largest 16:9 crop of the DISPLAY frame (FFmpeg autorotates before filters), centered on the face.
    Written relative to the input size, so it also fits a downscaled proxy."""
    cx = face[0] if face[0] >= 0 else 0.5
    cy = face[1] if face[1] >= 0 else 0.5
    w = r"trunc(min(iw\,ih*16/9)/2)*2"
    h = r"trunc(min(ih\,iw*9/16)/2)*2"
    x = rf"min(max({cx:.4f}*iw-ow/2\,0)\,iw-ow)"
    y = rf"min(max({cy:.4f}*ih-oh/2\,0)\,ih-oh)"
    return f"crop=w={w}:h={h}:x={x}:y={y}"


def video_filters(asset: Asset, framing: str, W: int, H: int, face: tuple[float, float],
                  needs_tonemap: bool | None = None) -> str:
    c = get_config().render
    hdr = asset.is_hdr if needs_tonemap is None else needs_tonemap
    # HDR without a proxy: tonemap the already-cropped/scaled W x H frame (not the full-resolution source)
    post = "," + HDR_CHAIN if hdr and zscale_available() else ""
    if framing == "crop":
        body = f"{crop_filter(asset, face)},scale={W}:{H}"
    elif framing == "fill_blur":
        body = (f"split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"gblur=sigma={c.blur_sigma},eq=brightness={c.blur_darken}[bg];"
                f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
    else:
        body = f"scale={W}:{H}"
    return f"{body}{post},fps={c.fps},tpad=stop_mode=clone:stop_duration=1,setsar=1,format=yuv420p"


def render_video_shot(shot: Shot, asset: Asset, out: Path, W: int, H: int, crf: int,
                      face: tuple[float, float] = (-1.0, -1.0)) -> Path:
    from montage.render.proxy import render_source
    c = get_config().render
    n = shot.frame_end - shot.frame_start
    src, needs_tonemap = render_source(asset)
    cmd = [c.ffmpeg_bin, "-y", "-v", "error", "-ss", f"{shot.src_in or 0.0:.3f}", "-i", src,
           "-t", f"{n / c.fps + 0.25:.3f}", "-vf", video_filters(asset, shot.framing, W, H, face, needs_tonemap),
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
