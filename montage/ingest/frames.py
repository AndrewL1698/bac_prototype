"""Analysis frames on fast scratch (may be regenerated any time).

Videos: ONE FFmpeg decode pass with three outputs:
  - analysis frames at ingest.analysis_fps, short side ingest.analysis_short_side (f_00000.jpg ...)
  - motion pairs: frames at t and t + 1/15 s, short side 320 (m_<idx>_a.jpg / m_<idx>_b.jpg)
  - a 360p proxy at 15 fps for scene detection (proxy.mp4)
Photos: one exif-transposed RGB JPEG at the analysis short side.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from montage import paths
from montage.config import get_config
from montage.log import get_logger
from montage.types import Asset

log = get_logger("frames")
FRAMES_VERSION = 1
HDR_CHAIN = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
             "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
_zscale_ok: bool | None = None


@dataclass
class FrameSet:
    asset_id: str
    times: list[float]                       # analysis frame times (source seconds)
    paths: list[str]                         # analysis JPEGs, aligned with times
    pairs: list[tuple[str, str] | None] = field(default_factory=list)  # motion pair per analysis frame
    proxy: str | None = None                 # 360p proxy for scene detection (videos)
    notes: list[str] = field(default_factory=list)


def zscale_available() -> bool:
    global _zscale_ok
    if _zscale_ok is None:
        try:
            out = subprocess.run([get_config().render.ffmpeg_bin, "-hide_banner", "-filters"],
                                 capture_output=True, text=True).stdout
            _zscale_ok = any(ln.split()[1:2] == ["zscale"] for ln in out.splitlines() if len(ln.split()) > 1)
        except OSError:
            _zscale_ok = False
    return _zscale_ok


def short_side_scale(n: int) -> str:
    # iw/ih are display dims (FFmpeg autorotates); -2 keeps even sizes
    return f"scale='if(gt(iw,ih),-2,{n})':'if(gt(iw,ih),{n},-2)'"


def frame_dir(asset: Asset) -> Path:
    return paths.scratch_dir("frames") / asset.hash


def _params(asset: Asset, limit_s: float | None) -> dict:
    c = get_config().ingest
    return {"v": FRAMES_VERSION, "fps": c.analysis_fps, "side": c.analysis_short_side,
            "pair_side": c.motion_pair_short_side, "pair_dt": c.motion_pair_dt_s,
            "limit": round(limit_s, 3) if limit_s else None, "hdr": asset.is_hdr}


def _load_index(d: Path, params: dict) -> FrameSet | None:
    idx = d / "index.json"
    try:
        data = json.loads(idx.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if data.get("params") != params:
        return None
    fs = FrameSet(asset_id=data["asset_id"], times=data["times"], paths=[str(d / p) for p in data["paths"]],
                  pairs=[(str(d / p[0]), str(d / p[1])) if p else None for p in data["pairs"]],
                  proxy=str(d / data["proxy"]) if data.get("proxy") else None, notes=data.get("notes", []))
    if not all(Path(p).exists() for p in fs.paths):
        return None
    return fs


def _save_index(d: Path, fs: FrameSet, params: dict) -> None:
    rel = lambda p: Path(p).name  # noqa: E731
    data = {"params": params, "asset_id": fs.asset_id, "times": fs.times, "paths": [rel(p) for p in fs.paths],
            "pairs": [[rel(p[0]), rel(p[1])] if p else None for p in fs.pairs],
            "proxy": rel(fs.proxy) if fs.proxy else None, "notes": fs.notes}
    (d / "index.json").write_text(json.dumps(data))


def _photo_frames(asset: Asset, d: Path) -> FrameSet:
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    side = get_config().ingest.analysis_short_side
    out = d / "f_00000.jpg"
    with Image.open(asset.src_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = side / min(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        im.save(out, quality=90)
    return FrameSet(asset_id=asset.id, times=[0.0], paths=[str(out)], pairs=[None])


def _video_frames(asset: Asset, d: Path, limit_s: float | None) -> FrameSet:
    c = get_config()
    fps = c.ingest.analysis_fps
    pair_fps = round(1.0 / c.ingest.motion_pair_dt_s)
    notes = []
    pre = ""
    if asset.is_hdr:
        if zscale_available():
            pre = HDR_CHAIN + ","
        else:
            notes.append(f"{asset.filename}: HDR video without zscale; colors may look washed out")
    mdir = d / "_m"
    mdir.mkdir(exist_ok=True)
    graph = (f"[0:v]{pre}split=3[a][b][c];"
             f"[a]fps={fps},{short_side_scale(c.ingest.analysis_short_side)}[o1];"
             f"[b]fps={pair_fps},{short_side_scale(c.ingest.motion_pair_short_side)}[o2];"
             f"[c]fps={pair_fps},{short_side_scale(360)}[o3]")
    cmd = [c.render.ffmpeg_bin, "-y", "-v", "error", "-i", asset.src_path]
    if limit_s:
        cmd += ["-t", f"{limit_s:.3f}"]
    cmd += ["-filter_complex", graph,
            "-map", "[o1]", "-q:v", "3", str(d / "f_%05d.jpg"),
            "-map", "[o2]", "-q:v", "4", str(mdir / "%06d.jpg"),
            "-map", "[o3]", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
            str(d / "proxy.mp4")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"frame extraction failed for {asset.filename}: {r.stderr.strip()[-300:]}")
    fpaths = sorted(d.glob("f_*.jpg"))
    # FFmpeg numbers from 1; frame k (0-based) shows time k / fps
    times = [round(k / fps, 4) for k in range(len(fpaths))]
    mframes = sorted(mdir.glob("*.jpg"))
    pairs: list[tuple[str, str] | None] = []
    for k, t in enumerate(times):
        j = int(round(t * pair_fps))
        if j + 1 < len(mframes):
            a, b = d / f"m_{k:05d}_a.jpg", d / f"m_{k:05d}_b.jpg"
            shutil.copyfile(mframes[j], a)
            shutil.copyfile(mframes[j + 1], b)
            pairs.append((str(a), str(b)))
        else:
            pairs.append(None)
    shutil.rmtree(mdir, ignore_errors=True)
    return FrameSet(asset_id=asset.id, times=times, paths=[str(p) for p in fpaths], pairs=pairs,
                    proxy=str(d / "proxy.mp4"), notes=notes)


def extract_frames(asset: Asset, limit_s: float | None = None) -> FrameSet:
    d = frame_dir(asset)
    params = _params(asset, limit_s)
    fs = _load_index(d, params)
    if fs is not None:
        fs.asset_id = asset.id
        return fs
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    fs = _photo_frames(asset, d) if asset.kind == "photo" else _video_frames(asset, d, limit_s)
    _save_index(d, fs, params)
    return fs
