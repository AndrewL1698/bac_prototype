"""Candidates: one per photo, overlapping 3 s windows per video scene, plus thumbnails."""

from __future__ import annotations

from pathlib import Path

from montage import paths
from montage.config import get_config
from montage.ingest.frames import FrameSet
from montage.types import Asset, Candidate

EPS = 1e-6


def window_starts(lo: float, hi: float, window: float, hop: float, min_window: float) -> list[tuple[float, float]]:
    """Windows inside [lo, hi]: full windows on a hop grid; a shorter range gives one window covering it."""
    length = hi - lo
    if length + EPS >= window:
        out, s = [], lo
        while s + window <= hi + EPS:
            out.append((round(s, 3), round(s + window, 3)))
            s += hop
        return out
    if length + EPS >= min_window:
        return [(round(lo, 3), round(hi, 3))]
    return []


def make_thumb(src: str, key: str) -> str:
    from PIL import Image
    w = get_config().candidates.thumb_width
    out = paths.scratch_dir("thumbs") / f"{key}.jpg"
    if not out.exists():
        with Image.open(src) as im:
            im = im.convert("RGB")
            h = max(1, round(im.height * w / im.width))
            im.resize((w, h), Image.LANCZOS).save(out, quality=85)
    return str(out)


def make_candidates(assets: list[Asset], scenes: dict[str, list[tuple[float, float]]],
                    frames: dict[str, FrameSet], limits: dict[str, float | None] | None = None) -> list[Candidate]:
    cfg = get_config().candidates
    limits = limits or {}
    out: list[Candidate] = []
    for a in assets:
        fs = frames[a.id]
        if a.kind == "photo":
            out.append(Candidate(id=a.id, asset_id=a.id, kind="photo",
                                 thumb_path=make_thumb(fs.paths[0], f"{a.hash}_p")))
            continue
        end = min(a.duration or 0.0, limits.get(a.id) or 1e9)
        lo_all, hi_all = cfg.edge_trim_s, end - cfg.edge_trim_s
        n = 0
        spans: list[tuple[float, float, float, int]] = []  # (start, end, max_end, scene idx)
        for si, (s, e) in enumerate(scenes.get(a.id) or [(0.0, end)]):
            lo, hi = max(s, lo_all), min(e, hi_all)
            for ws, we in window_starts(lo, hi, cfg.window_s, cfg.hop_s, cfg.min_window_s):
                spans.append((ws, we, round(hi, 3), si))
        if not spans and hi_all - lo_all >= 0.5:  # very short clip: keep it as one candidate
            spans.append((round(lo_all, 3), round(hi_all, 3), round(hi_all, 3), 0))
        for ws, we, max_end, si in spans:
            ft = [t for t in fs.times if ws - EPS <= t <= we + EPS]
            mid = (ws + we) / 2
            k = min(range(len(fs.times)), key=lambda i: abs(fs.times[i] - mid)) if fs.times else None
            thumb = make_thumb(fs.paths[k], f"{a.hash}_{int(round(ws * 1000))}") if k is not None else ""
            out.append(Candidate(id=f"{a.id}_w{n:02d}", asset_id=a.id, kind="video", src_start=ws, src_end=we,
                                 src_max_end=max_end, scene_id=f"{a.id}_s{si}", frame_times=ft, thumb_path=thumb))
            n += 1
    return out


def frame_indices(cand: Candidate, fs: FrameSet) -> list[int]:
    """Indices into the asset's analysis frames that belong to this candidate (photos: [0])."""
    if cand.kind == "photo":
        return [0]
    idx = [i for i, t in enumerate(fs.times) if cand.src_start - EPS <= t <= cand.src_end + EPS]
    if not idx and fs.times:
        mid = (cand.src_start + cand.src_end) / 2
        idx = [min(range(len(fs.times)), key=lambda i: abs(fs.times[i] - mid))]
    return idx


def thumb_exists(c: Candidate) -> bool:
    return bool(c.thumb_path) and Path(c.thumb_path).exists()
