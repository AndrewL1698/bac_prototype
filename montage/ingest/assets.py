"""Ingest: files -> validated, probed, ordered Assets (probe cached by content hash)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from montage import cache
from montage.config import get_config
from montage.ingest import probe
from montage.log import get_logger
from montage.types import Asset, UserError

log = get_logger("ingest")
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
PROBE_VERSION = 1


def kind_of(path: str | Path) -> str | None:
    ext = Path(path).suffix.lower()
    return "photo" if ext in PHOTO_EXT else "video" if ext in VIDEO_EXT else None


def collect(paths: list[str | Path], warnings: list[str] | None = None) -> list[Path]:
    """Expand directories (recursively), keep supported files, warn about the rest."""
    out: list[Path] = []
    for p in map(Path, paths):
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        for f in files:
            if f.name.startswith("."):
                continue
            if kind_of(f):
                out.append(f)
            else:
                msg = f"Skipped unsupported file {f.name}"
                log.warning(msg)
                if warnings is not None:
                    warnings.append(msg)
    seen, uniq = set(), []
    for f in out:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def _probe_one(path: Path) -> tuple[Path, str, dict]:
    h = cache.file_hash(path)
    cp = cache.cache_path("assets", h, ".json")
    data = cache.read_json(cp)
    if not data or data.get("probe_version") != PROBE_VERSION:
        k = kind_of(path)
        meta = probe.probe_video(path) if k == "video" else probe.probe_photo(path)
        data = {"probe_version": PROBE_VERSION, "kind": k, **meta}
        cache.write_json(cp, data)
    return path, h, data


def ingest_with_notes(paths: list[str | Path]) -> tuple[list[Asset], list[str]]:
    cfg = get_config().ingest
    notes: list[str] = []
    files = collect(paths, notes)
    if len(files) < cfg.min_assets:
        raise UserError(f"Need at least {cfg.min_assets} photos or videos (got {len(files)}).")
    if len(files) > cfg.max_assets:
        raise UserError(f"At most {cfg.max_assets} photos or videos per montage (got {len(files)}).")
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        probed = list(ex.map(_probe_one, files))

    rows = []
    mtime_count = 0
    for path, h, d in probed:
        taken = datetime.fromisoformat(d["taken_at"]) if d.get("taken_at") else None
        if taken is None:
            taken = datetime.fromtimestamp(os.path.getmtime(path))
            mtime_count += 1
        rows.append((taken, path.name, path, h, d))
    if mtime_count:
        notes.append(f"{mtime_count} file(s) had no capture date; used file time (order may be unreliable).")
    rows.sort(key=lambda r: (r[0], r[1]))

    assets: list[Asset] = []
    seen_hash: set[str] = set()
    for taken, name, path, h, d in rows:
        if h in seen_hash:
            notes.append(f"Skipped exact duplicate file {name}")
            continue
        seen_hash.add(h)
        assets.append(Asset(
            id=f"a{len(assets):03d}", kind=d["kind"], src_path=str(path), hash=h,
            width=d["width"], height=d["height"], rotation=d.get("rotation", 0),
            duration=d.get("duration"), fps=d.get("fps"), codec=d.get("codec"),
            is_hdr=bool(d.get("is_hdr")), taken_at=taken, filename=name,
        ))
    if len(assets) < cfg.min_assets:
        raise UserError(f"Need at least {cfg.min_assets} distinct photos or videos (got {len(assets)}).")
    total_video = sum(a.duration or 0 for a in assets if a.kind == "video")
    if total_video > cfg.max_total_video_s:
        notes.append(f"Videos total {total_video:.0f}s; analyzed only the first {cfg.max_per_video_s:.0f}s of each long video.")
    return assets, notes


def ingest(paths: list[str | Path]) -> list[Asset]:
    return ingest_with_notes(paths)[0]


def analysis_limit(asset: Asset, assets: list[Asset]) -> float | None:
    """Seconds of this video to analyze (None = all). Long videos are capped when the total is too long."""
    if asset.kind != "video" or asset.duration is None:
        return None
    cfg = get_config().ingest
    total = sum(a.duration or 0 for a in assets if a.kind == "video")
    if total > cfg.max_total_video_s:
        return min(asset.duration, cfg.max_per_video_s)
    return asset.duration
