"""Media preparation: ingest -> frames -> scenes -> candidates (all cached)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from montage.config import get_config
from montage.types import Asset, Candidate


@dataclass
class MediaSet:
    assets: list[Asset]
    candidates: list[Candidate]
    frames: dict = field(default_factory=dict)      # asset_id -> FrameSet
    notes: list[str] = field(default_factory=list)

    def asset(self, asset_id: str) -> Asset:
        return next(a for a in self.assets if a.id == asset_id)


def build_media(paths: list[str | Path], progress: Callable[[float, str], None] | None = None) -> MediaSet:
    from montage.candidates.scenes import detect_scenes
    from montage.candidates.windows import make_candidates
    from montage.ingest.assets import analysis_limit, ingest_with_notes
    from montage.ingest.frames import extract_frames

    workers = get_config().ingest.workers
    assets, notes = ingest_with_notes(paths)
    limits = {a.id: analysis_limit(a, assets) for a in assets}
    if progress:
        progress(0.1, f"ingested {len(assets)} files")

    def prep(a: Asset):
        fs = extract_frames(a, limits[a.id])
        sc = detect_scenes(a, fs, limits[a.id]) if a.kind == "video" else []
        return a.id, fs, sc

    frames, scenes = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (aid, fs, sc) in enumerate(ex.map(prep, assets)):
            frames[aid], scenes[aid] = fs, sc
            notes.extend(fs.notes)
            if progress:
                progress(0.1 + 0.8 * (i + 1) / len(assets), "extracting frames")
    cands = make_candidates(assets, scenes, frames, limits)
    if progress:
        progress(1.0, f"{len(cands)} candidates")
    return MediaSet(assets=assets, candidates=cands, frames=frames, notes=notes)
