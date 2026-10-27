"""Feature extraction: per-frame features cached per asset; candidate features aggregated on load.

Cache: DATA_ROOT/cache/features/<features.version>/<asset hash>.npz (+ .json header), rows aligned
with that asset's analysis frames (photos: 1 row). Batch-relative values (mood z-scores, duplicate
clusters) are computed after loading and never cached.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from montage import cache, paths
from montage.candidates.windows import frame_indices
from montage.config import get_config
from montage.features import aesthetic, clip_encoder, dedup, faces, mood, motion, quality, tags
from montage.ingest.frames import FrameSet
from montage.log import get_logger
from montage.types import Asset, Candidate, FeatureTable

log = get_logger("features")
FRAME_KEYS = ("emb", "aesthetic", "sharpness", "brightness", "contrast", "exposure_clip", "colorfulness",
              "motion", "gdx", "gdy", "face_count", "face_area", "face_cx", "face_cy", "times")


@dataclass
class Features:
    table: FeatureTable
    arrays: dict[str, np.ndarray]                       # candidate-aligned
    frames: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # asset_id -> per-frame arrays
    kinds: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return self.table.candidate_ids

    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.ids)}


def _cache_file(a: Asset, ext: str):
    return paths.cache_dir(f"features/{get_config().features.version}") / f"{a.hash}{ext}"


def load_asset(a: Asset, fs: FrameSet) -> dict[str, np.ndarray] | None:
    z = cache.load_npz(_cache_file(a, ".npz"))
    if z is None or any(k not in z for k in FRAME_KEYS):
        return None
    if len(z["times"]) != len(fs.times) or not np.allclose(z["times"], fs.times, atol=1e-3):
        return None
    return z


def compute_assets(todo: list[tuple[Asset, FrameSet]], progress: Callable | None = None) -> dict[str, dict]:
    """Run all models for the given assets (all frames through CLIP together)."""
    workers = get_config().ingest.workers
    all_paths, owners = [], []
    for a, fs in todo:
        all_paths += fs.paths
        owners += [a.id] * len(fs.paths)
    owners = np.array(owners)
    if progress:
        progress(0.05, f"CLIP on {len(all_paths)} frames")
    emb = clip_encoder.encode_images(all_paths)
    aes = aesthetic.score(emb)
    if progress:
        progress(0.5, "quality, motion, faces")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        q = list(ex.map(quality.frame_quality, all_paths))
        fc = list(ex.map(faces.frame_faces, all_paths))
        pairs = [p for _, fs in todo for p in fs.pairs]
        mo = list(ex.map(lambda p: motion.pair_motion(*p) if p else (np.nan, np.nan, np.nan), pairs))
    out: dict[str, dict] = {}
    for a, fs in todo:
        m = owners == a.id
        idx = np.where(m)[0]
        mm = np.array([mo[i] for i in idx], np.float32).reshape(-1, 3)
        if a.kind == "photo":
            mm[:] = 0.0
        d = {
            "emb": emb[m].astype(np.float32), "aesthetic": aes[m].astype(np.float32),
            **{k: np.array([q[i][k] for i in idx], np.float32) for k in quality.QUALITY_KEYS},
            "motion": mm[:, 0], "gdx": mm[:, 1], "gdy": mm[:, 2],
            "face_count": np.array([fc[i][0] for i in idx], np.float32),
            "face_area": np.array([fc[i][1] for i in idx], np.float32),
            "face_cx": np.array([fc[i][2] for i in idx], np.float32),
            "face_cy": np.array([fc[i][3] for i in idx], np.float32),
            "times": np.array(fs.times, np.float32),
        }
        cache.save_npz(_cache_file(a, ".npz"), **d)
        cache.write_json(_cache_file(a, ".json"), {"asset_hash": a.hash, "n_frames": len(fs.times),
                                                   "feature_version": get_config().features.version})
        out[a.id] = d
    return out


def _nanmean(x: np.ndarray, default: float = 0.0) -> float:
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else default


def _nanmedian(x: np.ndarray, default: float = 0.0) -> float:
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else default


def aggregate(c: Candidate, fs: FrameSet, f: dict[str, np.ndarray]) -> dict[str, float | np.ndarray]:
    idx = frame_indices(c, fs)
    e = f["emb"][idx].mean(0)
    e = e / max(np.linalg.norm(e), 1e-8)
    mid = 0.0 if c.kind == "photo" else (c.src_start + c.src_end) / 2
    k = idx[int(np.argmin([abs(fs.times[i] - mid) for i in idx]))]
    is_video = c.kind == "video"
    return {
        "emb": e.astype(np.float32),
        "aesthetic": _nanmean(f["aesthetic"][idx], 5.0),
        **{key: _nanmedian(f[key][idx]) for key in quality.QUALITY_KEYS},
        "motion": _nanmean(f["motion"][idx]) if is_video else 0.0,
        "shake": motion.shake(f["gdx"][idx], f["gdy"][idx]) if is_video else 0.0,
        "face_count": float(np.nanmax(f["face_count"][idx])),
        "face_area": float(np.nanmax(f["face_area"][idx])),
        "face_cx": float(f["face_cx"][k]), "face_cy": float(f["face_cy"][k]),
    }


def extract(assets: list[Asset], candidates: list[Candidate], frames: dict[str, FrameSet],
            progress: Callable[[float, str], None] | None = None) -> Features:
    per_asset: dict[str, dict] = {}
    todo = []
    for a in assets:
        z = load_asset(a, frames[a.id])
        if z is None:
            todo.append((a, frames[a.id]))
        else:
            per_asset[a.id] = z
    if todo:
        log.info("extracting features for %d/%d assets", len(todo), len(assets))
        per_asset.update(compute_assets(todo, progress))
    rows = [aggregate(c, frames[c.asset_id], per_asset[c.asset_id]) for c in candidates]
    ids = [c.id for c in candidates]
    arr: dict[str, np.ndarray] = {"emb": np.stack([r["emb"] for r in rows]) if rows else np.zeros((0, 768), np.float32)}
    for key in rows[0] if rows else []:
        if key != "emb":
            arr[key] = np.array([r[key] for r in rows], np.float32)
    kinds = [c.kind for c in candidates]
    asset_ids = [c.asset_id for c in candidates]
    is_video = np.array([k == "video" for k in kinds])
    arr["is_video"] = is_video
    arr["mood_arousal_raw"], arr["mood_valence_raw"] = mood.mood_raw(arr["emb"])
    arr["visual_arousal"], arr["visual_valence"] = mood.combine(arr, is_video)
    arr["dup_cluster"] = dedup.dup_clusters(arr["emb"], asset_ids, kinds)
    table = FeatureTable(candidate_ids=ids, tags=tags.top_tags(arr["emb"], ids),
                         feature_version=get_config().features.version)
    frame_view = {aid: {k: per_asset[aid][k] for k in ("times", "emb", "aesthetic", "motion")} for aid in per_asset}
    if progress:
        progress(1.0, f"features for {len(ids)} candidates")
    return Features(table=table, arrays=arr, frames=frame_view, kinds=kinds, asset_ids=asset_ids)
