"""Features for SumMe/TVSum windows with the INFERENCE pipeline, unchanged (spec 10 section 3).

    python scripts/run_job.py --name vsum_feats -- python -m training.importance.extract_clip

Each dataset video goes through montage.candidates.build_media + montage.features.extract exactly as an upload
would (2 fps frames, 3 s / 1.5 s windows, CLIP, aesthetic, quality, faces), one video at a time with the
ingest limits lifted (a single long video is not an "album"). Feature caches are keyed by file hash, so the
app never mixes them up. Window label = mean normalized label of picks inside the window; windows without
picks are dropped. Writes DATA_ROOT/datasets/importance/windows.npz: X (ranker_features), y, group, dataset,
heuristic (spec 04 H within each video), aesthetic, t_mid. Resumable per video.
"""

from __future__ import annotations

import json

import numpy as np

from montage import cache, config, jobs, paths
from montage.importance.heuristic import heuristic_scores
from montage.importance.ranker import ranker_features
from montage.log import get_logger
from training.common import datasets_dir
from training.importance.labels import build_mapping, load_labels

log = get_logger("vsum_feats")
LIFT = {"ingest.min_assets": 1, "ingest.max_total_video_s": 1e9, "ingest.max_per_video_s": 1e9}


def video_features(video: str):
    """(candidates, Features) for one video via the app's own code path."""
    from montage.candidates import build_media
    from montage.features.extract import extract
    with config.override(LIFT):
        ms = build_media([video])
        return ms.candidates, extract(ms.assets, ms.candidates, ms.frames)


def window_rows(vl, cands, f) -> dict | None:
    t0 = np.array([c.src_start for c in cands], float)
    t1 = np.array([c.src_end for c in cands], float)
    ys = []
    for a, b in zip(t0, t1):
        m = (vl.pick_times >= a) & (vl.pick_times <= b)
        ys.append(vl.score[m].mean() if m.any() else np.nan)
    y = np.array(ys)
    keep = ~np.isnan(y)
    if keep.sum() < 3:
        return None
    H = heuristic_scores(f.arrays, f.table.tags, f.ids)  # z-scores within this video, as at inference
    X = ranker_features(f.arrays)
    return {"X": X[keep].astype(np.float32), "y": y[keep].astype(np.float32), "heuristic": H[keep].astype(np.float32),
            "aesthetic": np.asarray(f.arrays["aesthetic"], np.float32)[keep], "t_mid": ((t0 + t1) / 2)[keep].astype(np.float32)}


def main() -> int:
    paths.load_colab_env()
    mapping = build_mapping()
    labels = load_labels(mapping)
    per_video = datasets_dir("importance") / "per_video"
    per_video.mkdir(exist_ok=True)
    for i, vl in enumerate(labels):
        out = per_video / f"{vl.key.replace('/', '__')}.npz"
        if not out.exists():
            cands, f = video_features(vl.video)
            rows = window_rows(vl, cands, f)
            if rows is None:
                log.warning("%s: fewer than 3 labeled windows; skipped", vl.key)
                cache.write_json(out.with_suffix(".skip.json"), {"reason": "few labeled windows"})
                continue
            cache.save_npz(out, **rows)
        jobs.report((i + 1) / len(labels), f"{i + 1}/{len(labels)} videos ({vl.key})")
    parts = {k: [] for k in ("X", "y", "heuristic", "aesthetic", "t_mid", "group", "dataset")}
    for vl in labels:
        z = cache.load_npz(per_video / f"{vl.key.replace('/', '__')}.npz")
        if z is None:
            continue
        n = len(z["y"])
        for k in ("X", "y", "heuristic", "aesthetic", "t_mid"):
            parts[k].append(z[k])
        parts["group"].append(np.array([vl.key] * n))
        parts["dataset"].append(np.array([vl.dataset] * n))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    cache.save_npz(datasets_dir("importance") / "windows.npz", **data)
    summary = {"videos": len(set(data["group"])), "windows": int(len(data["y"])),
               "by_dataset": {d: int((data["dataset"] == d).sum()) for d in ("summe", "tvsum")}}
    paths.atomic_write(datasets_dir("importance") / "windows.json", json.dumps(summary, indent=1))
    print("windows:", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
