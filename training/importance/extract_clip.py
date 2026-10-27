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
LIFT = {"ingest.min_assets": 1, "ingest.max_assets": 1000, "ingest.max_total_video_s": 1e9,
        "ingest.max_per_video_s": 1e9}
BATCH = 8


def video_features(video: str):
    """(candidates, Features) for one video via the app's own code path."""
    return videos_features([video])[video]


def videos_features(videos: list[str]) -> dict:
    """{video: (candidates, Features)} for several videos in ONE pipeline call (frames extract in parallel,
    CLIP sees full batches). The ranker inputs are per-candidate and the heuristic is recomputed within each
    video, so batching does not change anything the ranker sees."""
    from types import SimpleNamespace
    from montage.candidates import build_media
    from montage.features.extract import extract
    with config.override(LIFT):
        ms = build_media(videos)
        f = extract(ms.assets, ms.candidates, ms.frames)
    by_path = {str(a.src_path): a.id for a in ms.assets}
    out = {}
    for v in videos:
        aid = by_path.get(str(v))
        idx = [i for i, c in enumerate(ms.candidates) if c.asset_id == aid]
        cands = [ms.candidates[i] for i in idx]
        ids = [c.id for c in cands]
        sub = SimpleNamespace(arrays={k: (np.asarray(a)[idx] if np.ndim(a) else a) for k, a in f.arrays.items()},
                              table=SimpleNamespace(tags={c: f.table.tags.get(c, []) for c in ids}), ids=ids)
        out[v] = (cands, sub)
    return out


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
    out_of = lambda vl: per_video / f"{vl.key.replace('/', '__')}.npz"  # noqa: E731
    todo = [vl for vl in labels if not out_of(vl).exists() and not out_of(vl).with_suffix(".skip.json").exists()]
    log.info("%d of %d videos still need features", len(todo), len(labels))
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        feats = videos_features([vl.video for vl in batch])
        for vl in batch:
            cands, f = feats[vl.video]
            rows = window_rows(vl, cands, f) if cands else None
            if rows is None:
                log.warning("%s: fewer than 3 labeled windows; skipped", vl.key)
                cache.write_json(out_of(vl).with_suffix(".skip.json"), {"reason": "few labeled windows"})
                continue
            cache.save_npz(out_of(vl), **rows)
        done = len(labels) - len(todo) + b + len(batch)
        jobs.report(done / len(labels), f"{done}/{len(labels)} videos")
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
