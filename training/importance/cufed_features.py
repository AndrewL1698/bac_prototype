"""Features for CUFED photos with the INFERENCE pipeline, unchanged (phase 15, spec 15 section 4).

    python scripts/run_job.py --name cufed_feats -- python -m training.importance.cufed_features
    python -m training.importance.cufed_features --limit 5         # smoke run on 5 albums first

One album = one upload: montage.candidates.build_media + montage.features.extract on that album's photos, with
only the ingest count limits lifted. Albums are never batched together: ingest drops exact duplicate files within
a call (a photo shared by two albums would vanish from the second), and the heuristic baseline is a z-score within
the upload. One call per album reproduces both exactly as the app sees them.

Per album: X (ranker_features, identical to inference), y (min-max of the CUFED score over the photos that have
features), heuristic (spec 04 H within the album), aesthetic, photo, event. Resumable per album
(cufed/per_album/<album>.npz or .skip.json). Then DATA_ROOT/datasets/importance/cufed_photos.npz with keys
X, y, heuristic, aesthetic, group ("cufed/<album>"), dataset ("cufed"), event, photo: the core keys match
windows.npz, so training can concatenate the two.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from montage import cache, config, jobs, paths
from montage.config import get_config
from montage.importance.heuristic import heuristic_scores
from montage.importance.ranker import ranker_features
from montage.log import get_logger
from training.common import datasets_dir
from training.importance.cufed import cufed_dir, load_mapping

log = get_logger("cufed_feats")
LIFT = {"ingest.min_assets": 1, "ingest.max_assets": 10000}
KEYS = ("X", "y", "heuristic", "aesthetic", "group", "dataset", "event", "photo")


def per_album_dir() -> Path:
    d = cufed_dir() / "per_album"
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_of(album: str) -> Path:
    return per_album_dir() / f"{album}.npz"


def skip_of(album: str) -> Path:
    return per_album_dir() / f"{album}.skip.json"


def album_features(photo_paths: list[str]):
    """(MediaSet, Features) for one album via the app's own code path (one upload)."""
    from montage.candidates import build_media
    from montage.features.extract import extract
    with config.override(LIFT):
        ms = build_media(photo_paths)
        f = extract(ms.assets, ms.candidates, ms.frames)
    return ms, f


def _key(p: str) -> str:
    return os.path.normpath(str(p))


def album_rows(entry: dict, ms, f, min_photos: int) -> tuple[dict | None, str]:
    """(rows, "ok") for one album, or (None, reason). Photos ingest skipped (exact duplicate files) have no
    candidate and are dropped; y is re-normalized over the photos that remain."""
    by_path = {_key(p["path"]): p for p in entry["photos"]}
    src_of = {a.id: _key(a.src_path) for a in ms.assets}
    keep, scores, photos = [], [], []
    for i, c in enumerate(ms.candidates):
        p = by_path.get(src_of.get(c.asset_id, ""))
        if p is not None:
            keep.append(i)
            scores.append(p["score"])
            photos.append(p["photo"])
    if len(keep) < min_photos:
        return None, "few_photos_after_ingest"
    s = np.asarray(scores, float)
    if s.max() - s.min() <= 0:
        return None, "tied_scores"
    idx = np.asarray(keep)
    H = heuristic_scores(f.arrays, f.table.tags, f.ids)  # z-scores within this album, as at inference
    X = ranker_features(f.arrays)
    return {"X": X[idx].astype(np.float32), "y": ((s - s.min()) / (s.max() - s.min())).astype(np.float32),
            "heuristic": np.asarray(H, np.float32)[idx],
            "aesthetic": np.asarray(f.arrays["aesthetic"], np.float32)[idx],
            "photo": np.array(photos), "event": np.array(entry["events"][0])}, "ok"


def extract_albums(mapping: dict, limit: int | None = None, retry_errors: bool = False) -> dict:
    """Features for every album not done yet. Returns counts for this run."""
    cfg = get_config().importance.cufed
    albums = sorted(mapping["albums"])

    def pending(a: str) -> bool:
        if out_of(a).exists():
            return False
        skip = cache.read_json(skip_of(a))
        return skip is None or (retry_errors and skip.get("reason") == "error")

    todo = [a for a in albums if pending(a)]
    if limit:
        todo = todo[:limit]
    log.info("%d of %d albums need features (this run: %d)", sum(map(pending, albums)), len(albums), len(todo))
    streak: list[str] = []
    counts: Counter = Counter()
    for k, album in enumerate(todo, 1):
        entry = mapping["albums"][album]
        try:
            ms, f = album_features([p["path"] for p in entry["photos"]])
            rows, reason = album_rows(entry, ms, f, cfg.min_photos)
        except Exception as e:  # one bad album must not end a multi-hour job ...
            log.warning("%s: %s: %s", album, type(e).__name__, e)
            cache.write_json(skip_of(album), {"reason": "error", "error": f"{type(e).__name__}: {e}"[:500]})
            streak.append(album)
            counts["error"] += 1
            if len(streak) >= cfg.max_consecutive_failures:  # ... but a systemic failure must stop it
                for a in streak:  # these albums are probably fine: let the next run retry them
                    skip_of(a).unlink(missing_ok=True)
                raise SystemExit(f"{len(streak)} albums failed in a row (last: {type(e).__name__}: {e}); "
                                 "stopping. Fix the cause and rerun; finished albums are kept.")
            continue
        streak.clear()
        if rows is None:
            cache.write_json(skip_of(album), {"reason": reason})
            counts[reason] += 1
        else:
            cache.save_npz(out_of(album), **rows)
            skip_of(album).unlink(missing_ok=True)
            counts["ok"] += 1
        jobs.report(k / max(len(todo), 1), f"{k}/{len(todo)} albums")
    return dict(counts)


def combine(mapping: dict, out: Path | None = None) -> dict:
    """Concatenate finished albums into cufed_photos.npz (+ cufed_photos.json summary)."""
    parts: dict[str, list] = {k: [] for k in KEYS}
    skipped: Counter = Counter()
    pending = 0
    for album in sorted(mapping["albums"]):
        z = cache.load_npz(out_of(album))
        if z is None:
            skip = cache.read_json(skip_of(album))
            if skip:
                skipped[skip.get("reason", "?")] += 1
            else:
                pending += 1
            continue
        n = len(z["y"])
        for k in ("X", "y", "heuristic", "aesthetic", "photo"):
            parts[k].append(z[k])
        parts["group"].append(np.full(n, f"cufed/{album}"))
        parts["dataset"].append(np.full(n, "cufed"))
        parts["event"].append(np.full(n, str(z["event"])))
    if not parts["y"]:
        raise SystemExit("no CUFED album has features yet")
    data = {k: np.concatenate(v) for k, v in parts.items()}
    target = out or datasets_dir("importance") / "cufed_photos.npz"
    cache.save_npz(target, **data)
    summary = {"albums": int(len(set(data["group"]))), "photos": int(len(data["y"])),
               "albums_skipped": dict(skipped), "albums_pending": pending,
               "photos_by_event": dict(Counter(data["event"].tolist()).most_common())}
    paths.atomic_write(target.with_suffix(".json"), json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CUFED features through the app pipeline (spec 15 section 4)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--limit", type=int, default=None, help="process at most N albums this run")
    ap.add_argument("--retry-errors", action="store_true", help="retry albums skipped because of an error")
    a = ap.parse_args(argv)
    paths.load_colab_env()
    if a.set:
        config.set_overrides(a.set)
    mapping = load_mapping()
    counts = extract_albums(mapping, limit=a.limit, retry_errors=a.retry_errors)
    print("this run:", json.dumps(counts), flush=True)
    print("cufed_photos:", json.dumps(combine(mapping)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
