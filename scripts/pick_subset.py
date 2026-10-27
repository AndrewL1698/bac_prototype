"""Copy a demo-sized subset of a large album: the longest videos up to --max-video-s (more footage per clip;
phone albums are full of 2-3 s live-photo snippets) plus photos spread evenly over time.

    python scripts/run_job.py --name pick -- python scripts/pick_subset.py --src "DIR" --dest "DIR" [--n 60] [--max-videos 20]

Originals are never modified. Probing fills the probe cache, so ingesting the copies later is fast.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import jobs, paths  # noqa: E402
from montage.ingest.assets import _probe_one, collect  # noqa: E402


def pick(items: list[tuple[Path, dict]], n: int, max_videos: int, min_video_s: float = 2.0,
         max_video_s: float = 60.0) -> list[Path]:
    """items: (path, probe dict with kind, duration, taken_at). Returns the chosen paths in time order."""
    videos = sorted((x for x in items if x[1]["kind"] == "video"
                     and min_video_s <= (x[1].get("duration") or 0) <= max_video_s),
                    key=lambda x: -x[1]["duration"])[:max_videos]
    photos = sorted((x for x in items if x[1]["kind"] == "photo"), key=lambda x: (x[1].get("taken_at") or "", x[0].name))
    k = min(len(photos), max(0, n - len(videos)))
    idx = np.unique(np.round(np.linspace(0, len(photos) - 1, k)).astype(int)) if k else []
    chosen = videos + [photos[i] for i in idx]
    return [p for p, _ in sorted(chosen, key=lambda x: (x[1].get("taken_at") or "", x[0].name))]


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-videos", type=int, default=20)
    ap.add_argument("--max-video-s", type=float, default=60.0)
    args = ap.parse_args(argv)
    src, dest = Path(args.src), Path(args.dest)
    files = collect([src])
    jobs.report(0.02, f"probing {len(files)} files", force=True)
    done = []

    def probe(f: Path):
        p, _, d = _probe_one(f)
        done.append(p)
        jobs.report(0.02 + 0.6 * len(done) / len(files), f"probed {len(done)}/{len(files)}")
        return p, d

    with ThreadPoolExecutor(max_workers=8) as ex:
        items = list(ex.map(probe, files))
    chosen = pick(items, args.n, args.max_videos, max_video_s=args.max_video_s)
    dest.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for i, p in enumerate(chosen):
        name = p.name if p.name not in used else f"{p.stem}_{i}{p.suffix}"
        used.add(name)
        shutil.copy2(p, dest / name)
        jobs.report(0.62 + 0.38 * (i + 1) / len(chosen), f"copied {i + 1}/{len(chosen)}")
    n_vid = sum(d["kind"] == "video" for p, d in items if p in set(chosen))
    total_v = sum(d.get("duration") or 0 for p, d in items if p in set(chosen) and d["kind"] == "video")
    dates = sorted(d["taken_at"] for p, d in items if p in set(chosen) and d.get("taken_at"))
    print(f"picked {len(chosen)} of {len(files)}: {len(chosen) - n_vid} photos, {n_vid} videos "
          f"({total_v:.0f} s of video); dates {dates[0][:10] if dates else '?'} .. {dates[-1][:10] if dates else '?'}; "
          f"copied to {dest} at {datetime.now():%H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
