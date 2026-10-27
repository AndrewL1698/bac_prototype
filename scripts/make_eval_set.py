"""Build the evaluation set in DATA_ROOT/eval/: albums split by capture time + songs.

    python scripts/run_job.py --name eval_set -- python scripts/make_eval_set.py --src DIR [--exclude DIR]
        [--albums 2] [--prefix NAME] [--song FILE ...] [--deam-energetic]

Albums: media from --src not already present (by file name) in --exclude, sorted by capture time and split into
--albums contiguous chunks (eval/albums/<prefix>_<k>/). Songs: copied into eval/songs/; --deam-energetic adds
the DEAM full song (ids >= 2001, Creative Commons) with the highest mean annotated arousal. Originals untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import jobs, paths  # noqa: E402
from montage.config import get_config  # noqa: E402
from montage.ingest.assets import _probe_one, collect  # noqa: E402


def _base(name: str) -> str:
    return name.lower()


def deam_energetic(min_s: float = 60.0) -> Path | None:
    from training.emotion.download_deam import annotations_path, audio_dir
    songs = json.loads(annotations_path().read_text())["songs"]
    best = None
    for sid, s in songs.items():
        if int(sid) < 2001:
            continue
        mp3 = audio_dir() / f"{sid}.mp3"
        if not mp3.exists():
            continue
        r = subprocess.run([get_config().render.ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(mp3)], capture_output=True, text=True)
        dur = float(r.stdout.strip() or 0)
        a = float(np.mean(s["arousal"]))
        if dur >= min_s and (best is None or a > best[0]):
            best = (a, mp3, dur)
    if best:
        print(f"DEAM energetic pick: song {best[1].stem} (mean arousal {best[0]:+.2f}, {best[2]:.0f} s)")
    return best[1] if best else None


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--albums", type=int, default=2)
    ap.add_argument("--prefix", default="album")
    ap.add_argument("--song", action="append", default=[])
    ap.add_argument("--deam-energetic", action="store_true")
    args = ap.parse_args(argv)
    root = paths.data_root() / "eval"
    skip = {_base(p.name) for p in Path(args.exclude).rglob("*") if p.is_file()} if args.exclude else set()
    files = [f for f in collect([Path(args.src)]) if _base(f.name) not in skip]
    probed = []
    for i, f in enumerate(files):
        _, _, d = _probe_one(f)
        probed.append((d.get("taken_at") or "", f.name, f, d["kind"]))
        jobs.report(0.4 * (i + 1) / len(files), f"probed {i + 1}/{len(files)}")
    probed.sort()
    chunks = np.array_split(np.arange(len(probed)), args.albums)
    copied = 0
    for k, idx in enumerate(chunks):
        dest = root / "albums" / f"{args.prefix}_{k + 1}"
        dest.mkdir(parents=True, exist_ok=True)
        for j in idx:
            _, name, f, _ = probed[j]
            t = dest / name
            if not t.exists() or t.stat().st_size != f.stat().st_size:
                shutil.copy2(f, t)
            copied += 1
            jobs.report(0.4 + 0.5 * copied / len(probed), f"copied {copied}/{len(probed)}")
        kinds = [probed[j][3] for j in idx]
        print(f"{dest.name}: {len(idx)} items ({kinds.count('photo')} photos, {kinds.count('video')} videos), "
              f"{probed[idx[0]][0][:10]} .. {probed[idx[-1]][0][:10]}")
    songs = [Path(s) for s in args.song]
    if args.deam_energetic:
        e = deam_energetic()
        if e:
            songs.append(e)
    (root / "songs").mkdir(parents=True, exist_ok=True)
    for s in songs:
        name = s.name if not s.stem.isdigit() else f"deam_{s.stem}_energetic{s.suffix}"
        shutil.copy2(s, root / "songs" / name)
    print("songs:", sorted(p.name for p in (root / "songs").iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
