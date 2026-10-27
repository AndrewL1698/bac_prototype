"""SumMe (25) + TVSum (50) original videos and the standard h5 label files (Zenodo record 4884870).

    python scripts/run_job.py --name vsum_dl -- python -m training.importance.download

Downloads datasets.tar (3.2 GB) to scratch, extracts videos and h5 files to DATA_ROOT/datasets/importance/raw/.
Idempotent. The h5 GoogLeNet features are never used (labels and frame indices only).
"""

from __future__ import annotations

import json
import tarfile
from collections import Counter
from pathlib import Path

from montage import jobs, paths
from montage.log import get_logger
from training.common import datasets_dir, download

log = get_logger("vsum")
URL = "https://zenodo.org/api/records/4884870/files/datasets.tar/content"
VIDEO_EXT = {".mp4", ".webm", ".avi", ".mkv", ".mov", ".flv"}


def raw_dir() -> Path:
    return datasets_dir("importance") / "raw"


def main() -> int:
    paths.load_colab_env()
    out = raw_dir()
    done_marker = out / "_extracted.json"
    if not done_marker.exists():
        tar = download(URL, paths.scratch_dir("downloads") / "vsum_datasets.tar", 0.0, 0.7, "SumMe/TVSum")
        with tarfile.open(tar) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            log.info("tar: %d files; extensions %s", len(members),
                     Counter(Path(m.name).suffix.lower() for m in members).most_common(8))
            keep = [m for m in members if Path(m.name).suffix.lower() in VIDEO_EXT | {".h5", ".mat", ".csv", ".tsv"}]
            for i, m in enumerate(keep):
                rel = Path(*Path(m.name).parts[-3:])  # keep a little of the folder structure
                target = out / rel
                if not target.exists() or target.stat().st_size != m.size:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(m)
                    paths.atomic_write(target, src.read())
                if i % 5 == 0:
                    jobs.report(0.7 + 0.3 * i / len(keep), f"extracting {i}/{len(keep)}")
        files = [p for p in out.rglob("*") if p.is_file()]
        summary = {"n_files": len(files), "by_ext": dict(Counter(p.suffix.lower() for p in files)),
                   "h5": [str(p.relative_to(out)) for p in files if p.suffix == ".h5"]}
        paths.atomic_write(done_marker, json.dumps(summary, indent=1))
    summary = json.loads(done_marker.read_text())
    print("SumMe/TVSum extracted:", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
