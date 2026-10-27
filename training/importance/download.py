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


ARCHIVE_EXT = (".zip", ".tar", ".tgz", ".tar.gz")


def _tag(name: str, default: str = "") -> str:
    n = name.lower()
    return "summe" if "summe" in n else "tvsum" if "tvsum" in n else default


def _is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_EXT)


def extract_nested(archive: Path, out: Path, tag: str = "", depth: int = 0) -> int:
    """Walk zip/tar archives recursively (the Zenodo tar holds SumMe.zip and tvsum50_ver_1_1.tar); keep videos
    as raw/<dataset>/videos/<name> and h5 label files as raw/<name>. Returns the number of files kept."""
    import zipfile
    kept = 0
    scratch = paths.scratch_dir("downloads") / f"nested_{depth}"
    scratch.mkdir(parents=True, exist_ok=True)
    # Choose the reader by extension: zipfile.is_zipfile() is True for a tar that happens to END with an
    # embedded zip (this tar does), which would silently read only that inner zip.
    if archive.name.lower().endswith(".zip"):
        zf = zipfile.ZipFile(archive)
        entries = [(i.filename, i.file_size, lambda i=i: zf.open(i)) for i in zf.infolist() if not i.is_dir()]
    else:
        tf = tarfile.open(archive)
        entries = [(m.name, m.size, lambda m=m: tf.extractfile(m)) for m in tf.getmembers() if m.isfile()]
    for k, (name, size, opener) in enumerate(entries):
        base = Path(name).name
        if base.startswith("._") or "__MACOSX" in name:
            continue
        t = _tag(name, tag)
        suffix = Path(base).suffix.lower()
        if _is_archive(base):
            tmp = scratch / base
            with opener() as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 22):
                    dst.write(chunk)
            kept += extract_nested(tmp, out, t, depth + 1)
            tmp.unlink(missing_ok=True)
        elif suffix in VIDEO_EXT or suffix == ".h5":
            target = out / base if suffix == ".h5" else out / (t or "unknown") / "videos" / base
            if not target.exists() or target.stat().st_size != size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with opener() as src:
                    paths.atomic_write(target, src.read())
            kept += 1
            if kept % 5 == 0:
                jobs.report(0.7 + 0.29 * min(1.0, kept / 77), f"extracted {kept} files ({t})")
    return kept


def main() -> int:
    paths.load_colab_env()
    out = raw_dir()
    done_marker = out / "_extracted.json"
    if not done_marker.exists():
        tar = download(URL, paths.scratch_dir("downloads") / "vsum_datasets.tar", 0.0, 0.7, "SumMe/TVSum")
        n = extract_nested(tar, out)
        files = [p for p in out.rglob("*") if p.is_file() and p.name != done_marker.name]
        summary = {"kept": n, "n_files": len(files), "by_ext": dict(Counter(p.suffix.lower() for p in files)),
                   "videos": {d: len(list((out / d / "videos").glob("*"))) for d in ("summe", "tvsum")},
                   "h5": [p.name for p in files if p.suffix == ".h5"]}
        paths.atomic_write(done_marker, json.dumps(summary, indent=1))
    summary = json.loads(done_marker.read_text())
    print("SumMe/TVSum extracted:", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
