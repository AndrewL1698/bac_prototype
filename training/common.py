"""Shared helpers for training jobs: resumable downloads and dataset paths."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from montage import jobs, paths

CHUNK = 1 << 22


def datasets_dir(name: str) -> Path:
    d = paths.data_root() / "datasets" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except OSError:
        return None


def download(url: str, dest: Path, progress_lo: float = 0.0, progress_hi: float = 1.0, label: str = "") -> Path:
    """Resumable HTTP download (Range requests). Skips if dest already has the full size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    if dest.exists() and total and dest.stat().st_size == total:
        return dest
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:  # server ignored the range: restart
            have = 0
        with open(part, "ab" if have else "wb") as f:
            done = have
            while chunk := r.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if total:
                    jobs.report(progress_lo + (progress_hi - progress_lo) * done / total,
                                f"{label or dest.name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB")
    if total and part.stat().st_size != total:
        raise IOError(f"incomplete download of {dest.name}: {part.stat().st_size} of {total} bytes")
    part.replace(dest)
    return dest
