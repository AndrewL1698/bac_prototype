"""Download small pretrained weights into DATA_ROOT/weights/ (shared, atomic, idempotent)."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from montage import paths
from montage.config import get_config

URLS = {
    "sac+logos+ava1-l14-linearMSE.pth":
        "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def ensure(name: str, min_bytes: int = 10_000) -> Path:
    """Return DATA_ROOT/weights/<name>, downloading it first if missing."""
    target = paths.weights_dir() / name
    if target.exists() and target.stat().st_size >= min_bytes:
        return target
    if name not in URLS:
        raise FileNotFoundError(f"{target} missing and no download URL known")

    def fetch(tmp: Path) -> None:
        with urllib.request.urlopen(URLS[name], timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if tmp.stat().st_size < min_bytes:
            raise IOError(f"download of {name} too small ({tmp.stat().st_size} bytes)")

    return paths.atomic_write(target, fetch)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_verified(target: Path, urls: list[str], sha256: str, local: list[Path] = (),
                   timeout: float = 120) -> tuple[Path, str]:
    """Return (target, source) where target's content has the given sha256.

    Order: target itself, then local copies (e.g. a torch hub cache), then each URL. A source that
    fails or hashes wrong is skipped; the first good one is written to target atomically.
    source is "cache", "local:<path>" or the URL used. Raises IOError listing every failure.
    """
    target = Path(target)
    if target.exists() and sha256_of(target) == sha256:
        return target, "cache"
    errors: list[str] = []

    def check(tmp: Path) -> None:
        got = sha256_of(tmp)
        if got != sha256:
            raise IOError(f"sha256 mismatch ({got[:12]}...)")

    for p in local:
        p = Path(p)
        if not p.exists():
            continue
        try:
            def copy(tmp: Path, p: Path = p) -> None:
                shutil.copyfile(p, tmp)
                check(tmp)
            return paths.atomic_write(target, copy), f"local:{p}"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{p}: {e}")
    for url in urls:
        try:
            def fetch(tmp: Path, url: str = url) -> None:
                with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                check(tmp)
            return paths.atomic_write(target, fetch), url
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")
    raise IOError(f"no valid copy of {target.name}: " + "; ".join(errors or ["no sources"]))


def aesthetic_path() -> Path:
    return ensure(get_config().features.aesthetic_weights)


def yunet_path() -> Path:
    return ensure(get_config().features.yunet_model)
