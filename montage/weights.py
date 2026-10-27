"""Download small pretrained weights into DATA_ROOT/weights/ (shared, atomic, idempotent)."""

from __future__ import annotations

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


def aesthetic_path() -> Path:
    return ensure(get_config().features.aesthetic_weights)


def yunet_path() -> Path:
    return ensure(get_config().features.yunet_model)
