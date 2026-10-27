"""Content-addressed cache helpers. Every write is atomic (shared Drive)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from montage import paths

CHUNK = 1 << 20


def file_hash(path: str | Path, n: int = 16) -> str:
    """sha1 of file bytes (streamed), first n hex chars."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()[:n]


def text_hash(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:n]


def cache_path(kind: str, key: str, ext: str) -> Path:
    return paths.cache_dir(kind) / f"{key}{ext}"


def read_json(path: str | Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path: str | Path, obj: Any) -> Path:
    return paths.atomic_write(path, json.dumps(obj, indent=1, default=str))


def save_npz(path: str | Path, **arrays: Any) -> Path:
    import numpy as np
    return paths.atomic_write(path, lambda tmp: np.savez_compressed(tmp, **arrays))


def load_npz(path: str | Path) -> dict | None:
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (FileNotFoundError, OSError, ValueError):
        return None
