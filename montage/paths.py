"""Filesystem layout: DATA_ROOT (shared Drive on Colab), SCRATCH (local SSD), per-user namespaces.

Values resolve on every call from env, so tests and job processes can change them.
`paths.DATA_ROOT`, `paths.SCRATCH`, `paths.USER` also work as module attributes.
"""

from __future__ import annotations

import os
import secrets
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_ENV_FILE = Path("/content/montage_env.sh")


def is_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content/sample_data").exists()


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_root() -> Path:
    return Path(os.environ.get("MONTAGE_DATA_ROOT") or REPO_ROOT / "data")


def scratch() -> Path:
    env = os.environ.get("MONTAGE_SCRATCH")
    if env:
        return Path(env)
    return Path("/content/scratch") if is_colab() else data_root() / "scratch"


def user() -> str:
    return (os.environ.get("MONTAGE_USER") or "anon").strip().lower() or "anon"


def run_dir(run_id: str) -> Path:
    return _mk(data_root() / "runs" / run_id)


def cache_dir(kind: str) -> Path:
    return _mk(data_root() / "cache" / kind)


def scratch_dir(kind: str) -> Path:
    return _mk(scratch() / kind)


def weights_dir() -> Path:
    return _mk(data_root() / "weights")


def sessions_dir() -> Path:
    return _mk(data_root() / "sessions")


def jobs_dir(for_user: str | None = None) -> Path:
    return _mk(data_root() / "jobs" / (for_user or user()))


def logs_dir(for_user: str | None = None) -> Path:
    return _mk(data_root() / "logs" / (for_user or user()))


def new_run_id() -> str:
    return f"{user()}_{datetime.now():%Y%m%d-%H%M%S}_{secrets.token_hex(2)}"


def atomic_write(path: str | Path, data: bytes | str | Callable[[Path], None]) -> Path:
    """Write via a temp file in the same directory, then os.replace (safe on the shared Drive).

    `data` may be bytes, str (utf-8), or a callable that writes to the temp path it is given.
    The temp name keeps the target's suffix so writers like np.savez_compressed do not rename it.
    On any exception the temp file is removed and the target is untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if callable(data):
            data(tmp)
        else:
            tmp.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse `export KEY=VALUE` lines (as written by colab_bootstrap.py)."""
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def load_colab_env(path: str | Path = COLAB_ENV_FILE) -> dict[str, str]:
    """Load /content/montage_env.sh into os.environ (no-op if missing). Variables already set in the
    environment win (tests and callers can point at another DATA_ROOT/user). Returns what was applied."""
    applied = {}
    for k, v in parse_env_file(path).items():
        if k == "PATH":
            cur = os.environ.get("PATH", "")
            new = v.replace("$PATH", cur) if "$PATH" in v else v
            if new != cur:
                os.environ["PATH"] = applied[k] = new
        elif k not in os.environ:
            os.environ[k] = applied[k] = v
    return applied


def __getattr__(name: str):
    if name == "DATA_ROOT":
        return data_root()
    if name == "SCRATCH":
        return scratch()
    if name == "USER":
        return user()
    raise AttributeError(name)
