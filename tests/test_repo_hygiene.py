"""The repo is public: no media, data, weights, large files, or secrets may be tracked.

Secret patterns require a realistic token body after the prefix, so docs that merely
mention a prefix (CLAUDE.md does) pass, while a pasted key fails.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAX_BYTES = 2 * 1024 * 1024
IMG_ALLOW_BYTES = 500 * 1024
BAD_EXT = {".jpg", ".jpeg", ".png", ".heic", ".mp4", ".mov", ".wav", ".mp3", ".m4a",
           ".npz", ".pt", ".pth", ".onnx", ".zip"}
# Built by concatenation so this file does not match itself.
SECRET_RES = [re.compile(p) for p in (
    "sk" + r"-ant-[A-Za-z0-9_\-]{20,}",
    "gh" + r"p_[A-Za-z0-9]{30,}",
    "github" + r"_pat_[A-Za-z0-9_]{20,}",
    "AI" + r"za[0-9A-Za-z_\-]{30,}",
)]


def tracked_files(root: Path) -> list[str] | None:
    if shutil.which("git") is None:
        return None
    r = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True)
    if r.returncode != 0:
        return None
    return [f for f in r.stdout.decode("utf-8", "replace").split("\0") if f]


def problems(root: Path, files: list[str]) -> list[str]:
    out = []
    for rel in files:
        p = root / rel
        if not p.is_file():
            continue
        size = p.stat().st_size
        ext = p.suffix.lower()
        allowed_img = rel.startswith("docs/img/") and ext == ".png" and "/" not in rel[len("docs/img/"):] \
            and size <= IMG_ALLOW_BYTES
        if ext in BAD_EXT and not allowed_img:
            out.append(f"{rel}: media/data file type {ext}")
        if size > MAX_BYTES:
            out.append(f"{rel}: larger than 2 MB ({size} bytes)")
        if size <= MAX_BYTES:
            text = p.read_bytes().decode("utf-8", "ignore")
            for rx in SECRET_RES:
                if rx.search(text):
                    out.append(f"{rel}: contains a secret-like string ({rx.pattern[:6]}...)")
    return out


def test_repo_hygiene():
    files = tracked_files(REPO)
    if files is None:
        pytest.skip("git unavailable")
    bad = problems(REPO, files)
    assert not bad, "public-repo hygiene violations:\n" + "\n".join(bad)


def test_hygiene_detects_violations(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "x.mp4").write_bytes(b"\x00" * 10)
    (tmp_path / "leak.txt").write_text("key = " + "sk" + "-ant-" + "a1B2" * 10 + "\n")
    (tmp_path / "ok.md").write_text("mention of the sk" + "-ant- prefix is fine\n")
    (tmp_path / "docs" / "img").mkdir(parents=True)
    (tmp_path / "docs" / "img" / "slide.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
    subprocess.run(["git", "add", "-A", "-f"], cwd=tmp_path, check=True)
    bad = problems(tmp_path, tracked_files(tmp_path))
    joined = "\n".join(bad)
    assert "x.mp4" in joined
    assert "leak.txt" in joined
    assert "ok.md" not in joined
    assert "slide.png" not in joined
