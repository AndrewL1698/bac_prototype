from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _isolated_data_root(tmp_path_factory):
    """Tests never touch the real DATA_ROOT (the shared Drive on Colab)."""
    root = tmp_path_factory.mktemp("data_root")
    old = {k: os.environ.get(k) for k in ("MONTAGE_DATA_ROOT", "MONTAGE_SCRATCH", "MONTAGE_USER", "MONTAGE_JOB_NAME")}
    os.environ["MONTAGE_DATA_ROOT"] = str(root)
    os.environ["MONTAGE_SCRATCH"] = str(root / "scratch")
    os.environ["MONTAGE_USER"] = "test"
    os.environ.pop("MONTAGE_JOB_NAME", None)
    from montage import config
    config.reset()
    yield root
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def fresh_root(tmp_path, monkeypatch):
    """A per-test empty DATA_ROOT."""
    monkeypatch.setenv("MONTAGE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("MONTAGE_SCRATCH", str(tmp_path / "scratch"))
    return tmp_path / "data"


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def fixtures() -> Path:
    if not (FIXTURES / "truth.json").exists():
        pytest.skip("fixtures not generated (python scripts/make_fixtures.py)")
    return FIXTURES


@pytest.fixture(scope="session")
def truth(fixtures) -> dict:
    return json.loads((fixtures / "truth.json").read_text())


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
