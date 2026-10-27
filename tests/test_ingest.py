"""Phase 1 acceptance: ingest and candidates."""

from __future__ import annotations

import shutil
import subprocess
import time

import numpy as np
import pytest
from PIL import Image

from montage.candidates.windows import window_starts
from montage.types import UserError


def _photos(d, n, start=0):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        rgb = np.random.default_rng(i).integers(0, 255, (60, 80, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(d / f"p{i:02d}.jpg")
    return d


def test_ingest_counts(tmp_path):
    from montage.ingest.assets import ingest, ingest_with_notes
    with pytest.raises(UserError, match="at least 10"):
        ingest([_photos(tmp_path / "few", 5)])
    d = _photos(tmp_path / "ok", 12)
    (d / "notes.txt").write_text("not media")
    assets, notes = ingest_with_notes([d])
    assert len(assets) == 12
    assert any("notes.txt" in n for n in notes)
    assert [a.id for a in assets] == [f"a{i:03d}" for i in range(12)]


def test_ingest_too_many(tmp_path):
    from montage import config
    from montage.ingest.assets import ingest
    with config.override({"ingest.max_assets": 11}):
        with pytest.raises(UserError, match="At most"):
            ingest([_photos(tmp_path / "many", 12)])


def test_window_starts_grid():
    w = window_starts(0.3, 11.7, 3.0, 1.5, 1.0)
    assert w[0] == (0.3, 3.3)
    assert all(abs((e - s) - 3.0) < 1e-6 for s, e in w)
    assert all(abs(w[i + 1][0] - w[i][0] - 1.5) < 1e-6 for i in range(len(w) - 1))
    assert w[-1][1] <= 11.7 + 1e-6 and len(w) == 6
    assert window_starts(2.0, 3.5, 3.0, 1.5, 1.0) == [(2.0, 3.5)]
    assert window_starts(2.0, 2.5, 3.0, 1.5, 1.0) == []


# ---------------- media tests (Colab: FFmpeg + generated fixtures) ----------------

@pytest.mark.media
def test_probe_rotation(fixtures):
    from montage.ingest.probe import probe_video
    m = probe_video(fixtures / "vid_rotated.mov")
    assert m["rotation"] == 90
    assert m["width"] < m["height"] and (m["width"], m["height"]) == (720, 1280)


@pytest.mark.media
def test_probe_hdr_flag(fixtures):
    from montage.ingest.probe import probe_video
    assert probe_video(fixtures / "vid_hdr.mp4")["is_hdr"] is True
    assert probe_video(fixtures / "vid_static_landscape.mp4")["is_hdr"] is False


@pytest.mark.media
def test_probe_photo_exif(fixtures):
    from montage.ingest.probe import probe_photo
    m = probe_photo(fixtures / "photo_04_stripes_portrait.jpg")
    assert (m["width"], m["height"]) == (1200, 1600)
    assert m["taken_at"].startswith("2026-07-04")


@pytest.mark.media
def test_frames_rate(fixtures):
    from montage.ingest.assets import ingest
    from montage.ingest.frames import extract_frames
    assets = ingest([fixtures])
    a = next(x for x in assets if x.filename == "vid_motion_landscape.mp4")
    fs = extract_frames(a)
    assert abs(len(fs.times) - 24) <= 1
    assert sum(p is not None for p in fs.pairs) == len(fs.times)
    with Image.open(fs.paths[0]) as im:
        assert min(im.size) == 448
    with Image.open(fs.pairs[0][0]) as im:
        assert min(im.size) == 320


@pytest.mark.media
def test_frames_rotated_upright(fixtures):
    from montage.ingest.assets import ingest
    from montage.ingest.frames import extract_frames
    a = next(x for x in ingest([fixtures]) if x.filename == "vid_rotated.mov")
    with Image.open(extract_frames(a).paths[0]) as im:
        assert im.width < im.height  # autorotated portrait


@pytest.mark.media
def test_scenes_twoscenes(fixtures):
    from montage.candidates.scenes import detect_scenes
    from montage.ingest.assets import ingest
    from montage.ingest.frames import extract_frames
    a = next(x for x in ingest([fixtures]) if x.filename == "vid_twoscenes.mp4")
    scenes = detect_scenes(a, extract_frames(a))
    assert len(scenes) == 2, scenes
    assert abs(scenes[1][0] - 5.0) <= 0.3


@pytest.mark.media
def test_windows(fixtures):
    from montage.candidates import build_media
    ms = build_media([fixtures])
    a = next(x for x in ms.assets if x.filename == "vid_motion_landscape.mp4")
    wins = [c for c in ms.candidates if c.asset_id == a.id]
    assert len(wins) == 6
    for c in wins:
        assert abs((c.src_end - c.src_start) - 3.0) < 1e-6
        assert 0.3 - 1e-6 <= c.src_start and c.src_end <= a.duration - 0.3 + 1e-6
        assert c.src_max_end >= c.src_end and c.frame_times
    starts = [c.src_start for c in wins]
    assert all(abs(starts[i + 1] - starts[i] - 1.5) < 1e-6 for i in range(len(starts) - 1))
    ids = [c.id for c in ms.candidates]
    assert len(ids) == len(set(ids))
    assert all(shutil.os.path.exists(c.thumb_path) for c in ms.candidates)
    photos = [c for c in ms.candidates if c.kind == "photo"]
    assert len(photos) == 8 and all(c.id == c.asset_id for c in photos)


@pytest.mark.media
def test_cache_hit(fixtures, monkeypatch, fresh_root):
    from montage.ingest import assets as A
    from montage.ingest import probe
    A.ingest([fixtures])
    calls = []
    real = subprocess.run

    def counting(cmd, *a, **k):
        if "ffprobe" in str(cmd[0]):
            calls.append(cmd)
        return real(cmd, *a, **k)

    monkeypatch.setattr(probe.subprocess, "run", counting)
    A.ingest([fixtures])
    assert calls == []


@pytest.mark.media
def test_ingest_timing(fixtures, fresh_root):
    from montage.candidates import build_media
    t0 = time.time()
    ms = build_media([fixtures])
    dt = time.time() - t0
    assert len(ms.assets) >= 15
    assert dt < 30, f"cold ingest + candidates took {dt:.1f}s"
