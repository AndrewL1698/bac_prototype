"""Phase 13 acceptance: per-shot render cache (key rules, reuse on recut, pruning)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from montage import config
from montage.render import shot_cache
from montage.types import Asset, KenBurns
from tests.test_render import _edl, _forty, _frame_at, _probe_frames, _shot

KB = KenBurns(zoom_from=1.0, zoom_to=1.05, pan_x=0.0, pan_y=0.0)


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    """Every test starts with an empty shot cache."""
    monkeypatch.setenv("MONTAGE_SCRATCH", str(tmp_path / "scratch"))


@pytest.fixture
def count_encodes(monkeypatch):
    """Counts how many shots are actually encoded (cache misses)."""
    calls = []
    real = shot_cache.render_shot

    def counting(shot, *a, **kw):
        calls.append(shot.slot_idx)
        return real(shot, *a, **kw)

    monkeypatch.setattr(shot_cache, "render_shot", counting)
    return calls


def _photo(h="p0") -> Asset:
    return Asset(id="a000", kind="photo", src_path="x.jpg", hash=h, width=1920, height=1080, filename="x.jpg")


def _video(h="v0") -> Asset:
    return Asset(id="a001", kind="video", src_path="x.mp4", hash=h, width=1920, height=1080, duration=10,
                 filename="x.mp4")


# ---------------- key rules (no media) ----------------

def test_key_ignores_timeline_position():
    a = _photo()
    k = shot_cache.shot_key(_shot(0, 0, 30, a, "native", None, KB), a, 1280, 720, 23)
    # the same shot moved to another slot / time in the cut is still the same encode
    assert shot_cache.shot_key(_shot(7, 90, 120, a, "native", None, KB), a, 1280, 720, 23) == k


def test_key_changes_with_anything_that_changes_pixels():
    a, v = _photo(), _video()
    base = shot_cache.shot_key(_shot(0, 0, 30, a, "native", None, KB), a, 1280, 720, 23)
    other = [
        shot_cache.shot_key(_shot(0, 0, 31, a, "native", None, KB), a, 1280, 720, 23),          # length
        shot_cache.shot_key(_shot(0, 0, 30, a, "crop", None, KB), a, 1280, 720, 23),            # framing
        shot_cache.shot_key(_shot(0, 0, 30, a, "native", None,
                                  KB.model_copy(update={"zoom_to": 1.1})), a, 1280, 720, 23),   # Ken Burns
        shot_cache.shot_key(_shot(0, 0, 30, a, "native", None, KB), a, 1920, 1080, 18),         # export size
        shot_cache.shot_key(_shot(0, 0, 30, a, "native", None, KB), a, 1280, 720, 23, (0.3, 0.4)),  # face
        shot_cache.shot_key(_shot(0, 0, 30, _photo("p1"), "native", None, KB), _photo("p1"), 1280, 720, 23),
    ]
    with config.override({"render.blur_sigma": 12}):
        other.append(shot_cache.shot_key(_shot(0, 0, 30, a, "native", None, KB), a, 1280, 720, 23))
    assert len({base, *other}) == len(other) + 1
    v1 = shot_cache.shot_key(_shot(0, 0, 30, v, "native", 1.0), v, 1280, 720, 23)
    assert shot_cache.shot_key(_shot(0, 0, 30, v, "native", 1.5), v, 1280, 720, 23) != v1   # in-point


def test_key_ignores_settings_that_do_not_change_pixels():
    a = _photo()
    k = shot_cache.shot_key(_shot(0, 0, 30, a, "native", None, KB), a, 1280, 720, 23)
    with config.override({"render.workers": 2, "render.audio_bitrate": "128k", "render.shot_cache.max_gb": 1}):
        assert shot_cache.shot_key(_shot(0, 0, 30, a, "native", None, KB), a, 1280, 720, 23) == k


def test_audio_key():
    k = shot_cache.audio_key("song1", 12.0, 48.5, 1.5)
    assert shot_cache.audio_key("song1", 12.0, 48.5, 1.5) == k
    assert len({k, shot_cache.audio_key("song2", 12.0, 48.5, 1.5), shot_cache.audio_key("song1", 12.5, 48.5, 1.5),
                shot_cache.audio_key("song1", 12.0, 50.0, 1.5), shot_cache.audio_key("song1", 12.0, 48.5, 1.0)}) == 5
    with config.override({"render.audio_bitrate": "128k"}):
        assert shot_cache.audio_key("song1", 12.0, 48.5, 1.5) != k


def test_song_file_hash_follows_file_contents(tmp_path):
    p = tmp_path / "song.wav"
    p.write_bytes(b"a" * 100)
    h1 = shot_cache.song_file_hash(p)
    assert shot_cache.song_file_hash(p) == h1
    p.write_bytes(b"b" * 101)                     # the file at the same path changed
    assert shot_cache.song_file_hash(p) != h1


def test_cached_shot_hit_and_miss(monkeypatch):
    calls = []

    def fake(shot, asset, out, *a, **kw):
        calls.append(1)
        Path(out).write_bytes(b"x" * 10)

    monkeypatch.setattr(shot_cache, "render_shot", fake)
    a = _photo()
    s = _shot(0, 0, 30, a, "native", None, KB)
    k = shot_cache.shot_key(s, a, 1280, 720, 23)
    p1, hit1 = shot_cache.cached_shot(k, s, a, 1280, 720, 23)
    p2, hit2 = shot_cache.cached_shot(k, s, a, 1280, 720, 23)
    assert (hit1, hit2) == (False, True) and p1 == p2 and len(calls) == 1
    assert not list(p1.parent.glob("*.tmp*"))  # atomic write left no temp file


def test_failed_encode_leaves_no_entry(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(shot_cache, "render_shot", boom)
    a = _photo()
    s = _shot(0, 0, 30, a, "native", None, KB)
    with pytest.raises(RuntimeError):
        shot_cache.cached_shot("k1", s, a, 1280, 720, 23)
    assert not list(shot_cache.cache_dir().glob("*"))


def test_prune_drops_least_recently_used():
    d = shot_cache.cache_dir()
    now = time.time()
    files = []
    for i in range(5):
        p = d / f"k{i}.mp4"
        p.write_bytes(b"x" * 1000)
        os.utime(p, (now - 100 + i, now - 100 + i))  # k0 oldest
        files.append(p)
    n = shot_cache.prune(max_gb=2500 / 1e9, keep={files[0]})
    assert n == 3
    assert [p.exists() for p in files] == [True, False, False, False, True]


# ---------------- media ----------------

def _solid_photos(tmp_path, n=6):
    colors = [(230, 30, 30), (30, 200, 40), (40, 60, 230), (240, 240, 40), (200, 40, 220), (20, 220, 220)]
    amap = {}
    for j in range(n):
        p = tmp_path / f"solid{j}.png"
        Image.new("RGB", (1920, 1080), colors[j % len(colors)]).save(p)
        a = Asset(id=f"s{j}", kind="photo", src_path=str(p), hash=f"s{j}", width=1920, height=1080, filename=p.name)
        amap[a.id] = a
    return amap


def _cut(amap, lens, kb=KB):
    shots, f0 = [], 0
    for j, (a, n) in enumerate(zip(amap.values(), lens)):
        shots.append(_shot(j, f0, f0 + n, a, "native", None, kb))
        f0 += n
    return shots


@pytest.mark.media
def test_recut_encodes_only_changed_shots(fixtures, tmp_path, count_encodes):
    from montage.render.assemble import render_edl
    song = str(fixtures / "click_120.wav")
    amap = _solid_photos(tmp_path)
    shots = _cut(amap, [41, 29, 50, 37, 61, 44])

    out1 = render_edl(_edl(shots, run_id="c1"), amap, song, out_path=tmp_path / "1.mp4")
    assert len(count_encodes) == 6
    count_encodes.clear()

    render_edl(_edl(shots, run_id="c2"), amap, song, out_path=tmp_path / "2.mp4")    # same cut: nothing encoded
    assert count_encodes == []

    recut = [s.model_copy() for s in shots]
    recut[3] = recut[3].model_copy(update={"kenburns": KB.model_copy(update={"zoom_to": 1.12})})
    recut[0], recut[1] = (recut[1].model_copy(update={"slot_idx": 0, "frame_start": 0, "frame_end": 29}),
                          recut[0].model_copy(update={"slot_idx": 1, "frame_start": 29, "frame_end": 70}))
    out3 = render_edl(_edl(recut, run_id="c3"), amap, song, out_path=tmp_path / "3.mp4")
    assert count_encodes == [3]                     # swapped shots are reused, only the changed one is encoded
    assert _probe_frames(out3) == shots[-1].frame_end == _probe_frames(out1)
    # the reordered cut shows shot 1's color first
    assert np.abs(_frame_at(out3, 5).astype(float) - _frame_at(out1, 45).astype(float)).mean() < 6


@pytest.mark.media
def test_cached_render_matches_uncached(fixtures, tmp_path):
    from montage.render.assemble import render_edl
    song = str(fixtures / "click_120.wav")
    amap = _solid_photos(tmp_path, 3)
    edl = _edl(_cut(amap, [20, 25, 30]), run_id="m")
    render_edl(edl, amap, song, out_path=tmp_path / "warm.mp4")                  # fills the cache
    cached = render_edl(edl, amap, song, out_path=tmp_path / "cached.mp4")
    with config.override({"render.shot_cache.enabled": False}):
        plain = render_edl(edl, amap, song, out_path=tmp_path / "plain.mp4")
    assert _probe_frames(cached) == _probe_frames(plain) == 75
    for i in (3, 30, 70):
        assert np.abs(_frame_at(cached, i).astype(float) - _frame_at(plain, i).astype(float)).mean() < 1


@pytest.mark.media
def test_audio_cache_keyed_on_song_file(fixtures, tmp_path):
    """The same plan (same song_hash) rendered with another song file must not reuse the first song's audio."""
    from montage.render.assemble import render_edl
    amap = _solid_photos(tmp_path, 2)
    edl = _edl(_cut(amap, [30, 30]), run_id="s")
    render_edl(edl, amap, str(fixtures / "click_120.wav"), out_path=tmp_path / "a.mp4")
    render_edl(edl, amap, str(fixtures / "click_090.wav"), out_path=tmp_path / "b.mp4")
    assert len(list(shot_cache.cache_dir().glob("audio_*.m4a"))) == 2


@pytest.mark.media
def test_recut_timing(assets, fixtures, tmp_path):
    """40-shot cut: cold render vs a recut that changes 3 shots."""
    from montage.render.assemble import render_edl
    song = str(fixtures / "click_120.wav")
    shots, amap = _forty(assets)
    t0 = time.time()
    render_edl(_edl(shots, run_id="t1"), amap, song, out_path=tmp_path / "a.mp4")
    t1 = time.time()
    recut = list(shots)
    for j in (5, 17, 31):
        s = recut[j]
        kb = s.kenburns.model_copy(update={"zoom_to": s.kenburns.zoom_to + 0.02}) if s.kenburns else None
        recut[j] = s.model_copy(update={"kenburns": kb, "src_in": None if s.src_in is None else s.src_in + 0.1})
    render_edl(_edl(recut, run_id="t2"), amap, song, out_path=tmp_path / "b.mp4")
    t2 = time.time()
    print(json.dumps({"cold_s": round(t1 - t0, 1), "recut_3_of_40_s": round(t2 - t1, 1)}))
    assert t2 - t1 < (t1 - t0) / 2


@pytest.fixture(scope="module")
def assets(fixtures):
    from montage.ingest.assets import ingest
    return {a.filename: a for a in ingest([fixtures])}
