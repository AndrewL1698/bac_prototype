"""Beat This! checkpoint mirror: verified fetch order, corrupt-cache repair, retry after a failed load.
Pure Python, no network: sources are file:// URLs."""

from __future__ import annotations

import hashlib

import pytest

from montage import config, weights


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_fetch_verified_skips_dead_and_wrong_sources(tmp_path):
    good = b"checkpoint" * 1000
    (tmp_path / "wrong.ckpt").write_bytes(b"tampered")
    (tmp_path / "good.ckpt").write_bytes(good)
    urls = [(tmp_path / "missing.ckpt").as_uri(), (tmp_path / "wrong.ckpt").as_uri(), (tmp_path / "good.ckpt").as_uri()]
    target = tmp_path / "weights" / "beat_this-x.ckpt"
    p, source = weights.fetch_verified(target, urls, _sha(good))
    assert p == target and target.read_bytes() == good and source == urls[2]
    # second call: served from the cache, sources untouched
    (tmp_path / "good.ckpt").unlink()
    assert weights.fetch_verified(target, urls, _sha(good))[1] == "cache"
    assert not list(target.parent.glob(".*tmp*"))   # no temp files left behind


def test_fetch_verified_repairs_corrupt_cache_from_local(tmp_path):
    good = b"weights" * 500
    target = tmp_path / "beat_this-x.ckpt"
    target.write_bytes(b"truncated")
    hub = tmp_path / "hub.ckpt"
    hub.write_bytes(good)
    p, source = weights.fetch_verified(target, [], _sha(good), local=[tmp_path / "nope.ckpt", hub])
    assert source.startswith("local:") and p.read_bytes() == good


def test_fetch_verified_raises_with_reasons(tmp_path):
    with pytest.raises(IOError, match="missing.ckpt"):
        weights.fetch_verified(tmp_path / "t.ckpt", [(tmp_path / "missing.ckpt").as_uri()], "0" * 64)


def test_beat_this_config_consistent():
    cfg = config.load_config().music.beat_this
    assert cfg.checkpoint in cfg.sha256
    assert all(len(h) == 64 for h in cfg.sha256.values())
    assert cfg.sources and all("{name}" in s for s in cfg.sources)


def test_ensure_checkpoint_uses_configured_sources(tmp_path, monkeypatch):
    from montage.music import beats
    good = b"beat this" * 100
    (tmp_path / "final0.ckpt").write_bytes(good)
    monkeypatch.setenv("MONTAGE_DATA_ROOT", str(tmp_path / "data"))
    with config.override({"music.beat_this.sources": [(tmp_path).as_uri() + "/{name}.ckpt"],
                          "music.beat_this.sha256": {"final0": _sha(good)}}):
        p, source = beats.ensure_checkpoint()
        assert p == beats.checkpoint_on_drive() and p.read_bytes() == good and source.startswith("file:")


def test_failed_load_retries_after_window(monkeypatch):
    from montage.music import beats
    calls = []

    def boom():
        calls.append(1)
        raise IOError("host down")

    monkeypatch.setattr(beats, "_beat_this", None)
    monkeypatch.setattr(beats, "_failed_at", None)
    monkeypatch.setattr(beats, "_load_model", boom)
    with pytest.raises(IOError):
        beats._get_beat_this()
    with pytest.raises(RuntimeError, match="retrying after"):
        beats._get_beat_this()       # inside the window: no new attempt
    assert len(calls) == 1
    monkeypatch.setattr(beats, "_failed_at", beats._failed_at - 10_000)
    with pytest.raises(IOError):
        beats._get_beat_this()       # window over: tried again
    assert len(calls) == 2
    # analyze_beats still answers via librosa while Beat This! is unavailable
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    sr = 22050
    y = np.zeros(sr * 8, dtype=np.float32)
    y[:: sr // 2] = 1.0
    assert beats.analyze_beats(y, sr)["beat_source"] == "librosa"
