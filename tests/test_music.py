"""Phase 2 acceptance: music analysis."""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import gpu_available


def _f(ref, est, tol=0.07):
    mir_eval = pytest.importorskip("mir_eval")
    r = mir_eval.beat.trim_beats(np.asarray(ref, float))
    e = mir_eval.beat.trim_beats(np.asarray(est, float))
    return mir_eval.beat.f_measure(r, e, f_measure_threshold=tol)


def _load(fixtures, name):
    pytest.importorskip("librosa")
    from montage.music.load import load_audio
    return load_audio(fixtures / name, 22050)


def test_emotion_fallback(monkeypatch):
    from montage.music import emotion
    monkeypatch.setattr(emotion, "mert_available", lambda: False)
    grid = np.arange(0, 60, 0.5)
    rng = np.random.default_rng(0)
    curves = {k: rng.normal(size=len(grid)).astype(np.float32) for k in ("rms_db", "onset_density", "centroid")}
    a, v, src = emotion.predict_emotion(np.zeros(24000 * 60, np.float32), grid, curves)
    assert src == "heuristic" and len(a) == len(v) == len(grid)
    assert np.all(v == 0) and np.all(np.abs(a) <= 1)


def test_make_windows_centered():
    from montage.music.emotion import make_windows
    sr = 100
    y = np.arange(1000, dtype=np.float32)  # 10 s
    w = make_windows(y, np.array([0.0, 5.0, 9.9]), sr=sr, window_s=2.0)
    assert w.shape == (3, 200)
    assert w[1, 100] == 500.0            # center sample at t=5 s
    assert np.all(w[0, :100] == 0)       # zero-padded before the start
    assert w[2, 100] == 990.0 and np.all(w[2, 110:] == 0)


@pytest.mark.media
@pytest.mark.parametrize("name", ["click_090.wav", "click_120.wav", "click_150.wav"])
def test_beats_click_librosa(fixtures, truth, name):
    from montage.music.beats import librosa_beats
    beats, _ = librosa_beats(_load(fixtures, name), 22050)
    assert _f(truth[name]["beats"], beats) >= 0.9


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["click_090.wav", "click_120.wav", "click_150.wav"])
def test_beats_click_beat_this(fixtures, truth, name):
    if not gpu_available():
        pytest.skip("no GPU")
    from montage.music.beats import beat_this_beats
    beats, _ = beat_this_beats(_load(fixtures, name), 22050)
    assert _f(truth[name]["beats"], beats) >= 0.95


@pytest.mark.media
def test_downbeats_click(fixtures, truth):
    from montage.music.beats import beat_this_beats, librosa_beats
    y = _load(fixtures, "click_120.wav")
    ref = truth["click_120.wav"]["downbeats"]
    _, d = librosa_beats(y, 22050)
    assert _f(ref, d) >= 0.9
    if gpu_available():
        _, d2 = beat_this_beats(y, 22050)
        assert _f(ref, d2) >= 0.9


@pytest.mark.media
def test_half_time(fixtures, truth):
    from montage import config
    from montage.music.beats import analyze_beats
    y = _load(fixtures, "click_150.wav")
    with config.override({"music.max_bpm": 140}):
        b = analyze_beats(y, 22050)
    assert b["beat_mode"] == "half"
    n_true = len(truth["click_150.wav"]["beats"])
    assert abs(len(b["beats"]) - n_true / 2) <= 3


@pytest.mark.media
def test_sections(fixtures, truth):
    from montage.music.profile import analyze_song
    fa = analyze_song(fixtures / "song_sections.wav")
    bar = 2.0
    for t in truth["song_sections.wav"]["boundaries"]:
        assert any(abs(b - t) <= bar + 1e-6 for b in fa.boundaries), (t, fa.boundaries)


@pytest.mark.media
def test_intensity_steps(fixtures):
    from montage.music.profile import analyze_song
    fa = analyze_song(fixtures / "click_120_dynamic.wav")
    g, i = fa.a("grid"), fa.a("intensity")
    m = [i[(g >= a) & (g < b)].mean() for a, b in ((2, 18), (22, 38), (42, 68))]
    assert m[0] < m[1] < m[2], m


@pytest.mark.media
@pytest.mark.parametrize("name", ["song_sections.wav", "click_120_dynamic.wav"])
def test_excerpt_rules(fixtures, name):
    from montage.music.excerpt import choose_excerpt
    from montage.music.profile import analyze_song, build_profile
    fa = analyze_song(fixtures / name)
    for hint in ("auto", "start", "peak", "custom"):
        ex = choose_excerpt(fa, 0, hint, 10.0)
        L = ex.song_end - ex.song_start
        assert 45 - 1e-6 <= L <= 60 + 1e-6
        bars = L / fa.bar
        assert abs(bars - round(bars)) * fa.bar <= 1 / 30
        assert min(abs(d - ex.song_start) for d in fa.downbeats) < 1e-3
        p = build_profile(fa, ex)
        assert p.beats[0] == 0.0 and all(0 <= b < L for b in p.beats)
        assert abs(len(p.grid_times) - L / 0.5) <= 1
        assert p.sections[0].start == 0 and abs(p.sections[-1].end - L) < 1e-3
    if name == "song_sections.wav":
        ex = choose_excerpt(fa, 0, "peak")
        p = build_profile(fa, ex)
        frac = p.peak_time / (ex.song_end - ex.song_start)
        assert 0.55 <= frac <= 0.85, frac


@pytest.mark.media
def test_music_cache(fixtures):
    from montage.music import beats
    from montage.music.profile import analyze_song
    analyze_song(fixtures / "click_090.wav")
    before = dict(beats.CALLS)
    fa = analyze_song(fixtures / "click_090.wav")
    assert beats.CALLS == before
    assert fa.beat_source in ("beat_this", "librosa")


@pytest.mark.media
def test_summary(fixtures):
    from montage.music.excerpt import choose_excerpt
    from montage.music.profile import analyze_song, build_profile, summarize
    fa = analyze_song(fixtures / "song_sections.wav")
    text = summarize(build_profile(fa, choose_excerpt(fa)))
    assert "120 BPM" in text and "S0 0.0-" in text


@pytest.mark.gpu
def test_music_timing(fixtures, tmp_path, fresh_root):
    if not gpu_available():
        pytest.skip("no GPU")
    from scipy.io import wavfile
    from montage.music.profile import analyze_song
    sr, y = wavfile.read(fixtures / "song_sections.wav")
    long = np.concatenate([y, y, y[: int(30 * sr)]])  # 3.5 min
    wavfile.write(tmp_path / "long.wav", sr, long)
    t0 = time.time()
    analyze_song(tmp_path / "long.wav")
    dt = time.time() - t0
    assert dt < 20, f"full-song analysis took {dt:.1f}s"
