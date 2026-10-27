"""Beats and downbeats: Beat This! (GPU) with a librosa fallback; half/double-time adjustment."""

from __future__ import annotations

import threading

import numpy as np

from montage.config import get_config
from montage.log import get_logger

log = get_logger("beats")
_lock = threading.Lock()
_beat_this = None
CALLS = {"beat_this": 0, "librosa": 0}


def _get_beat_this():
    global _beat_this
    with _lock:
        if _beat_this is None:
            import torch
            from beat_this.inference import Audio2Beats
            _beat_this = Audio2Beats(checkpoint_path="final0", device="cuda" if torch.cuda.is_available() else "cpu",
                                     dbn=False)
        return _beat_this


def beat_this_beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    CALLS["beat_this"] += 1
    beats, downbeats = _get_beat_this()(y, sr)
    return np.asarray(beats, float), np.asarray(downbeats, float)


def downbeat_phase(y: np.ndarray, sr: int, beats: np.ndarray, hop: int = 512) -> int:
    """Pick the bar phase (0..3) whose beats carry the most low-frequency (< 200 Hz) energy."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 200].sum(axis=0)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = min(len(low), len(onset))
    strength = low[:n] / (low[:n].max() + 1e-9) + 0.5 * onset[:n] / (onset[:n].max() + 1e-9)
    frames = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=hop), 0, n - 1)
    # take the max over a short window after each beat (attack may lag the beat estimate)
    vals = np.array([strength[f:f + 3].max() for f in frames])
    scores = [vals[p::4].sum() / max(1, len(vals[p::4])) for p in range(4)]
    return int(np.argmax(scores))


def librosa_beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    CALLS["librosa"] += 1
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", trim=False)
    beats = np.asarray(beats, float)
    if len(beats) < 8:
        return beats, beats[::4]
    return beats, beats[downbeat_phase(y, sr, beats)::4]


def tempo_of(beats: np.ndarray) -> float:
    d = np.diff(beats)
    return float(60.0 / np.median(d)) if len(d) else 120.0


def adjust_mode(beats: np.ndarray, downbeats: np.ndarray, tempo: float) -> tuple[np.ndarray, str]:
    cfg = get_config().music
    if tempo > cfg.max_bpm and len(beats) > 4:
        # keep every other beat, choosing the parity that keeps the downbeats
        hits = []
        for p in (0, 1):
            kept = beats[p::2]
            hits.append(sum(np.min(np.abs(kept - d)) < 0.05 for d in downbeats) if len(downbeats) else 0)
        return beats[int(np.argmax(hits))::2], "half"
    if tempo < cfg.min_bpm and len(beats) > 1:
        mids = (beats[:-1] + beats[1:]) / 2
        return np.sort(np.concatenate([beats, mids])), "double"
    return beats, "normal"


def analyze_beats(y: np.ndarray, sr: int) -> dict:
    source = "librosa"
    try:
        beats, downbeats = beat_this_beats(y, sr)
        if len(beats) < 8:
            raise RuntimeError(f"only {len(beats)} beats")
        source = "beat_this"
    except Exception as e:  # noqa: BLE001  optional component
        log.warning("Beat This! unavailable (%s); using librosa beats", e)
        beats, downbeats = librosa_beats(y, sr)
    if len(downbeats) == 0:
        downbeats = beats[::4]
    tempo = tempo_of(beats)
    beats, mode = adjust_mode(beats, downbeats, tempo)
    return {"beats": [round(float(b), 4) for b in beats], "downbeats": [round(float(d), 4) for d in downbeats],
            "tempo": round(tempo, 2), "beat_mode": mode, "beat_source": source}
