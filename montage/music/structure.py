"""Section boundaries: beat-synchronous MFCC+chroma, self-similarity, Foote novelty, peak picking."""

from __future__ import annotations

import numpy as np

from montage.config import get_config

HOP = 512


def beat_sync_features(y: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
    """(n_beats, 32): 20 MFCC + 12 chroma averaged over each beat interval, z-scored per dim."""
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    n = min(mfcc.shape[1], chroma.shape[1])
    feats = np.vstack([mfcc[:, :n], chroma[:, :n]])
    bf = np.maximum.accumulate(np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=HOP), 0, n - 1))
    X = np.zeros((len(beats), feats.shape[0]))
    for i, a in enumerate(bf):  # one row per beat: mean over [beat i, beat i+1)
        b = bf[i + 1] if i + 1 < len(bf) else n
        X[i] = feats[:, a:max(b, a + 1)].mean(axis=1)
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd > 1e-9, sd, 1.0)


def checkerboard(half: int) -> np.ndarray:
    idx = np.arange(-half, half) + 0.5
    g = np.exp(-(idx[:, None] ** 2 + idx[None, :] ** 2) / (2 * (half / 2.0) ** 2))
    sign = np.sign(idx[:, None]) * np.sign(idx[None, :])
    return g * sign


def foote_novelty(X: np.ndarray, half: int) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    S = Xn @ Xn.T
    n = len(S)
    K = checkerboard(half)
    P = np.pad(S, half, mode="constant")
    nov = np.array([(K * P[i:i + 2 * half, i:i + 2 * half]).sum() for i in range(n)])
    return np.maximum(nov, 0)


def find_boundaries(beats: np.ndarray, downbeats: np.ndarray, nov: np.ndarray, duration: float) -> list[float]:
    from scipy.signal import find_peaks
    cfg = get_config().music.structure
    if len(nov) < 3:
        return []
    thr = nov.mean() + cfg.peak_k * nov.std()
    peaks, _ = find_peaks(nov, height=thr, distance=max(1, cfg.min_section_beats))
    bar = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else 2.0
    out: list[float] = []
    for p in peaks:
        t = float(beats[p])
        if len(downbeats):
            t = float(downbeats[np.argmin(np.abs(downbeats - t))])
        if t < 2 * bar or t > duration - 2 * bar:
            continue
        if not out or t - out[-1] > bar / 2:
            out.append(round(t, 4))
    return out


def novelty_grid(beats: np.ndarray, nov: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(nov) == 0:
        return np.zeros(len(grid), np.float32)
    g = np.interp(grid, beats, nov, left=0.0, right=0.0)
    return (g / g.max() if g.max() > 0 else g).astype(np.float32)


def analyze_structure(y: np.ndarray, sr: int, beats: list[float], downbeats: list[float],
                      grid: np.ndarray, duration: float) -> tuple[list[float], np.ndarray]:
    b, d = np.asarray(beats), np.asarray(downbeats)
    if len(b) < 8:
        return [], np.zeros(len(grid), np.float32)
    X = beat_sync_features(y, sr, b)
    nov = foote_novelty(X, get_config().music.structure.kernel_beats)
    return find_boundaries(b, d, nov, duration), novelty_grid(b, nov, grid)
