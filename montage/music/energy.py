"""Energy curves on the 2 Hz grid: rms_db, onset envelope/density, spectral centroid."""

from __future__ import annotations

import numpy as np

HOP = 512
GRID_DT = 0.5


def grid_for(duration: float) -> np.ndarray:
    return np.round(np.arange(0.0, duration, GRID_DT), 3)


def to_grid(values: np.ndarray, frame_times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Average frames in a 0.5 s bin centered on each grid time (empty bins take the nearest frame)."""
    out = np.empty(len(grid), np.float32)
    idx = np.searchsorted(frame_times, np.concatenate([grid - GRID_DT / 2, [grid[-1] + GRID_DT / 2]]))
    for k in range(len(grid)):
        a, b = idx[k], idx[k + 1]
        if b > a:
            out[k] = values[a:b].mean()
        else:
            out[k] = values[min(len(values) - 1, max(0, a))]
    return out


def energy_curves(y: np.ndarray, sr: int, grid: np.ndarray) -> dict[str, np.ndarray]:
    import librosa
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time")
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    ft = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    lo = np.searchsorted(onsets, grid - 1.0)
    hi = np.searchsorted(onsets, grid + 1.0)
    return {
        "rms_db": to_grid(rms_db, ft, grid),
        "onset_env": to_grid(onset_env[: len(ft)], ft[: len(onset_env)], grid),
        "onset_density": ((hi - lo) / 2.0).astype(np.float32),
        "centroid": to_grid(cent, ft[: len(cent)], grid),
    }


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    sd = x.std()
    return ((x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)).astype(np.float32)


def gauss_smooth(x: np.ndarray, sigma_s: float, dt: float = GRID_DT) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    if sigma_s <= 0:
        return np.asarray(x, np.float32)
    return gaussian_filter1d(np.asarray(x, np.float64), sigma_s / dt, mode="nearest").astype(np.float32)
