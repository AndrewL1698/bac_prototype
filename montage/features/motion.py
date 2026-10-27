"""Optical-flow motion per motion pair (t, t + 1/15 s) and per-candidate shake."""

from __future__ import annotations

import numpy as np


def pair_motion(a_path: str, b_path: str) -> tuple[float, float, float]:
    """(mean magnitude, global dx, global dy), all divided by the frame diagonal."""
    import cv2
    a = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != b.shape:
        return float("nan"), float("nan"), float("nan")
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    diag = float(np.hypot(*a.shape))
    mag = np.hypot(flow[..., 0], flow[..., 1])
    return float(mag.mean() / diag), float(np.median(flow[..., 0]) / diag), float(np.median(flow[..., 1]) / diag)


def shake(gdx: np.ndarray, gdy: np.ndarray) -> float:
    """Std of frame-to-frame changes of the global motion vector (high-frequency jitter)."""
    ok = ~(np.isnan(gdx) | np.isnan(gdy))
    gx, gy = gdx[ok], gdy[ok]
    if len(gx) < 3:
        return 0.0
    ddx, ddy = np.diff(gx), np.diff(gy)
    return float(np.sqrt(ddx.var() + ddy.var()))
