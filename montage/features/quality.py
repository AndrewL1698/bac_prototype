"""Per-frame image quality on the 448 px analysis image."""

from __future__ import annotations

import numpy as np

QUALITY_KEYS = ("sharpness", "brightness", "contrast", "exposure_clip", "colorfulness")


def frame_quality(path: str) -> dict[str, float]:
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return {k: float("nan") for k in QUALITY_KEYS}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return {
        "sharpness": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
        "brightness": float(luma.mean() / 255.0),
        "contrast": float(luma.std() / 255.0),
        "exposure_clip": float(((luma < 5) | (luma > 250)).mean()),
        "colorfulness": float(colorful / 100.0),
    }
