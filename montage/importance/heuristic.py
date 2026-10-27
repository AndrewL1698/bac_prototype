"""Heuristic importance (batch-relative z-scores) and hard rejects."""

from __future__ import annotations

import numpy as np

from montage.config import get_config


def _z(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = np.zeros_like(x)
    m = np.ones(len(x), bool) if mask is None else mask
    if m.sum() == 0:
        return out
    v = x[m]
    sd = v.std()
    out[m] = (v - v.mean()) / sd if sd > 1e-9 else 0.0
    return out


def is_junk(tags: dict[str, list[str]], ids: list[str]) -> np.ndarray:
    junk = set(get_config().importance.junk_tags)
    return np.array([bool(junk & set(tags.get(c, []))) for c in ids], float)


def heuristic_scores(arr: dict[str, np.ndarray], tags: dict[str, list[str]], ids: list[str]) -> np.ndarray:
    w = get_config().importance.heuristic
    vid = np.asarray(arr["is_video"], bool)
    H = (w.aes * _z(arr["aesthetic"]) + w.sharp * _z(arr["sharpness"]) - w.exp * _z(arr["exposure_clip"])
         - w.bright * np.abs(arr["brightness"] - 0.5) * 4
         + w.face_count * np.minimum(arr["face_count"], 3) / 3 + w.face_area * _z(arr["face_area"])
         + w.color * _z(arr["colorfulness"])
         + w.motion * _z(arr["motion"], vid) - w.shake * _z(arr["shake"], vid)
         - w.junk * is_junk(tags, ids))
    return H.astype(np.float32)


def hard_rejects(arr: dict[str, np.ndarray], H: np.ndarray, n_needed: int = 0) -> np.ndarray:
    """Boolean mask of rejected candidates. If too few would remain, the best rejects come back."""
    c = get_config().importance
    sharp = arr["sharpness"]
    pct = np.percentile(sharp, c.reject_sharpness_pct) if len(sharp) else 0.0
    rej = ((sharp < pct) & (sharp < c.abs_min_sharpness)) | (arr["exposure_clip"] > c.reject_exposure_clip)
    shortfall = n_needed - int((~rej).sum())
    if shortfall > 0 and rej.any():
        back = np.where(rej)[0]
        back = back[np.argsort(-H[back])][:shortfall]
        rej[back] = False
    return rej
