"""In/out points: where inside a chosen video window the shot starts."""

from __future__ import annotations

import numpy as np

from montage.config import get_config
from montage.types import Candidate


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def choose_in(c: Candidate, dur: float, frames: dict[str, np.ndarray], qemb: np.ndarray | None,
              taken: list[tuple[float, float]]) -> tuple[float, bool]:
    """Best src_in in [src_start, src_max_end - dur] (0.1 s steps). Returns (src_in, overlap_free)."""
    cfg = get_config().planner.inout
    times = np.asarray(frames["times"], float)
    aes = _z(np.asarray(frames["aesthetic"], float))
    mot = np.nan_to_num(np.asarray(frames["motion"], float))
    mot = _z(mot)
    if qemb is not None and len(qemb):
        rel = _z((np.asarray(frames["emb"]) @ qemb.T).max(1))
    else:
        rel = np.zeros(len(times))
    per_frame = cfg.w_aes * aes + cfg.w_motion * mot + cfg.w_rel * rel
    lo = c.src_start or 0.0
    hi = max(lo, (c.src_max_end or lo + dur) - dur)
    starts = np.round(np.arange(lo, hi + 1e-6, cfg.step_s), 3)
    if len(starts) == 0:
        starts = np.array([lo])
    best, best_free = None, None
    for s in starts:
        m = (times >= s - 1e-6) & (times <= s + dur + 1e-6)
        val = per_frame[m].mean() if m.any() else per_frame[int(np.argmin(np.abs(times - (s + dur / 2))))]
        score = val - cfg.drift_penalty * (s - lo)
        free = all(s + dur <= a + 1e-6 or s >= b - 1e-6 for a, b in taken)
        if best is None or score > best[0]:
            best = (score, s)
        if free and (best_free is None or score > best_free[0]):
            best_free = (score, s)
    if best_free is not None:
        return float(best_free[1]), True
    return float(best[1]), False
