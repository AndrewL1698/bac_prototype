"""Cost terms for the beam search (precomputed once per plan, on the candidate pool)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from montage.config import get_config
from montage.types import Candidate, Slot

INF = 1e9


def rank01(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    return np.full(len(x), 0.5) if len(x) < 2 else (rankdata(x) - 1) / (len(x) - 1)


def thumb_hist(path: str) -> np.ndarray:
    from PIL import Image
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB").resize((64, 36)), np.int32) // 32
    except (OSError, ValueError):
        return np.full(512, 1 / 512)
    h = np.bincount((a[..., 0] * 64 + a[..., 1] * 8 + a[..., 2]).ravel(), minlength=512).astype(float)
    return h / h.sum()


def visual_jump(hists: np.ndarray, luma: np.ndarray) -> np.ndarray:
    """vj = 0.5 * chi-square(hist) + 0.5 * 2|dluma|, divided by the batch 95th percentile, clipped to [0, 1]."""
    n = len(hists)
    if n < 2:
        return np.zeros((n, n))
    a, b = hists[:, None, :], hists[None, :, :]
    chi = 0.5 * ((a - b) ** 2 / (a + b + 1e-9)).sum(-1)
    vj = 0.5 * chi + 0.5 * 2 * np.abs(luma[:, None] - luma[None, :])
    off = vj[~np.eye(n, dtype=bool)]
    p95 = np.percentile(off, 95) if len(off) else 1.0
    return np.clip(vj / max(p95, 1e-9), 0, 1)


@dataclass
class CostModel:
    pool: np.ndarray                 # indices into the full candidate list
    slots: list[Slot]
    U: np.ndarray                    # (M, K) unary, INF where infeasible
    P: np.ndarray                    # (M, M) boundary-independent pairwise
    vj: np.ndarray                   # (M, M) visual jump
    b: np.ndarray                    # (K,) boundary strengths
    parts: dict[str, np.ndarray] = field(default_factory=dict)   # explanation terms, (M,) or (M, K)

    def pair(self, last: np.ndarray, j: int) -> np.ndarray:
        """(len(last), M) pairwise cost of following each `last` with every pool item at slot j."""
        c = get_config().planner
        coef = c.w_jump * (1 - self.b[j]) - c.w_contrast * self.b[j]
        return self.P[last] + coef * self.vj[last]


def build_costs(pool: np.ndarray, slots: list[Slot], cands: list[Candidate], arr: dict[str, np.ndarray],
                imp: np.ndarray, D: np.ndarray, target_arousal: np.ndarray, taken: list[datetime | None],
                chronological: bool, must: np.ndarray, emotion_source: str, seed: int) -> CostModel:
    """D: (N, K) director scores; must: (N,) bool must-include/pinned. Everything is sliced to the pool."""
    c = get_config().planner
    K, M = len(slots), len(pool)
    dur = np.array([s.end - s.start for s in slots])
    is_video = np.array([cands[i].kind == "video" for i in pool])
    vr = rank01(arr["visual_arousal"][pool])
    vv = rank01(arr["visual_valence"][pool])
    tv = rank01(np.array([s.valence for s in slots]))
    irank = rank01(np.array([s.intensity for s in slots]))
    w_val = 0.0 if emotion_source == "heuristic" else c.w_val
    imp_p = imp[pool]
    mood = c.w_mood * np.abs(vr[:, None] - target_arousal[None, :]) + w_val * np.abs(vv[:, None] - tv[None, :])
    Dp = D[pool]
    U = -c.w_imp * imp_p[:, None] + mood - c.w_dir * Dp
    span = np.array([(cands[i].src_max_end or 0) - (cands[i].src_start or 0) if cands[i].kind == "video" else np.inf
                     for i in pool])
    U = U + np.where(is_video[:, None] & (span[:, None] + 1e-6 < dur[None, :]), INF, 0.0)
    U = U + np.where(~is_video[:, None] & (dur[None, :] < c.photo_min_s), c.w_photo_short, 0.0)
    if is_video.any():
        U = U + np.where(~is_video[:, None], c.w_photo_hi * irank[None, :], 0.0)
    U = U - c.w_must * must[pool][:, None]
    finite = U[U < INF / 2]
    if c.variety > 0 and finite.size > 1:
        rng = np.random.default_rng(seed)
        U = U + c.variety * finite.std() * rng.gumbel(size=U.shape)

    emb = arr["emb"][pool]
    cos = emb @ emb.T
    assets = np.array([cands[i].asset_id for i in pool])
    P = c.w_rep * np.maximum(cos - c.rep_tau, 0) + c.w_src * (assets[:, None] == assets[None, :])
    if chronological:
        ts = np.array([t.timestamp() if t else np.nan for t in (taken[i] for i in pool)])
        later_first = (ts[None, :] < ts[:, None] - c.chrono_tol_s) & ~np.isnan(ts[:, None]) & ~np.isnan(ts[None, :])
        P = P + c.w_chrono * later_first
    hists = np.stack([thumb_hist(cands[i].thumb_path) for i in pool]) if M else np.zeros((0, 512))
    vj = visual_jump(hists, arr["brightness"][pool])
    b = np.array([s.boundary_strength for s in slots])
    return CostModel(pool=pool, slots=slots, U=U, P=P, vj=vj, b=b,
                     parts={"importance": -c.w_imp * imp_p, "mood": mood, "director": -c.w_dir * Dp,
                            "imp_raw": imp_p, "D_raw": Dp, "vr": vr, "ta": target_arousal})
