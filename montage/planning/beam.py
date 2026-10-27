"""Beam search over the slot grid: pick and order shots minimizing unary + pairwise cost under constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from montage.config import get_config
from montage.planning.costs import INF, CostModel
from montage.types import Candidate, UserError


@dataclass
class BeamResult:
    seq: list[int]           # pool positions, one per slot
    total: float
    reuse_mode: bool


@dataclass
class Constraints:
    cluster: np.ndarray       # (M,) dup cluster id (compact)
    asset: np.ndarray         # (M,) asset id (compact)
    is_video: np.ndarray      # (M,) bool
    overlap: np.ndarray       # (M, M) same video asset and overlapping source windows (incl. self)
    must_asset: np.ndarray    # (M,) index into must list, -1 if not a must asset
    n_must: int
    band: tuple[float, float]  # photo fraction band


def make_constraints(pool: np.ndarray, cands: list[Candidate], dup: np.ndarray, must_assets: list[str], K: int) -> Constraints:
    c = get_config().planner
    M = len(pool)
    cl = {v: i for i, v in enumerate(dict.fromkeys(int(dup[i]) for i in pool))}
    aid = {v: i for i, v in enumerate(dict.fromkeys(cands[i].asset_id for i in pool))}
    cluster = np.array([cl[int(dup[i])] for i in pool], int)
    asset = np.array([aid[cands[i].asset_id] for i in pool], int)
    is_video = np.array([cands[i].kind == "video" for i in pool])
    s = np.array([cands[i].src_start or 0.0 for i in pool])
    e = np.array([cands[i].src_end or 0.0 for i in pool])
    same = asset[:, None] == asset[None, :]
    overlap = same & is_video[:, None] & (s[:, None] < e[None, :] - 1e-6) & (s[None, :] < e[:, None] - 1e-6)
    mi = {a: k for k, a in enumerate(must_assets)}
    must_asset = np.array([mi.get(cands[i].asset_id, -1) for i in pool], int)
    n_photo = int((~is_video).sum())
    video_cap = 0
    for a in np.unique(asset[is_video]):
        idx = np.where((asset == a) & is_video)[0]
        order = idx[np.argsort(s[idx])]
        count, last_end = 0, -1.0
        for i in order:
            if s[i] >= last_end - 1e-6:
                count, last_end = count + 1, e[i]
        video_cap += min(count, c.max_windows_per_video)
    lo = min(c.min_photo_frac, n_photo / max(K, 1))
    hi = max(c.max_photo_frac, 1 - video_cap / max(K, 1))
    return Constraints(cluster, asset, is_video, overlap, must_asset, len(must_assets), (lo, min(1.0, hi)))


def beam_search(cm: CostModel, cons: Constraints, reuse_mode: bool = False) -> BeamResult:
    c = get_config().planner
    U = cm.U
    M, K = U.shape
    B, E = c.beam_width, c.expand_top
    n_cl, n_as = int(cons.cluster.max(initial=-1)) + 1, int(cons.asset.max(initial=-1)) + 1
    lo, hi = cons.band
    photo = (~cons.is_video).astype(float)

    def ratio_pen(p: np.ndarray, done: bool) -> np.ndarray:
        frac = p / K
        over = np.maximum(frac - hi, 0) ** 2
        under = np.maximum(lo - frac, 0) ** 2 if done else 0.0
        return c.w_ratio * K * (over + under)

    # state arrays
    cost = np.zeros(1)
    seqs = np.zeros((1, 0), int)
    used = np.zeros((1, M), int)
    clc = np.zeros((1, n_cl), int)
    asc = np.zeros((1, n_as), int)
    pc = np.zeros(1)
    mustc = np.zeros((1, max(cons.n_must, 1)), bool)

    for j in range(K):
        base = cost[:, None] + U[None, :, j]
        if j > 0:
            base = base + cm.pair(seqs[:, -1], j)
        # photo-ratio: incremental share of the over-the-band penalty
        new_pc = pc[:, None] + photo[None, :]
        base = base + ratio_pen(new_pc, False) - ratio_pen(pc, False)[:, None]
        mask = base >= INF / 2
        if reuse_mode:
            base = base + c.w_reuse * used
            if j > 0:
                mask |= np.arange(M)[None, :] == seqs[:, -1][:, None]
        else:
            mask |= used > 0
            mask |= clc[:, cons.cluster] >= c.max_per_cluster
            vid_full = (asc[:, cons.asset] >= c.max_windows_per_video) & cons.is_video[None, :]
            mask |= vid_full
            mask |= (used @ cons.overlap.astype(int)) > 0
        if cons.n_must:
            covers = (cons.must_asset[None, :] >= 0) & ~np.take_along_axis(
                mustc, np.maximum(cons.must_asset, 0)[None, :].repeat(len(cost), 0), 1)
            remaining = (~mustc).sum(1)[:, None] - covers
            mask |= remaining > (K - j - 1)
        base = np.where(mask, np.inf, base)
        k = min(E, M)
        part = np.argpartition(base, k - 1, axis=1)[:, :k] if k < M else np.tile(np.arange(M), (len(cost), 1))
        vals = np.take_along_axis(base, part, 1)
        flat = np.argsort(vals, axis=None)[:B]
        si, ci = np.unravel_index(flat, vals.shape)
        keep = np.isfinite(vals[si, ci])
        si, ci = si[keep], ci[keep]
        if len(si) == 0:
            raise _NoSolution()
        m = part[si, ci]
        cost = vals[si, ci]
        seqs = np.concatenate([seqs[si], m[:, None]], 1)
        used = used[si].copy()
        used[np.arange(len(m)), m] += 1
        clc = clc[si].copy()
        clc[np.arange(len(m)), cons.cluster[m]] += 1
        asc = asc[si].copy()
        asc[np.arange(len(m)), cons.asset[m]] += 1
        pc = pc[si] + photo[m]
        mustc = mustc[si].copy()
        hit = cons.must_asset[m] >= 0
        mustc[np.where(hit)[0], cons.must_asset[m][hit]] = True

    final = cost + ratio_pen(pc, True) - ratio_pen(pc, False)
    if cons.n_must:
        final = np.where(mustc.all(1), final, np.inf)
    best = int(np.argmin(final))
    if not np.isfinite(final[best]):
        raise _NoSolution()
    return BeamResult(seq=seqs[best].tolist(), total=float(final[best]), reuse_mode=reuse_mode)


class _NoSolution(Exception):
    pass


def solve(cm: CostModel, cons: Constraints, reuse_mode: bool = False) -> BeamResult:
    try:
        return beam_search(cm, cons, reuse_mode)
    except _NoSolution:
        if not reuse_mode:
            try:
                return beam_search(cm, cons, True)
            except _NoSolution:
                pass
        raise UserError("Not enough usable footage for this song length. Add more media or shorten the video.")


def sequence_cost(cm: CostModel, seq: list[int]) -> float:
    """Unary + pairwise cost of a fixed sequence (no ratio penalty); used to compare with baselines."""
    total = 0.0
    for j, m in enumerate(seq):
        total += float(cm.U[m, j])
        if j > 0:
            total += float(cm.pair(np.array([seq[j - 1]]), j)[0, m])
    return total


def objective(cm: CostModel, cons: Constraints, seq: list[int]) -> float:
    """What the beam minimizes for a complete sequence: unary + pairwise + photo-ratio penalty."""
    c = get_config().planner
    K = len(seq)
    lo, hi = cons.band
    frac = float(np.sum(~cons.is_video[seq])) / max(K, 1)
    pen = c.w_ratio * K * (max(frac - hi, 0.0) ** 2 + max(lo - frac, 0.0) ** 2)
    return sequence_cost(cm, seq) + pen
