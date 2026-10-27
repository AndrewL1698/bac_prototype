"""Slot grid: when every cut happens. All cuts on beats; section boundaries force cuts."""

from __future__ import annotations

import numpy as np

from montage.config import get_config
from montage.types import MusicProfile, Slot

EPS = 1e-6
SNAP = np.array([1, 2, 4, 8])


def _snap_beats(x: float, carry: float = 0.0) -> tuple[int, float]:
    """Nearest of {1, 2, 4, 8} beats in log2 space, with error diffusion: the rounding remainder carries to
    the next shot (clipped to +/-0.5), so the AVERAGE shot length follows pacing smoothly while every
    individual shot stays a whole 1/2/4/8 beats."""
    lx = np.log2(max(x, 1e-3))
    k = int(np.argmin(np.abs(np.log2(SNAP) - (lx + carry))))
    return int(SNAP[k]), float(np.clip(carry + lx - np.log2(SNAP[k]), -0.5, 0.5))


def _grid_mean(grid: np.ndarray, vals: np.ndarray, a: float, b: float) -> float:
    m = (grid >= a - EPS) & (grid < b - EPS)
    if m.any():
        return float(vals[m].mean())
    return float(vals[int(np.argmin(np.abs(grid - (a + b) / 2)))])


def walk(profile: MusicProfile, pacing: list[float], extra_doublings: dict[int, int] | None = None) -> tuple[list[int], list[bool], list[float]]:
    """Cut beat indices (len(bt) means the excerpt end), forced flags per cut, and the beat times."""
    cfg = get_config().planner
    extra = extra_doublings or {}
    L = profile.excerpt.song_end - profile.excerpt.song_start
    bt = [b for b in profile.beats if b < L - EPS]
    if not bt or bt[0] > EPS:
        bt = [0.0] + bt
    B = len(bt)
    t_at = lambda k: bt[k] if k < B else L  # noqa: E731
    med = float(np.median(np.diff(bt))) if B > 1 else 0.5
    grid, inten = np.asarray(profile.grid_times), np.asarray(profile.intensity)
    secs = profile.sections
    sec_of = lambda t: next((s.idx for s in secs if s.start - EPS <= t < s.end - EPS), len(secs) - 1)  # noqa: E731
    bound_k = []
    if cfg.use_forced_cuts:
        for s in secs[1:]:
            k = int(np.argmin(np.abs(np.asarray(bt) - s.start)))
            if 0 < k < B and (not bound_k or k > bound_k[-1]):
                bound_k.append(k)
    cuts, forced = [0], [False]
    k = 0
    carry = 0.0
    while k < B:
        t = bt[k]
        d = bt[k + 1] - t if k + 1 < B else med
        z = _grid_mean(grid, inten, t, t + d)
        level = float(np.clip((z + 1) / 2.5, 0, 1))
        sec = sec_of(t)
        p = pacing[sec] if 0 <= sec < len(pacing) else 0.0
        n, carry = _snap_beats(8 * 2 ** (-3 * level) * 2 ** (-p), carry)
        n *= 2 ** extra.get(sec, 0)
        while n < 64 and k + n < B and t_at(k + n) - t < cfg.min_shot_s:
            n *= 2
        while n > 1 and t_at(min(k + n, B)) - t > cfg.max_shot_s + EPS:
            n //= 2
        nk = min(k + n, B)
        nb = next((b for b in bound_k if b > k), None)
        is_forced = False
        if nb is not None and nk >= nb:
            nk, is_forced = nb, True
        if not is_forced and nk < B and L - t_at(nk) < cfg.min_shot_s:
            nk = B  # tail too short for its own shot: extend to the end
        if is_forced and t_at(nk) - t < cfg.min_shot_s and len(cuts) > 1:
            cuts[-1], forced[-1] = nk, True  # merge: move the previous cut onto the boundary
            k = nk
            continue
        if nk < B:
            cuts.append(nk)
            forced.append(is_forced)
        k = nk
    cuts.append(B)
    forced.append(False)
    return cuts, forced, bt


def make_slots(profile: MusicProfile, cuts: list[int], forced: list[bool], bt: list[float]) -> list[Slot]:
    cfg = get_config()
    L = profile.excerpt.song_end - profile.excerpt.song_start
    B = len(bt)
    t_at = lambda k: bt[k] if k < B else L  # noqa: E731
    grid = np.asarray(profile.grid_times)
    inten, aro, val, nov = (np.asarray(x) for x in (profile.intensity, profile.arousal, profile.valence, profile.novelty))
    downs = np.asarray(profile.downbeats) if profile.downbeats else np.array([0.0])
    half_frame = 0.5 / cfg.render.fps
    secs = profile.sections
    out = []
    for j in range(len(cuts) - 1):
        a, b = t_at(cuts[j]), t_at(cuts[j + 1])
        sec = next((s.idx for s in secs if s.start - EPS <= a < s.end - EPS), len(secs) - 1)
        if j == 0:
            strength = 0.0
        elif forced[j]:
            strength = 1.0
        else:
            strength = float(cfg.planner.boundary_novelty_scale * np.interp(a, grid, nov))
        out.append(Slot(idx=j, start=round(a, 4), end=round(b, 4), n_beats=int(cuts[j + 1] - cuts[j]),
                        on_downbeat=bool(np.min(np.abs(downs - a)) <= half_frame),
                        intensity=_grid_mean(grid, inten, a, b), arousal=_grid_mean(grid, aro, a, b),
                        valence=_grid_mean(grid, val, a, b), section_idx=int(sec),
                        boundary_strength=round(min(1.0, strength), 4), forced=bool(forced[j] and j > 0)))
    return out


def build_slots(profile: MusicProfile, pacing: list[float] | None, n_usable: int) -> tuple[list[Slot], bool, list[str]]:
    """Returns (slots, reuse_mode, notes). Lengthens the calmest sections first when footage is scarce."""
    cfg = get_config().planner
    pacing = pacing if pacing is not None else [0.0] * len(profile.sections)
    limit = int(np.floor(cfg.max_fill_ratio * n_usable))
    extra: dict[int, int] = {}
    order = sorted(range(len(profile.sections)), key=lambda i: profile.sections[i].mean_intensity) or [0]
    cuts, forced, bt = walk(profile, pacing, extra)
    best = (cuts, forced)
    for _ in range(4 * len(order)):  # each pass doubles one more section, calmest first
        if len(best[0]) - 1 <= limit:
            break
        sec = min(order, key=lambda i: (extra.get(i, 0), order.index(i)))
        extra[sec] = extra.get(sec, 0) + 1
        cuts, forced, bt = walk(profile, pacing, extra)
        if len(cuts) < len(best[0]):
            best = (cuts, forced)
    slots = make_slots(profile, best[0], best[1], bt)
    notes = []
    reuse = len(slots) > n_usable
    if reuse:
        notes.append(f"Only {n_usable} usable shots for {len(slots)} cuts: some shots are reused.")
    elif len(slots) > limit:
        notes.append("Few usable shots for this song length; cuts were lengthened.")
    return slots, reuse, notes
