"""Excerpt selection: a whole number of bars in [45, 60] s, starting on a downbeat."""

from __future__ import annotations

import numpy as np

from montage.config import get_config
from montage.music.profile import FullAnalysis


def excerpt_length(fa: FullAnalysis, target_s: float = 0.0) -> float:
    c = get_config().music.excerpt
    bar = fa.bar
    target = target_s or c.default_length_s
    n = max(1, int(round(target / bar)))
    while n * bar < c.min_length_s - 1e-6:
        n += 1
    while n * bar > c.max_length_s + 1e-6 and n > 1:
        n -= 1
    first = fa.downbeats[0] if fa.downbeats else 0.0
    usable = fa.duration - first
    if n * bar > usable - 2.0:  # short song: longest whole-bar length that fits
        n = max(1, int((usable - 0.05) // bar))
    return n * bar


def _scores(fa: FullAnalysis, starts: np.ndarray, L: float) -> dict[str, np.ndarray]:
    c = get_config().music.excerpt
    grid, inten, rms = fa.a("grid"), fa.a("intensity"), fa.a("rms_db")
    z = (inten - inten.mean()) / (inten.std() + 1e-9)
    bar = fa.bar
    bounds = np.asarray(fa.boundaries)
    out = {k: np.zeros(len(starts)) for k in ("mean", "peak", "start", "end", "silence")}
    for i, s in enumerate(starts):
        m = (grid >= s) & (grid < s + L)
        if not m.any():
            continue
        zz, tt = z[m], grid[m]
        out["mean"][i] = zz.mean()
        p = (tt[int(np.argmax(zz))] - s) / L
        out["peak"][i] = np.exp(-((p - c.peak_target_frac) / c.peak_width) ** 2)
        near = lambda t: bool(len(bounds)) and np.min(np.abs(bounds - t)) <= bar  # noqa: E731
        out["start"][i] = 1.0 if (near(s) or s <= bar) else 0.0
        e = s + L
        tail = zz[tt >= e - 2.0]
        before = zz[(tt >= e - 4.0) & (tt < e - 2.0)]
        falling = len(tail) and len(before) and tail.mean() < before.mean()
        out["end"][i] = 1.0 if (near(e) or falling or e >= fa.duration - bar) else 0.0
        head = rms[(grid >= s) & (grid < s + 2.0)]
        out["silence"][i] = 1.0 if len(head) and head.mean() < c.silence_db else 0.0
    return out


def choose_excerpt(fa: FullAnalysis, target_length_s: float = 0.0, hint: str = "auto", start_s: float = 0.0):
    from montage.types import Excerpt
    c = get_config().music.excerpt
    L = excerpt_length(fa, target_length_s)
    downs = np.asarray(fa.downbeats if fa.downbeats else fa.beats[::4])
    starts = downs[downs + L <= fa.duration + 1e-6]
    if len(starts) == 0:
        starts = downs[:1] if len(downs) else np.array([0.0])
    sc = _scores(fa, starts, L)
    w = c.weights
    total = w.mean * sc["mean"] + w.peak * sc["peak"] + w.start * sc["start"] + w.end * sc["end"] - w.silence * sc["silence"]
    if hint == "start":
        ok = np.where(sc["silence"] < 0.5)[0]
        i = int(ok[0]) if len(ok) else 0
    elif hint == "peak":
        i = int(np.lexsort((-sc["mean"], -sc["peak"]))[0])
    elif hint == "custom":
        i = int(np.argmin(np.abs(starts - start_s)))
    else:
        i = int(np.argmax(total))
    s = float(starts[i])
    return Excerpt(song_start=round(s, 4), song_end=round(s + L, 4), fade_out=c.fade_out_s)
