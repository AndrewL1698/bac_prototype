"""Edit metrics from an EDL + MusicProfile + features (spec 11 section 2).

Cut times are the internal cuts (0 and L excluded). The beat-hit rate of `full` is high by construction;
its point is the contrast with `naive`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from montage.config import get_config
from montage.types import EDL, MusicProfile, Slot


def internal_cuts(edl: EDL) -> np.ndarray:
    return np.array([s.start for s in edl.shots[1:]], float)


def _nearest(t: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(ref, float))
    if len(ref) == 0 or len(t) == 0:
        return np.full(len(t), np.inf)
    i = np.clip(np.searchsorted(ref, t), 1, len(ref) - 1)
    return np.minimum(np.abs(t - ref[i - 1]), np.abs(t - ref[i]))


def hit_rate(cuts: np.ndarray, ref: list[float], tol: float) -> float:
    return float(np.mean(_nearest(cuts, np.asarray(ref)) <= tol + 1e-9)) if len(cuts) else float("nan")


def boundary_hit_rate(cuts: np.ndarray, profile: MusicProfile) -> float:
    bounds = np.array([s.start for s in profile.sections[1:]], float)
    if len(bounds) == 0:
        return float("nan")
    beat = float(np.median(np.diff(profile.beats))) if len(profile.beats) > 1 else 0.5
    return float(np.mean(_nearest(bounds, cuts) <= beat + 1e-6))


def pace_coupling(cuts: np.ndarray, profile: MusicProfile, window: float | None = None) -> float:
    w = window or get_config().eval.pace_window_s
    grid = np.asarray(profile.grid_times)
    rate = np.array([np.sum((cuts >= t - w / 2) & (cuts < t + w / 2)) / w for t in grid])
    inten = np.asarray(profile.intensity)
    if np.std(rate) == 0 or np.std(inten) == 0:
        return 0.0
    return float(spearmanr(rate, inten).statistic)


def _grid_mean(profile: MusicProfile, key: str, a: float, b: float) -> float:
    g = np.asarray(profile.grid_times)
    v = np.asarray(getattr(profile, key))
    m = (g >= a) & (g < b)
    return float(v[m].mean()) if m.any() else float(v[np.argmin(np.abs(g - (a + b) / 2))])


def compute(edl: EDL, profile: MusicProfile, features, importance: np.ndarray, n_assets: int,
            D: np.ndarray | None = None) -> dict:
    """features: montage.features.extract.Features for the session; importance: final scores (N,);
    D: optional (N, K) director scores for the EDL's shots as slots."""
    cfg = get_config().eval
    idx = features.index()
    chosen = np.array([idx[s.candidate_id] for s in edl.shots])
    cuts = internal_cuts(edl)
    off = _nearest(cuts, np.asarray(profile.beats))
    emb = features.arrays["emb"][chosen]
    cos = emb @ emb.T
    K = len(chosen)
    iu = np.triu_indices(K, 1)
    adj = np.array([cos[j, j + 1] for j in range(K - 1)]) if K > 1 else np.zeros(0)
    dup = features.arrays["dup_cluster"][chosen]
    same_dup = int(sum(1 for a, b in zip(*iu) if dup[a] == dup[b]))
    va = features.arrays["visual_arousal"][chosen]
    music_a = np.array([_grid_mean(profile, "arousal", s.start, s.end) for s in edl.shots])
    mood = float(spearmanr(va, music_a).statistic) if K > 2 and np.std(va) > 0 and np.std(music_a) > 0 else 0.0
    out = {
        "n_shots": K,
        "beat_hit_rate": hit_rate(cuts, profile.beats, cfg.beat_tol_s),
        "downbeat_hit_rate": hit_rate(cuts, profile.downbeats, cfg.beat_tol_s),
        "mean_offset_ms": float(np.mean(off) * 1000) if len(off) else float("nan"),
        "boundary_hit_rate": boundary_hit_rate(cuts, profile),
        "pace_coupling": pace_coupling(cuts, profile),
        "mood_coupling": mood,
        "diversity": float(cos[iu].mean()) if len(iu[0]) else float("nan"),
        "max_adjacent_cos": float(adj.max()) if len(adj) else float("nan"),
        "dup_violations": same_dup,
        "coverage": float(len({s.asset_id for s in edl.shots}) / max(n_assets, 1)),
        "same_source_adjacent": int(sum(edl.shots[j].asset_id == edl.shots[j + 1].asset_id for j in range(K - 1))),
        "quality_lift": float(np.mean(importance[chosen]) - np.mean(importance)),
    }
    if D is not None and D.size:
        out["director_adherence"] = float(np.mean(D[chosen, np.arange(K)]) - np.mean(D))
    return out


def slots_from_edl(edl: EDL, profile: MusicProfile) -> list[Slot]:
    """Minimal Slots for the EDL's shots (used to score director adherence on any system's cut)."""
    secs = profile.sections
    out = []
    for j, s in enumerate(edl.shots):
        sec = next((x.idx for x in secs if x.start - 1e-6 <= s.start < x.end - 1e-6), len(secs) - 1)
        out.append(Slot(idx=j, start=s.start, end=s.end, n_beats=0, on_downbeat=False,
                        intensity=_grid_mean(profile, "intensity", s.start, s.end),
                        arousal=_grid_mean(profile, "arousal", s.start, s.end),
                        valence=_grid_mean(profile, "valence", s.start, s.end), section_idx=sec,
                        boundary_strength=0.0, forced=False))
    return out
