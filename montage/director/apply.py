"""Compile an EditPlan into planner directives: section mapping, pacing, target arousal, query scores."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from montage.config import get_config
from montage.features import clip_encoder
from montage.types import EditPlan, MusicProfile, Slot


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def relevance(emb: np.ndarray, queries: list[str]) -> np.ndarray:
    """rel(i, Q) = max_q cos(e_i, text(q)), z-scored over candidates; zeros for an empty set."""
    qs = [q for q in queries if q and q.strip()]
    if not qs or len(emb) == 0:
        return np.zeros(len(emb), np.float32)
    return _z((emb @ clip_encoder.encode_text(qs).T).max(1)).astype(np.float32)


def rank01(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    return np.full(len(x), 0.5) if len(x) < 2 else (rankdata(x) - 1) / (len(x) - 1)


def map_anchor(anchor: str, section_index: int, profile: MusicProfile) -> list[int]:
    secs = profile.sections
    n = len(secs)
    if n == 0:
        return []
    peak = next((s.idx for s in secs if s.start <= profile.peak_time < s.end), n - 1)
    if anchor == "intro":
        return [0]
    if anchor == "outro":
        return [n - 1]
    if anchor == "peak":
        return [peak]
    if anchor == "build":
        if peak == 0:
            return []
        out = [peak - 1]
        k = peak - 2
        while k >= 0 and secs[k].label == "building":
            out.insert(0, k)
            k -= 1
        return out
    if anchor == "section":
        return [section_index] if 0 <= section_index < n else []
    return []


@dataclass
class Directives:
    plan: EditPlan
    n_sections: int
    section_pacing: list[float]
    section_mood: list[str | None]
    seg_queries: list[list[str]]
    z_vibe: np.ndarray
    z_avoid: np.ndarray
    z_open: np.ndarray
    z_close: np.ndarray
    z_seg: list[np.ndarray]
    excluded: np.ndarray                      # (N,) bool: removed by plan.exclude or UI excludes
    must_assets: list[str]
    pinned: np.ndarray                        # (N,) bool: UI pins (candidate level)
    chronological: bool
    notes: list[str] = field(default_factory=list)

    def target_arousal(self, slots: list[Slot]) -> np.ndarray:
        cfg = get_config().director
        music_rank = rank01(np.array([s.arousal for s in slots]))
        out = np.empty(len(slots))
        for j, s in enumerate(slots):
            mood = self.section_mood[s.section_idx] if 0 <= s.section_idx < self.n_sections else None
            out[j] = cfg.mood_targets[mood] if mood in cfg.mood_targets else music_rank[j]
        return np.clip(out + cfg.energy_bias_scale * self.plan.energy_bias, 0, 1)

    def director_scores(self, slots: list[Slot]) -> np.ndarray:
        """D (N, K) for the given slots."""
        w = get_config().director.weights
        K, N = len(slots), len(self.z_vibe)
        D = np.repeat((w.vibe * self.z_vibe - w.avoid * np.maximum(self.z_avoid - w.avoid_margin, 0))[:, None], K, 1)
        for j, s in enumerate(slots):
            if 0 <= s.section_idx < self.n_sections and self.seg_queries[s.section_idx]:
                D[:, j] += w.seg * self.z_seg[s.section_idx]
        if K:
            D[:, 0] += w.open * self.z_open
            D[:, K - 1] += w.close * self.z_close
        return D.astype(np.float32) if N else np.zeros((0, K), np.float32)


def compile_plan(plan: EditPlan, profile: MusicProfile, features, candidate_assets: list[str],
                 pins: list[str] | None = None, excluded: list[str] | None = None) -> Directives:
    emb = features.arrays["emb"]
    ids = features.ids
    n_sec = len(profile.sections)
    pacing = [max(-1.0, min(1.0, plan.pacing))] * n_sec
    mood: list[str | None] = [None] * n_sec
    seg_q: list[list[str]] = [[] for _ in range(n_sec)]
    for seg in plan.segments:
        for k in map_anchor(seg.anchor, seg.section_index, profile):
            seg_q[k] += [q for q in seg.content_queries if q not in seg_q[k]]
            mood[k] = seg.mood
            if seg.pacing:
                pacing[k] = max(-1.0, min(1.0, seg.pacing))
    excl_assets = set(plan.exclude)
    excl_ids = set(excluded or [])
    ex_mask = np.array([a in excl_assets or c in excl_ids for c, a in zip(ids, candidate_assets)])
    pin_ids = set(pins or [])
    known_assets = set(candidate_assets)
    notes = []
    must = [a for a in plan.must_include if a in known_assets and a not in excl_assets]
    for a in plan.must_include:
        if a not in known_assets:
            notes.append(f"must-include {a} is not in this upload")
    return Directives(
        plan=plan, n_sections=n_sec, section_pacing=pacing, section_mood=mood, seg_queries=seg_q,
        z_vibe=relevance(emb, plan.vibe_queries), z_avoid=relevance(emb, plan.avoid_queries),
        z_open=relevance(emb, [plan.opening_query]), z_close=relevance(emb, [plan.closing_query]),
        z_seg=[relevance(emb, q) for q in seg_q], excluded=ex_mask, must_assets=must,
        pinned=np.array([c in pin_ids for c in ids]), chronological=plan.chronological, notes=notes,
    )
