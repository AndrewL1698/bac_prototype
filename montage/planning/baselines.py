"""Baselines for comparison: naive (fixed 2 s cuts, random shots), beat_random, greedy."""

from __future__ import annotations

import numpy as np

from montage.config import get_config
from montage.planning.costs import INF, CostModel
from montage.types import EDL, Slot


def seq_total(cm: CostModel, seq: list[int]) -> float:
    from montage.planning.beam import sequence_cost
    return sequence_cost(cm, seq)


def _feasible(cm: CostModel, cons, j: int, used: set[int], chosen: list[int]) -> np.ndarray:
    c = get_config().planner
    ok = cm.U[:, j] < INF / 2
    for m in used:
        ok[m] = False
        ok &= ~cons.overlap[m]
    for cl in set(cons.cluster[list(used)]) if used else []:
        if (cons.cluster[list(used)] == cl).sum() >= c.max_per_cluster:
            ok &= cons.cluster != cl
    for a in set(cons.asset[list(used)]) if used else []:
        if (cons.asset[list(used)] == a).sum() >= c.max_windows_per_video:
            ok &= ~((cons.asset == a) & cons.is_video)
    if not ok.any():  # footage exhausted: allow reuse, never back-to-back
        ok = cm.U[:, j] < INF / 2
        if chosen:
            ok[chosen[-1]] = False
    return ok


def random_sequence(cm: CostModel, cons, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    seq: list[int] = []
    used: set[int] = set()
    for j in range(len(cm.slots)):
        ok = np.where(_feasible(cm, cons, j, used, seq))[0]
        m = int(rng.choice(ok)) if len(ok) else 0
        seq.append(m)
        used.add(m)
    return seq


def greedy_sequence(cm: CostModel, cons, imp_pool: np.ndarray) -> list[int]:
    seq: list[int] = []
    used: set[int] = set()
    for j in range(len(cm.slots)):
        ok = _feasible(cm, cons, j, used, seq)
        m = int(np.argmax(np.where(ok, imp_pool, -np.inf)))
        seq.append(m)
        used.add(m)
    return seq


def naive_slots(L: float) -> list[Slot]:
    step = get_config().planner.naive_cut_s
    cuts = list(np.arange(0.0, L - 1e-6, step)) + [L]
    if len(cuts) > 2 and cuts[-1] - cuts[-2] < 0.5:
        cuts.pop(-2)
    return [Slot(idx=j, start=round(a, 4), end=round(b, 4), n_beats=0, on_downbeat=False, intensity=0.0, arousal=0.0,
                 valence=0.0, section_idx=0, boundary_strength=0.0, forced=False)
            for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]


def naive(inp, run_id: str, seed: int, usable: np.ndarray, preview: bool) -> EDL:
    """Random shots on a fixed 2 s grid ignoring beats; no director terms."""
    from montage.planning import finalize
    from montage.planning.beam import make_constraints
    from montage.planning.costs import build_costs
    prof = inp.profile
    L = prof.excerpt.song_end - prof.excerpt.song_start
    slots = naive_slots(L)
    K = len(slots)
    pool = np.where(usable)[0]
    zeros = np.zeros((len(inp.candidates), K), np.float32)
    with_cfg = build_costs(pool, slots, inp.candidates, inp.features.arrays, inp.importance, zeros,
                           np.full(K, 0.5), [None] * len(inp.candidates), False, np.zeros(len(inp.candidates), bool),
                           "heuristic", seed)
    cons = make_constraints(pool, inp.candidates, inp.features.arrays["dup_cluster"], [], K)
    seq = random_sequence(with_cfg, cons, seed)
    return finalize(inp, run_id, seed, "naive", slots, with_cfg, seq, seq_total(with_cfg, seq), [], preview)
