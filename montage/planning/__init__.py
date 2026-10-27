"""Planner: excerpt profile + directives -> slots -> pool -> costs -> beam/baseline -> in/out -> EDL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from montage.config import get_config
from montage.director.apply import Directives
from montage.features import clip_encoder
from montage.importance.heuristic import hard_rejects
from montage.planning import baselines
from montage.planning.beam import make_constraints, solve
from montage.planning.costs import INF, build_costs, rank01
from montage.planning.edl import assemble, make_reason
from montage.planning.inout import choose_in
from montage.planning.slots import build_slots
from montage.render.framing import decide_framing, kenburns
from montage.types import EDL, Asset, Candidate, MusicProfile, ShotScores, UserError


@dataclass
class PlanInputs:
    assets: list[Asset]
    candidates: list[Candidate]
    features: object            # montage.features.extract.Features
    importance: np.ndarray      # (N,) final importance
    heuristic: np.ndarray       # (N,) raw heuristic (for rejects)
    profile: MusicProfile
    directives: Directives


def select_pool(score: np.ndarray, usable: np.ndarray, required: np.ndarray, K: int) -> np.ndarray:
    M = max(3 * K, get_config().planner.min_pool)
    order = [i for i in np.argsort(-score) if usable[i]]
    pool = list(dict.fromkeys(order[:M] + [i for i in np.where(required & usable)[0]]))
    return np.array(sorted(pool), int)


def best_query(d: Directives, emb: np.ndarray, section: int, j: int, K: int) -> str | None:
    qs = []
    if j == 0 and d.plan.opening_query:
        qs.append(d.plan.opening_query)
    if j == K - 1 and d.plan.closing_query:
        qs.append(d.plan.closing_query)
    if 0 <= section < d.n_sections:
        qs += d.seg_queries[section]
    qs += d.plan.vibe_queries
    qs = [q for q in qs if q][:8]
    if not qs:
        return None
    sims = (np.atleast_2d(emb) @ clip_encoder.encode_text(qs).T)[0]
    k = int(np.argmax(sims))
    return qs[k] if sims[k] > 0.2 and len(qs[k].split()) <= 8 else None


def plan_edit(inp: PlanInputs, run_id: str, seed: int = 0, system: str = "full", preview: bool = True) -> EDL:
    cfg = get_config()
    cands, f, d, prof = inp.candidates, inp.features, inp.directives, inp.profile
    arr = f.arrays
    N = len(cands)
    notes = list(d.notes)
    asset_by_id = {a.id: a for a in inp.assets}
    must_c = np.array([c.asset_id in set(d.must_assets) for c in cands]) | d.pinned
    rej = hard_rejects(arr, inp.heuristic, n_needed=0)
    usable = ~d.excluded & (~rej | d.pinned | must_c)
    pacing = d.section_pacing if system not in ("naive",) else None
    slots, reuse, slot_notes = build_slots(prof, pacing, int(usable.sum()))
    K = len(slots)
    if int(usable.sum()) < K:  # bring back the best rejects before reusing shots
        rej = hard_rejects(arr, inp.heuristic, n_needed=K)
        usable = ~d.excluded & (~rej | d.pinned | must_c)
        slots, reuse, slot_notes = build_slots(prof, pacing, int(usable.sum()))
        K = len(slots)
    notes += slot_notes
    if usable.sum() == 0:
        raise UserError("No usable photos or videos left after exclusions.")

    if system == "naive":
        return baselines.naive(inp, run_id, seed, usable, preview)

    D = d.director_scores(slots)
    pool = select_pool(inp.importance + 0.5 * D.max(1), usable, must_c, K)
    ta = d.target_arousal(slots)
    taken = [asset_by_id[c.asset_id].taken_at for c in cands]
    cm = build_costs(pool, slots, cands, arr, inp.importance, D, ta, taken, d.chronological, must_c,
                     prof.emotion_source, seed)
    cons = make_constraints(pool, cands, arr["dup_cluster"], d.must_assets, K)
    if system == "beat_random":
        seq = baselines.random_sequence(cm, cons, seed)
        total = baselines.seq_total(cm, seq)
    elif system == "greedy":
        seq = baselines.greedy_sequence(cm, cons, inp.importance[pool])
        total = baselines.seq_total(cm, seq)
    else:
        res = solve(cm, cons, reuse_mode=reuse)
        seq, total = res.seq, res.total
        if res.reuse_mode and not reuse:
            notes.append("Some shots are reused (not enough distinct usable footage).")
    missing = [a for a in d.must_assets if a not in {cands[pool[m]].asset_id for m in seq}]
    if missing:
        notes.append(f"Could not fit must-include assets: {', '.join(missing)}")
    return finalize(inp, run_id, seed, system, slots, cm, seq, total, notes, preview)


def finalize(inp: PlanInputs, run_id: str, seed: int, system: str, slots, cm, seq: list[int], total: float,
             notes: list[str], preview: bool) -> EDL:
    cands, f, d, prof = inp.candidates, inp.features, inp.directives, inp.profile
    arr = f.arrays
    K = len(slots)
    asset_by_id = {a.id: a for a in inp.assets}
    irank = rank01(np.array([s.intensity for s in slots]))
    imp_rank = rank01(inp.importance)
    rng = np.random.default_rng(seed + 7)
    taken_iv: dict[str, list[tuple[float, float]]] = {}
    picks = []
    for j, (s, m) in enumerate(zip(slots, seq)):
        i = int(cm.pool[m])
        c = cands[i]
        a = asset_by_id[c.asset_id]
        dur = s.end - s.start
        src_in = src_out = None
        if c.kind == "video":
            qs = (d.seg_queries[s.section_idx] if 0 <= s.section_idx < d.n_sections else []) or d.plan.vibe_queries
            qemb = clip_encoder.encode_text(qs) if qs else None
            src_in, free = choose_in(c, dur, f.frames[c.asset_id], qemb, taken_iv.get(c.asset_id, []))
            if not free:
                notes.append(f"{c.asset_id}: overlapping source time reused")
            src_out = round(src_in + dur, 3)
            taken_iv.setdefault(c.asset_id, []).append((src_in, src_out))
        framing = decide_framing(a)
        kb = kenburns(a, framing, float(irank[j]), float(arr["face_cx"][i]), float(arr["face_cy"][i]), rng)
        pair_in = float(cm.pair(np.array([seq[j - 1]]), j)[0, m]) if j > 0 else 0.0
        u = float(cm.U[m, j]) if cm.U[m, j] < INF / 2 else 0.0
        scores = ShotScores(importance=round(float(cm.parts["imp_raw"][m]), 4),
                            mood=round(float(cm.parts["mood"][m, j]), 4),
                            director=round(float(cm.parts["D_raw"][m, j]), 4),
                            unary_total=round(u, 4), pairwise_in=round(pair_in, 4))
        label = prof.sections[s.section_idx].label if 0 <= s.section_idx < len(prof.sections) else "middle"
        q = best_query(d, arr["emb"][i:i + 1], s.section_idx, j, K) if float(cm.parts["D_raw"][m, j]) > 0.3 else None
        reason = make_reason(float(imp_rank[i]), abs(float(cm.parts["vr"][m]) - float(cm.parts["ta"][j])), label, q,
                             j == 0 and bool(d.plan.opening_query) and float(d.z_open[i]) > 0.5,
                             j == K - 1 and bool(d.plan.closing_query) and float(d.z_close[i]) > 0.5,
                             float(cm.parts["vr"][m]))
        picks.append({"cand": c, "src_in": None if src_in is None else round(src_in, 3), "src_out": src_out,
                      "framing": framing, "kenburns": kb, "scores": scores, "reason": reason})
    return assemble(run_id, system, seed, prof.song_hash, prof.excerpt, d.plan, slots, picks, total, notes, preview)
