"""Phase 5 acceptance: slots, costs, beam search, EDL (synthetic inputs, runs anywhere)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr

from montage import config
from montage.director.apply import compile_plan
from montage.planning import PlanInputs, plan_edit
from montage.planning.beam import make_constraints, objective, solve
from montage.planning.baselines import greedy_sequence
from montage.planning.costs import build_costs
from montage.planning.edl import frame_bounds
from montage.planning.slots import build_slots
from montage.types import Asset, Candidate, EditPlan, Excerpt, MusicProfile, Section


def make_profile(L=48.0, bpm=120.0, bounds=(16.0, 32.0)) -> MusicProfile:
    beats = list(np.round(np.arange(0, L, 60 / bpm), 4))
    grid = np.arange(0, L, 0.5)
    inten = np.interp(grid, [0, L], [-1.2, 1.5])
    edges = [0.0, *bounds, L]
    labels = ["low", "building", "peak", "high", "falling"]
    secs = [Section(idx=i, start=edges[i], end=edges[i + 1], label=labels[i], mean_intensity=float(np.mean(
        inten[(grid >= edges[i]) & (grid < edges[i + 1])]))) for i in range(len(edges) - 1)]
    nov = np.zeros(len(grid))
    return MusicProfile(song_hash="h", song_duration=200, tempo=bpm, beat_mode="normal",
                        excerpt=Excerpt(song_start=10.0, song_end=10.0 + L, fade_out=1.5), beats=beats,
                        downbeats=beats[::4], grid_times=list(grid), intensity=list(inten), arousal=list(inten / 2),
                        valence=[0.0] * len(grid), novelty=list(nov), sections=secs, peak_time=float(L - 1),
                        emotion_source="heuristic", beat_source="librosa")


def plan(**kw) -> EditPlan:
    base = dict(vibe="", vibe_queries=[], avoid_queries=[], pacing=0, energy_bias=0, chronological=False,
                target_length_s=0, excerpt_hint="auto", excerpt_start_s=0, opening_query="", closing_query="",
                segments=[], must_include=[], exclude=[], rationale="")
    base.update(kw)
    return EditPlan(**base)


def make_inputs(n_photo=30, n_video=8, seed=0, profile=None, p=None, pins=None) -> PlanInputs:
    rng = np.random.default_rng(seed)
    assets, cands = [], []
    t0 = datetime(2026, 7, 1)
    for i in range(n_photo):
        assets.append(Asset(id=f"a{i:03d}", kind="photo", src_path="x.jpg", hash=f"p{i}", width=1600, height=1200,
                            filename=f"p{i}.jpg", taken_at=t0 + timedelta(minutes=i)))
        cands.append(Candidate(id=f"a{i:03d}", asset_id=f"a{i:03d}", kind="photo", thumb_path="/nonexistent.jpg"))
    for v in range(n_video):
        aid = f"a{n_photo + v:03d}"
        assets.append(Asset(id=aid, kind="video", src_path="v.mp4", hash=f"v{v}", width=1920, height=1080,
                            duration=12.0, fps=30, filename=f"v{v}.mp4", taken_at=t0 + timedelta(hours=1, minutes=v)))
        for k in range(6):
            s = 0.3 + 1.5 * k
            cands.append(Candidate(id=f"{aid}_w{k:02d}", asset_id=aid, kind="video", src_start=s, src_end=s + 3,
                                   src_max_end=11.7, scene_id=f"{aid}_s0", frame_times=[], thumb_path="/nonexistent.jpg"))
    N = len(cands)
    emb = rng.normal(size=(N, 768)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    arr = {"emb": emb, "visual_arousal": rng.normal(size=N), "visual_valence": rng.normal(size=N),
           "brightness": rng.uniform(0.2, 0.8, N), "dup_cluster": np.arange(N), "face_cx": np.full(N, -1.0),
           "face_cy": np.full(N, -1.0), "sharpness": rng.normal(6, 0.5, N), "exposure_clip": np.zeros(N),
           "is_video": np.array([c.kind == "video" for c in cands])}
    frames = {a.id: {"times": np.arange(0, 12, 0.5), "emb": emb[:24] if N >= 24 else np.repeat(emb[:1], 24, 0),
                     "aesthetic": rng.normal(size=24), "motion": rng.uniform(0, 0.01, 24)}
              for a in assets if a.kind == "video"}
    feats = SimpleNamespace(arrays=arr, ids=[c.id for c in cands], frames=frames,
                            table=SimpleNamespace(tags={c.id: [] for c in cands}))
    prof = profile or make_profile()
    d = compile_plan(p or plan(), prof, feats, [c.asset_id for c in cands], pins=pins)
    imp = rng.uniform(0, 1, N)
    return PlanInputs(assets=assets, candidates=cands, features=feats, importance=imp, heuristic=imp,
                      profile=prof, directives=d)


def test_slots_invariants():
    prof = make_profile()
    slots, reuse, _ = build_slots(prof, [0.0] * 3, 200)
    L = 48.0
    assert abs(sum(s.end - s.start for s in slots) - L) < 1e-6
    beats = np.array(prof.beats + [L])
    for s in slots:
        assert np.min(np.abs(beats - s.start)) < 1e-6
    for s in slots[:-1]:
        assert 0.5 - 1e-6 <= s.end - s.start <= 4.0 + 1e-6
    starts = [s.start for s in slots]
    for b in (16.0, 32.0):
        assert b in starts and slots[starts.index(b)].forced
    assert not reuse
    assert all(slots[i].end == slots[i + 1].start for i in range(len(slots) - 1))


def test_slots_pacing():
    prof = make_profile()
    fast, _, _ = build_slots(prof, [1.0] * 3, 500)
    slow, _, _ = build_slots(prof, [-1.0] * 3, 500)
    assert len(fast) > len(slow)


def test_slots_pacing_small_steps():
    """A +0.5 pacing note must shorten the average shot even though shots snap to 1/2/4/8 beats."""
    prof = make_profile()
    counts = [len(build_slots(prof, [p] * 3, 500)[0]) for p in (-0.5, 0.0, 0.5)]
    assert counts[0] < counts[1] < counts[2], counts


def test_slots_scarcity():
    slots, reuse, notes = build_slots(make_profile(), [0.0] * 3, 10)
    assert len(slots) <= 9 or reuse


def test_beam_constraints():
    inp = make_inputs(p=plan(exclude=["a003", "a031"], must_include=["a017"]))
    inp.importance[16] = 0.0  # a017: least important, must still appear
    edl = plan_edit(inp, "r", seed=0)
    ids = [s.candidate_id for s in edl.shots]
    assert len(ids) == len(set(ids))
    assets = [s.asset_id for s in edl.shots]
    assert "a003" not in assets and "a031" not in assets
    assert "a017" in assets
    by_asset: dict[str, list] = {}
    for s in edl.shots:
        if s.kind == "video":
            by_asset.setdefault(s.asset_id, []).append((s.src_in, s.src_out))
    for ivs in by_asset.values():
        assert len(ivs) <= 3
        ivs.sort()
        assert all(ivs[i][1] <= ivs[i + 1][0] + 1e-6 for i in range(len(ivs) - 1))


def test_cluster_limit():
    inp = make_inputs()
    arr = inp.features.arrays
    arr["dup_cluster"][:6] = 0  # six near-duplicate photos
    inp.importance[:6] = 1.0
    edl = plan_edit(inp, "r", seed=0)
    chosen = [s.candidate_id for s in edl.shots]
    assert sum(c in {f"a{i:03d}" for i in range(6)} for c in chosen) <= 1


def test_pins():
    inp = make_inputs(pins=["a005"])
    inp.importance[5] = 0.0
    assert "a005" in [s.candidate_id for s in plan_edit(inp, "r", seed=0).shots]


def test_beam_determinism():
    inp = make_inputs()
    a = plan_edit(inp, "r", seed=1)
    b = plan_edit(inp, "r", seed=1)
    c = plan_edit(inp, "r", seed=2)
    seq = lambda e: [s.candidate_id for s in e.shots]  # noqa: E731
    assert seq(a) == seq(b)
    assert seq(a) != seq(c)


def _model(inp, seed=0):
    slots, _, _ = build_slots(inp.profile, inp.directives.section_pacing, len(inp.candidates))
    N = len(inp.candidates)
    pool = np.arange(N)
    D = inp.directives.director_scores(slots)
    taken = [None] * N
    cm = build_costs(pool, slots, inp.candidates, inp.features.arrays, inp.importance, D,
                     inp.directives.target_arousal(slots), taken, False, np.zeros(N, bool), "heuristic", seed)
    cons = make_constraints(pool, inp.candidates, inp.features.arrays["dup_cluster"], [], len(slots))
    return slots, cm, cons


def test_beam_beats_greedy():
    inp = make_inputs()
    _, cm, cons = _model(inp)
    res = solve(cm, cons)
    g = greedy_sequence(cm, cons, inp.importance)
    assert res.total <= objective(cm, cons, g) + 1e-6
    assert abs(res.total - objective(cm, cons, res.seq)) < 1e-6


def test_mood_alignment():
    inp = make_inputs(n_photo=60, n_video=0)
    inp.importance[:] = 0.5
    with config.override({"planner.w_mood": 3.0, "planner.w_imp": 0.0, "planner.variety": 0.0,
                          "planner.w_rep": 0.0, "planner.w_jump": 0.0, "planner.w_contrast": 0.0}):
        slots, cm, cons = _model(inp)
        res = solve(cm, cons)
    vr = cm.parts["vr"][res.seq]
    rho = spearmanr(vr, cm.parts["ta"]).correlation
    assert rho > 0.5, rho


def test_boundary_contrast():
    inp = make_inputs(n_photo=60, n_video=0)
    with config.override({"planner.w_jump": 2.0, "planner.w_contrast": 2.0, "planner.w_mood": 0.0}):
        slots, cm, cons = _model(inp)
        res = solve(cm, cons)
    vj = [cm.vj[res.seq[j - 1], res.seq[j]] for j in range(1, len(slots))]
    forced = [v for v, s in zip(vj, slots[1:]) if s.forced]
    ordinary = [v for v, s in zip(vj, slots[1:]) if not s.forced]
    assert forced and np.mean(forced) > np.mean(ordinary)


def test_frame_quantization():
    from montage.types import Slot
    rng = np.random.default_rng(0)
    cuts = np.sort(rng.uniform(0, 50, 39))
    times = np.r_[0.0, cuts, 50.123]
    slots = [Slot(idx=j, start=float(times[j]), end=float(times[j + 1]), n_beats=1, on_downbeat=False, intensity=0,
                  arousal=0, valence=0, section_idx=0, boundary_strength=0, forced=False) for j in range(40)]
    f = frame_bounds(slots, 30)
    assert sum(f[k + 1] - f[k] for k in range(40)) == round(50.123 * 30)
    assert all(f[k] == round(times[k] * 30) for k in range(41))


def test_edl_fields():
    inp = make_inputs()
    edl = plan_edit(inp, "run1", seed=0)
    assert edl.shots[0].frame_start == 0 and edl.shots[-1].frame_end == round(48.0 * 30)
    assert all(s.reason for s in edl.shots)
    for s in edl.shots:
        if s.kind == "photo":
            # 4:3 loses 25% when cropped (> render.max_crop_loss 0.2), so it is fill_blur; photos get Ken Burns
            assert s.framing == "fill_blur" and s.kenburns is not None
        else:
            assert s.framing == "native" and s.src_out - s.src_in == pytest.approx(s.end - s.start, abs=1e-3)


@pytest.mark.parametrize("system", ["naive", "beat_random", "greedy"])
def test_baselines(system):
    edl = plan_edit(make_inputs(), "r", seed=0, system=system)
    assert edl.system == system and edl.shots
    if system == "naive":
        assert all(abs((s.end - s.start) - 2.0) < 1e-6 for s in edl.shots[:-1])


def test_beam_speed():
    inp = make_inputs(n_photo=120, n_video=0, profile=make_profile(L=60.0), p=plan(pacing=1.0))
    slots, cm, cons = _model(inp)
    pool = np.arange(120)
    assert len(slots) >= 35
    with config.override({"planner.beam_width": 128}):
        t0 = time.time()
        solve(cm, cons)
        dt = time.time() - t0
    assert dt < 2.0, dt
    assert len(pool) == 120
