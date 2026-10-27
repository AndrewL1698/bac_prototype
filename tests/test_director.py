"""Phase 4 acceptance: rule-based director and plan compilation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from montage.director.apply import compile_plan, map_anchor
from montage.director.rules import plan_from_rules
from montage.types import EditPlan, Excerpt, MusicProfile, Section, SegmentDirective, Slot


def profile4() -> MusicProfile:
    secs = [Section(idx=0, start=0, end=10, label="low", mean_intensity=-1),
            Section(idx=1, start=10, end=24, label="building", mean_intensity=0),
            Section(idx=2, start=24, end=40, label="peak", mean_intensity=1),
            Section(idx=3, start=40, end=48, label="falling", mean_intensity=0)]
    grid = list(np.arange(0, 48, 0.5))
    return MusicProfile(song_hash="h", song_duration=200, tempo=120, beat_mode="normal",
                        excerpt=Excerpt(song_start=0, song_end=48, fade_out=1.5), beats=list(np.arange(0, 48, 0.5)),
                        downbeats=list(np.arange(0, 48, 2.0)), grid_times=grid, intensity=[0.0] * len(grid),
                        arousal=[0.0] * len(grid), valence=[0.0] * len(grid), novelty=[0.0] * len(grid),
                        sections=secs, peak_time=30.0, emotion_source="heuristic", beat_source="librosa")


def plan(**kw) -> EditPlan:
    base = dict(vibe="", vibe_queries=[], avoid_queries=[], pacing=0, energy_bias=0, chronological=False,
                target_length_s=0, excerpt_hint="auto", excerpt_start_s=0, opening_query="", closing_query="",
                segments=[], must_include=[], exclude=[], rationale="")
    base.update(kw)
    return EditPlan(**base)


def slot(j, sec, arousal=0.0):
    return Slot(idx=j, start=j, end=j + 1, n_beats=2, on_downbeat=True, intensity=0, arousal=arousal, valence=0,
                section_idx=sec, boundary_strength=0, forced=False)


def feats(n=8):
    rng = np.random.default_rng(0)
    e = rng.normal(size=(n, 768)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return SimpleNamespace(arrays={"emb": e}, ids=[f"a00{i // 2}_w0{i % 2}" for i in range(n)])


def test_rules_parse():
    p = plan_from_rules("chill, chronological, skip anything blurry, end on the sunset, 45 seconds")
    assert p.pacing == -0.5
    assert p.chronological is True
    assert p.avoid_queries == ["blurry photo"]
    assert p.closing_query == "the sunset"
    assert p.target_length_s == 45
    assert p.source == "rules"
    assert all("blurry" not in q and "end on" not in q for q in p.vibe_queries)


def test_rules_more():
    p = plan_from_rules("Hype birthday party, open on the cake, no screenshots, 90 seconds")
    assert p.pacing == 0.5 and p.opening_query == "the cake"
    assert "screenshot" in p.avoid_queries and p.target_length_s == 60
    assert "birthday party" in p.vibe_queries
    assert plan_from_rules("").vibe_queries == []


def test_anchor_mapping():
    pr = profile4()
    assert map_anchor("intro", -1, pr) == [0]
    assert map_anchor("build", -1, pr) == [1]
    assert map_anchor("peak", -1, pr) == [2]
    assert map_anchor("outro", -1, pr) == [3]
    assert map_anchor("section", 1, pr) == [1] and map_anchor("section", 9, pr) == []


def test_target_arousal():
    seg = SegmentDirective(anchor="intro", section_index=-1, content_queries=[], mood="calm", pacing=0)
    f = feats()
    assets = [i.split("_")[0] for i in f.ids]
    d = compile_plan(plan(segments=[seg]), profile4(), f, assets)
    assert d.target_arousal([slot(0, 0), slot(1, 2)])[0] == pytest.approx(0.15)
    d2 = compile_plan(plan(segments=[seg], energy_bias=1.0), profile4(), f, assets)
    assert d2.target_arousal([slot(0, 0), slot(1, 2)])[0] == pytest.approx(0.4)


def test_section_pacing():
    seg = SegmentDirective(anchor="peak", section_index=-1, content_queries=[], mood="match_music", pacing=0.8)
    f = feats()
    d = compile_plan(plan(segments=[seg], pacing=-0.3), profile4(), f, [i.split("_")[0] for i in f.ids])
    assert d.section_pacing == [-0.3, -0.3, 0.8, -0.3]


def test_director_scores_shape(monkeypatch):
    from montage.director import apply
    rng = np.random.default_rng(1)
    monkeypatch.setattr(apply.clip_encoder, "encode_text",
                        lambda qs: (lambda e: e / np.linalg.norm(e, axis=1, keepdims=True))(rng.normal(size=(len(qs), 768))))
    f = feats()
    assets = [i.split("_")[0] for i in f.ids]
    seg = SegmentDirective(anchor="peak", section_index=-1, content_queries=["friends dancing"], mood="energetic", pacing=0)
    d = compile_plan(plan(vibe_queries=["beach"], avoid_queries=["blurry photo"], opening_query="sunrise",
                          segments=[seg], exclude=["a001"]), profile4(), f, assets)
    slots = [slot(j, s) for j, s in enumerate([0, 1, 2, 2, 3])]
    D = d.director_scores(slots)
    assert D.shape == (len(f.ids), len(slots))
    assert d.excluded.tolist() == [a == "a001" for a in assets]
    assert not np.allclose(D[:, 0], D[:, 1])  # opening term only on the first slot
