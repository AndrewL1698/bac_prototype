"""Phase 9A: LLM director fallbacks (mocked), live plans (api), notes rules."""

from __future__ import annotations

import json
import os
from datetime import datetime
from types import SimpleNamespace

import pytest

from montage import director
from montage.director import llm
from montage.director.notes import revise_rules
from montage.director.rules import plan_from_rules
from montage.types import EDL, Excerpt, Shot, ShotScores

GOOD = {"vibe": "sunny beach day", "vibe_queries": ["golden hour beach", "friends laughing"], "avoid_queries": [],
        "pacing": 0.0, "energy_bias": 0.0, "chronological": False, "target_length_s": 0, "excerpt_hint": "auto",
        "excerpt_start_s": 0, "opening_query": "sunrise over the ocean", "closing_query": "",
        "segments": [{"anchor": "peak", "section_index": 3, "content_queries": ["dancing at a party"],
                      "mood": "energetic", "pacing": 0.0}],
        "must_include": ["a001", "a999"], "exclude": [], "rationale": "Beach first, party on the drop."}


class FakeMessages:
    def __init__(self, behavior):
        self.behavior, self.calls = behavior, 0

    def create(self, **kw):
        self.calls += 1
        b = self.behavior
        if b == "timeout":
            raise TimeoutError("timed out")
        if b in ("refusal", "max_tokens"):
            return SimpleNamespace(stop_reason=b, content=[SimpleNamespace(type="text", text="")])
        text = "{not json" if b == "bad_json" else json.dumps(b)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm, "json_schema", lambda: {})

    def use(behavior):
        msgs = FakeMessages(behavior)
        monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(messages=msgs))
        return msgs
    return use


CTX = {"song_summary": "Song: 120 BPM", "inventory": "a000 photo | beach", "tag_vocab": "beach",
       "asset_ids": ["a000", "a001"]}


@pytest.mark.parametrize("behavior", ["timeout", "refusal", "max_tokens", "bad_json"])
def test_llm_fallback_paths(fake, behavior):
    msgs = fake(behavior)
    plan, fb = director.make_plan("chill beach recap", CTX)
    assert plan.source == "rules" and fb == ["director:rules"]
    assert msgs.calls == 2  # one retry before falling back


def test_llm_wrong_case_enums(fake):
    bad = json.loads(json.dumps(GOOD))
    bad["excerpt_hint"] = "AUTO"
    bad["segments"][0]["anchor"] = "Peak"
    bad["segments"][0]["mood"] = "ENERGETIC"
    fake(bad)
    plan, fb = director.make_plan("beach then party", CTX)
    assert plan.source == "llm" and fb == []
    assert plan.excerpt_hint == "auto" and plan.segments[0].anchor == "peak" and plan.segments[0].mood == "energetic"
    assert plan.segments[0].section_index == -1          # only "section" anchors keep an index
    assert plan.must_include == ["a001"]                 # unknown id dropped


def test_llm_repair_clamps():
    d = json.loads(json.dumps(GOOD))
    d.update(pacing=3, energy_bias=-7, target_length_s=90,
             vibe_queries=["one two three four five six seven eight nine ten", "", "a", "b", "c", "d", "e", "f"])
    p = llm.repair(d, ["a000"])
    assert p.pacing == 1 and p.energy_bias == -1 and p.target_length_s == 60
    assert len(p.vibe_queries) <= 6 and all(len(q.split()) <= 8 for q in p.vibe_queries) and "" not in p.vibe_queries


def test_revise_falls_back(fake):
    from montage.director.notes import revise_plan
    fake("timeout")
    prev = plan_from_rules("beach")
    plan, fb = revise_plan(prev, None, "faster", CTX)
    assert fb == ["director:rules"] and plan.pacing == pytest.approx(prev.pacing + 0.4)


def _edl() -> EDL:
    sc = ShotScores(importance=0, mood=0, director=0, unary_total=0, pairwise_in=0)
    times = [0, 3, 6, 10, 13, 16]
    shots = [Shot(slot_idx=j, start=times[j], end=times[j + 1], frame_start=times[j] * 30, frame_end=times[j + 1] * 30,
                  candidate_id=f"a00{j}", asset_id=f"a00{j}", kind="photo", src_in=None, src_out=None,
                  framing="native", kenburns=None, scores=sc, reason="") for j in range(5)]
    return EDL(run_id="r", created=datetime.now(), system="full", seed=0, fps=30, width=1280, height=720, song_hash="h",
               excerpt=Excerpt(song_start=0, song_end=16, fade_out=1.5), plan=plan_from_rules(""), shots=shots,
               total_cost=0)


def test_notes_rules():
    edl = _edl()
    p = revise_rules(edl.plan, edl, "lose the shot at 0:12")
    assert p.exclude == ["a003"]  # 10-13 s
    p2 = revise_rules(edl.plan, edl, "slower at the start, more of the dog")
    assert any(s.anchor == "intro" and s.pacing < 0 for s in p2.segments)
    assert "dog" in p2.vibe_queries
    p3 = revise_rules(edl.plan, edl, "no selfies")
    assert "selfies" in p3.avoid_queries


@pytest.mark.api
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "").startswith("test"),
                    reason="needs ANTHROPIC_API_KEY")
@pytest.mark.parametrize("brief", [
    "Warm, nostalgic summer recap. Open slow on the beach at sunrise, build through the road trip, save the party for "
    "the drop, end on the sunset. Skip anything blurry.",
    "Hype birthday montage, fast cuts, lots of friends and cake, 45 seconds.",
    "Calm, chronological travel diary of our mountain hike; no screenshots; finish on the summit view.",
])
def test_llm_live(brief):
    inventory = "\n".join([
        "a000 photo 2026-07-14 | beach, ocean, sunrise", "a001 video 0:20 2026-07-14 | road trip, car, highway",
        "a002 photo 2026-07-14 | group of friends, party, dancing", "a003 photo 2026-07-15 | sunset, beach",
        "a004 video 0:15 2026-07-15 | mountain, hiking, forest", "a005 photo 2026-07-15 | cake, birthday party",
        "a006 photo 2026-07-16 | screenshot, text document", "a007 photo 2026-07-16 | dog, park"])
    ctx = {"song_summary": "Song: 118 BPM, length 3:30; excerpt 0:50 (starts at song 1:02). Peak at 0:36.\n"
                           "S0 0.0-12.0s low\nS1 12.0-28.0s building\nS2 28.0-44.0s peak\nS3 44.0-50.0s falling",
           "inventory": inventory, "tag_vocab": "beach, ocean, sunrise, sunset, party, dancing, mountain, cake",
           "asset_ids": [f"a00{i}" for i in range(8)]}
    plan = llm.plan_from_llm(brief, ctx)
    assert plan.source == "llm" and plan.rationale
    for q in plan.vibe_queries + plan.avoid_queries + [q for s in plan.segments for q in s.content_queries]:
        assert 0 < len(q.split()) <= 8
    assert set(plan.must_include + plan.exclude) <= set(ctx["asset_ids"])
    print(plan.model_dump_json(indent=1))
