"""Phase 0 acceptance: config, types, atomic writes, paths."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from montage import config, paths
from montage import types as T


def test_config_loads():
    cfg = config.load_config()
    assert cfg.render.fps == 30
    assert cfg.planner.beam_width == 128
    assert cfg.music.excerpt.weights.peak == 1.0
    assert "arousal_high" in cfg.mood.prompts.model_dump()


def test_config_dotted_override():
    cfg = config.load_config(overrides=["planner.beam_width=64", "director.use_llm=false"])
    assert cfg.planner.beam_width == 64
    assert cfg.director.use_llm is False
    cfg2 = config.load_config(overrides={"render.preview.crf": 30})
    assert cfg2.render.preview.crf == 30
    with pytest.raises(Exception):
        config.load_config(overrides=["nosuch.section.key=1"])


def test_config_override_context():
    base = config.get_config().planner.w_mood
    with config.override({"planner.w_mood": 0.0}) as c:
        assert c.planner.w_mood == 0.0
        assert config.get_config().planner.w_mood == 0.0
    assert config.get_config().planner.w_mood == base


def test_config_extra_yaml(tmp_path, monkeypatch):
    extra = tmp_path / "extra.yaml"
    extra.write_text("planner:\n  beam_width: 7\n")
    monkeypatch.setenv("MONTAGE_CONFIG", str(extra))
    assert config.load_config().planner.beam_width == 7


def _samples() -> list:
    kb = T.KenBurns(zoom_from=1.0, zoom_to=1.08, pan_x=0.01, pan_y=-0.02)
    sc = T.ShotScores(importance=0.8, mood=0.1, director=0.5, unary_total=-1.2, pairwise_in=0.3)
    seg = T.SegmentDirective(anchor="peak", section_index=-1, content_queries=["friends dancing"], mood="energetic", pacing=0.5)
    plan = T.EditPlan(vibe="fun", vibe_queries=["beach"], avoid_queries=[], pacing=0, energy_bias=0, chronological=False,
                      target_length_s=0, excerpt_hint="auto", excerpt_start_s=0, opening_query="", closing_query="",
                      segments=[seg], must_include=[], exclude=[], rationale="r")
    ex = T.Excerpt(song_start=10.0, song_end=60.0, fade_out=1.5)
    shot = T.Shot(slot_idx=0, start=0, end=2, frame_start=0, frame_end=60, candidate_id="a001_w00", asset_id="a001",
                  kind="video", src_in=1.0, src_out=3.0, framing="crop", kenburns=kb, scores=sc, reason="x")
    return [
        T.Asset(id="a000", kind="photo", src_path="/x.jpg", hash="0" * 16, width=4, height=3, filename="x.jpg",
                taken_at=datetime(2026, 7, 14, 10, 0)),
        T.Candidate(id="a001_w00", asset_id="a001", kind="video", src_start=0.3, src_end=3.3, src_max_end=8.0,
                    scene_id="a001_s0", frame_times=[0.5, 1.0], thumb_path="/t.jpg"),
        T.FeatureTable(candidate_ids=["a000"], tags={"a000": ["beach"]}, feature_version="f1"),
        T.ImportanceScores(candidate_ids=["a000"], heuristic=[0.1], learned=None, final=[0.5], source="heuristic"),
        T.Section(idx=0, start=0, end=8, label="low", mean_intensity=-0.5),
        ex,
        T.MusicProfile(song_hash="h", song_duration=200, tempo=120, beat_mode="normal", excerpt=ex, beats=[0, 0.5],
                       downbeats=[0], grid_times=[0, 0.5], intensity=[0, 1], arousal=[0, 0], valence=[0, 0],
                       novelty=[0, 0], sections=[T.Section(idx=0, start=0, end=50, label="peak", mean_intensity=1)],
                       peak_time=20, emotion_source="heuristic", beat_source="librosa"),
        seg, plan,
        T.Slot(idx=0, start=0, end=2, n_beats=4, on_downbeat=True, intensity=0.1, arousal=0, valence=0,
               section_idx=0, boundary_strength=0, forced=False),
        kb, sc, shot,
        T.EDL(run_id="r", created=datetime.now(), system="full", seed=0, fps=30, width=1280, height=720,
              song_hash="h", excerpt=ex, plan=plan, shots=[shot], total_cost=-3.0),
        T.RunResult(run_id="r", edl_path="/e.json", video_path=None, preview=True, editor_note="n",
                    timings={"ingest": 1.0}, fallbacks=["director:rules"]),
    ]


def test_types_roundtrip():
    for obj in _samples():
        again = type(obj).model_validate_json(obj.model_dump_json())
        assert again == obj, type(obj).__name__


def test_atomic_write(tmp_path):
    target = tmp_path / "sub" / "x.json"
    paths.atomic_write(target, b"old")

    def bad_writer(tmp: Path):
        tmp.write_bytes(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        paths.atomic_write(target, bad_writer)
    assert target.read_bytes() == b"old"
    assert [p.name for p in target.parent.iterdir()] == ["x.json"]  # no temp left behind

    fresh = tmp_path / "new.bin"
    with pytest.raises(RuntimeError):
        paths.atomic_write(fresh, bad_writer)
    assert not fresh.exists()

    paths.atomic_write(target, "new")
    assert target.read_text() == "new"


def test_atomic_write_keeps_suffix(tmp_path):
    seen = {}
    paths.atomic_write(tmp_path / "a.npz", lambda tmp: (seen.setdefault("p", tmp), tmp.write_bytes(b"1")))
    assert seen["p"].suffix == ".npz"


def test_paths_and_run_id(fresh_root, monkeypatch):
    monkeypatch.setenv("MONTAGE_USER", "Keshav")
    assert paths.user() == "keshav"
    assert paths.jobs_dir() == fresh_root / "jobs" / "keshav"
    rid = paths.new_run_id()
    assert rid.startswith("keshav_") and len(rid.split("_")) == 3
    assert paths.DATA_ROOT == fresh_root


def test_parse_env_file(tmp_path):
    f = tmp_path / "env.sh"
    f.write_text("export MONTAGE_USER=keshav\nexport X='a b'\n# comment\n")
    assert paths.parse_env_file(f) == {"MONTAGE_USER": "keshav", "X": "a b"}
