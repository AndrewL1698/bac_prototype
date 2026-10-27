"""Planner on real fixtures: media -> features -> music -> rules plan -> EDL (Colab)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import gpu_available

pytestmark = pytest.mark.gpu


def test_plan_on_fixtures(fixtures):
    if not gpu_available():
        pytest.skip("no GPU")
    from montage.candidates import build_media
    from montage.director import make_plan
    from montage.director.apply import compile_plan
    from montage.features.extract import extract
    from montage.importance.blend import importance
    from montage.music.excerpt import choose_excerpt
    from montage.music.profile import analyze_song, build_profile
    from montage.planning import PlanInputs, plan_edit

    ms = build_media([fixtures])
    f = extract(ms.assets, ms.candidates, ms.frames)
    imp = importance(f)
    fa = analyze_song(fixtures / "song_sections.wav")
    plan, _ = make_plan("Colorful patterns, skip anything blurry, end on the sunset")
    prof = build_profile(fa, choose_excerpt(fa, plan.target_length_s, plan.excerpt_hint, plan.excerpt_start_s))
    d = compile_plan(plan, prof, f, [c.asset_id for c in ms.candidates])
    inp = PlanInputs(ms.assets, ms.candidates, f, np.array(imp.final), np.array(imp.heuristic), prof, d)
    edl = plan_edit(inp, "test_run", seed=0)
    L = prof.excerpt.song_end - prof.excerpt.song_start
    assert edl.shots[-1].frame_end == round(L * 30)
    beats = np.array(prof.beats + [L])
    assert all(np.min(np.abs(beats - s.start)) < 1e-3 for s in edl.shots)
    assert edl.shots[-1].asset_id is not None and all(s.reason for s in edl.shots)
    blurry = next(a.id for a in ms.assets if a.filename == "photo_06_blurry.jpg")
    print("\n".join(f"{s.start:5.1f}-{s.end:5.1f} {s.candidate_id:9s} {s.framing:9s} {s.reason}" for s in edl.shots))
    print("notes:", edl.notes)
    assert edl.notes is not None and blurry  # blurry may still be needed when footage is scarce
