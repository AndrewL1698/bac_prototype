"""Phase 9A acceptance with the live LLM director (Colab, needs ANTHROPIC_API_KEY)."""

from __future__ import annotations

import os
import shutil

import pytest

from conftest import gpu_available

pytestmark = [pytest.mark.gpu, pytest.mark.media, pytest.mark.api,
              pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")]


@pytest.fixture(scope="module")
def session(fixtures, tmp_path_factory):
    if not gpu_available():
        pytest.skip("no GPU")
    from test_timing import make_album
    from montage import pipeline
    root = tmp_path_factory.mktemp("p9a")
    old = os.environ["MONTAGE_DATA_ROOT"], os.environ["MONTAGE_SCRATCH"]
    os.environ["MONTAGE_DATA_ROOT"], os.environ["MONTAGE_SCRATCH"] = str(root / "data"), str(root / "scratch")
    album = make_album(root / "album", n_photos=40, n_videos=0)
    for f in fixtures.iterdir():
        if f.suffix in (".mp4", ".mov", ".jpg"):
            shutil.copy(f, album / f.name)
    res = pipeline.generate([str(album)], str(fixtures / "song_sections.wav"),
                            "Bright and playful, colorful circles, slower at the start, end on the sunset", seed=0)
    yield pipeline, res
    os.environ["MONTAGE_DATA_ROOT"], os.environ["MONTAGE_SCRATCH"] = old


def test_llm_plan(session):
    pipeline, res = session
    edl = pipeline.load_edl(res.run_id)
    assert edl.plan.source == "llm", res.fallbacks
    assert "director:rules" not in res.fallbacks
    assert res.editor_note and "Cut " not in res.editor_note[:4]  # LLM note, not the template
    print("plan:", edl.plan.model_dump_json(indent=1))
    print("note:", res.editor_note)


def test_note_faster_adds_shots(session):
    pipeline, res = session
    s = pipeline.session_for_run(res.run_id)
    before = pipeline.load_edl(s.runs[-1])
    r = pipeline.recut(s.session_id, note="faster")
    after = pipeline.load_edl(r.run_id)
    print(len(before.shots), "->", len(after.shots), before.plan.pacing, "->", after.plan.pacing)
    assert len(after.shots) > len(before.shots)


def test_note_lose_shot(session):
    pipeline, res = session
    s = pipeline.session_for_run(res.run_id)
    before = pipeline.load_edl(s.runs[-1])
    target = next(sh for sh in before.shots if sh.start <= 5.0 < sh.end)
    r = pipeline.recut(s.session_id, note="lose the shot at 0:05")
    after = pipeline.load_edl(r.run_id)
    assert target.asset_id in after.plan.exclude
    assert target.asset_id not in {sh.asset_id for sh in after.shots}


def test_pin_forces_inclusion(session):
    pipeline, res = session
    s = pipeline.session_for_run(res.run_id)
    st = pipeline.state(s)
    used = {sh.candidate_id for sh in pipeline.load_edl(s.runs[-1]).shots}
    excluded_assets = set(s.plan_history[-1].exclude)
    pick = next(c.id for c in st.media.candidates if c.id not in used and c.kind == "photo"
                and c.asset_id not in excluded_assets)
    r = pipeline.recut(s.session_id, pins=[pick])
    assert pick in {sh.candidate_id for sh in pipeline.load_edl(r.run_id).shots}
