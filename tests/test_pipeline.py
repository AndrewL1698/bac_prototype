"""Phase 7 acceptance: end-to-end generate / recut / export on fixtures with optional components off."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from conftest import gpu_available
from montage.types import UserError

pytestmark = [pytest.mark.gpu, pytest.mark.media]

OFF = {"director.use_llm": False, "music.emotion.use_mert": False, "music.use_beat_this": False}


def _frames(path) -> tuple[int, int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets,width,height", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    w, h, n = r.stdout.strip().split(",")
    return int(n), int(w), int(h)


@pytest.fixture(scope="module")
def run(fixtures, tmp_path_factory):
    if not gpu_available():
        pytest.skip("no GPU")
    import os
    from montage import config, pipeline
    root = tmp_path_factory.mktemp("pipe")
    old = os.environ["MONTAGE_DATA_ROOT"], os.environ["MONTAGE_SCRATCH"]
    os.environ["MONTAGE_DATA_ROOT"], os.environ["MONTAGE_SCRATCH"] = str(root / "data"), str(root / "scratch")
    media = root / "media"
    media.mkdir()
    for f in fixtures.iterdir():
        if f.suffix in (".mp4", ".mov", ".jpg"):
            shutil.copy(f, media / f.name)
    with config.override(OFF):
        res = pipeline.generate([str(media)], str(fixtures / "song_sections.wav"),
                                "Colorful and fun, skip anything blurry, end on the sunset", seed=0)
        yield pipeline, res, media
    os.environ["MONTAGE_DATA_ROOT"], os.environ["MONTAGE_SCRATCH"] = old


def test_generate_end_to_end(run):
    pipeline, res, _ = run
    edl = pipeline.load_edl(res.run_id)
    n, w, h = _frames(res.video_path)
    assert n == edl.shots[-1].frame_end and (w, h) == (1280, 720)
    for fb in ("director:rules", "emotion:heuristic", "beats:librosa", "importance:heuristic"):
        assert fb in res.fallbacks, res.fallbacks
    rd = pipeline.paths.run_dir(res.run_id)
    for f in ("plan.json", "edl.json", "result.json", "preview.mp4", "curves.png", "storyboard.jpg"):
        assert (rd / f).exists(), f
    assert res.editor_note and "BPM" in res.editor_note
    for k in ("ingest", "features", "music", "plan", "render"):
        assert k in res.timings
    print(res.timings)


def test_recut_with_note(run):
    pipeline, res, _ = run
    from montage import config
    s = pipeline.session_for_run(res.run_id)
    with config.override(OFF):
        r2 = pipeline.recut(s.session_id, note="faster")
    assert r2.run_id != res.run_id
    a, b = pipeline.load_edl(res.run_id), pipeline.load_edl(r2.run_id)
    assert len(b.shots) >= len(a.shots)
    assert b.plan.pacing > a.plan.pacing


def test_recut_lose_shot(run):
    pipeline, res, _ = run
    from montage import config
    s = pipeline.session_for_run(res.run_id)
    edl = pipeline.load_edl(s.runs[-1])
    target = next(sh for sh in edl.shots[1:] if sh.end - sh.start >= 1.5)
    t = int((target.start + target.end) / 2)  # a whole second inside the target shot
    with config.override(OFF):
        r = pipeline.recut(s.session_id, note=f"lose the shot at 0:{t:02d}")
    assert target.asset_id not in {sh.asset_id for sh in pipeline.load_edl(r.run_id).shots}


def test_export(run):
    pipeline, res, _ = run
    out = pipeline.export(res.run_id)
    n, w, h = _frames(out)
    assert (w, h) == (1920, 1080) and n == pipeline.load_edl(res.run_id).shots[-1].frame_end


def test_too_few_files(run, fixtures, tmp_path):
    pipeline, _, media = run
    few = tmp_path / "few"
    few.mkdir()
    for f in list(media.glob("*.jpg"))[:3]:
        shutil.copy(f, few / f.name)
    with pytest.raises(UserError, match="at least 10"):
        pipeline.generate([str(few)], str(fixtures / "song_sections.wav"), "")
