"""Phase 8 acceptance: the Gradio app driven through gradio_client (Colab)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pytest

from conftest import gpu_available

pytestmark = [pytest.mark.gpu, pytest.mark.media]
PORT = 7871


@pytest.fixture(scope="module")
def app(fixtures, tmp_path_factory):
    if not gpu_available():
        pytest.skip("no GPU")
    root = tmp_path_factory.mktemp("app")
    data = root / "data"
    album = data / "demo" / "fixtures_album"
    album.mkdir(parents=True)
    (data / "demo" / "songs").mkdir(parents=True)
    for f in fixtures.iterdir():
        if f.suffix in (".mp4", ".mov", ".jpg"):
            shutil.copy(f, album / f.name)
    shutil.copy(fixtures / "song_sections.wav", data / "demo" / "songs" / "song_sections.wav")
    env = {**os.environ, "MONTAGE_DATA_ROOT": str(data), "MONTAGE_SCRATCH": str(root / "scratch"),
           "MONTAGE_USER": "apptest", "MONTAGE_CONFIG": str(root / "cfg.yaml")}
    (root / "cfg.yaml").write_text("director:\n  use_llm: false\n")
    log = open(root / "app.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "montage.app.gradio_app", "--port", str(PORT)], env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    url_file = data / "jobs" / "apptest" / "app.url"
    t0 = time.time()
    while not url_file.exists() and time.time() - t0 < 120 and proc.poll() is None:
        time.sleep(1)
    if not url_file.exists():
        proc.kill()
        pytest.fail("app did not start: " + (root / "app.log").read_text()[-2000:])
    print(f"app.url after {time.time() - t0:.1f}s")
    from gradio_client import Client
    yield Client(f"http://127.0.0.1:{PORT}", verbose=False), fixtures
    proc.terminate()
    proc.wait(timeout=20)


def test_generate_and_recut(app):
    client, _ = app
    out = client.predict(None, "fixtures_album", None, "song_sections.wav", "Colorful and fun, end on the sunset",
                         0, 0, "", False, "full", api_name="/generate")
    video, status, note, plan_md = out[0], out[1], out[2], out[3]
    assert video and "shots" in status and "Director: rules" in status
    assert "Vibe" in plan_md and note
    out2 = client.predict("faster", api_name="/recut")
    assert out2[0]
    import json
    assert json.loads(out2[4])["pacing"] > json.loads(out[4])["pacing"]  # plan JSON
    assert "faster" in out2[9]  # notes history


def test_too_few_files(app):
    client, fixtures = app
    from gradio_client import handle_file
    files = [handle_file(str(p)) for p in sorted(fixtures.glob("photo_0*.jpg"))[:3]]
    with pytest.raises(Exception, match="at least 10"):
        client.predict(files, "", None, "song_sections.wav", "", 0, 0, "", False, "full", api_name="/generate")
