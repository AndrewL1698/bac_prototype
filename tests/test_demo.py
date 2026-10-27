"""Phase 11 (demo parts): live vote analysis and demo-mode preparation."""

from __future__ import annotations

import os
import shutil

import pytest
from scipy.stats import binomtest

from montage.eval.vote import analyze, load_votes, record_vote, wilson


def _rows(wins, losses, ties):
    rows = []
    for i in range(wins):  # full wins from either side
        rows.append({"left_system": "full", "right_system": "naive", "choice": "left"} if i % 2 else
                    {"left_system": "naive", "right_system": "full", "choice": "right"})
    for _ in range(losses):
        rows.append({"left_system": "naive", "right_system": "full", "choice": "left"})
    for _ in range(ties):
        rows.append({"left_system": "full", "right_system": "naive", "choice": "tie"})
    return rows


def test_vote_analysis():
    res = analyze(_rows(15, 5, 2))
    assert (res["wins"], res["losses"], res["ties"]) == (15, 5, 2)
    assert res["win_rate"] == pytest.approx(0.75)
    assert res["p_value"] == pytest.approx(binomtest(15, 20, 0.5).pvalue)
    lo, hi = res["ci95"]
    assert lo < 0.75 < hi and (lo, hi) == pytest.approx(wilson(15, 20))


def test_record_vote_roundtrip(fresh_root, monkeypatch):
    monkeypatch.setenv("MONTAGE_USER", "a")
    record_vote("Ann", "album1", "full", "naive", "left")
    monkeypatch.setenv("MONTAGE_USER", "b")
    record_vote("Bo", "album1", "naive", "full", "left")
    record_vote("", "album1", "full", "naive", "tie")
    rows = load_votes()
    assert len(rows) == 3
    res = analyze(rows)
    assert (res["wins"], res["losses"], res["ties"]) == (1, 1, 1)
    with pytest.raises(ValueError):
        record_vote("x", "p", "full", "naive", "maybe")


def test_demo_albums_nested(fresh_root):
    from montage.app.demo_mode import demo_albums
    nested = fresh_root / "demo" / "Summer Trip" / "export"
    nested.mkdir(parents=True)
    (nested / "IMG_0001.HEIC").write_bytes(b"x")
    for name in ("songs", "songs_full", "backups", "vote"):
        (fresh_root / "demo" / name).mkdir(parents=True, exist_ok=True)
        (fresh_root / "demo" / name / "x.mp4").write_bytes(b"x")
    assert demo_albums() == ["Summer Trip"]


@pytest.mark.gpu
@pytest.mark.media
def test_demo_prepare(fixtures, tmp_path, monkeypatch):
    from conftest import gpu_available
    if not gpu_available():
        pytest.skip("no GPU")
    data = tmp_path / "data"
    album = data / "demo" / "fixtures_album"
    album.mkdir(parents=True)
    (data / "demo" / "songs").mkdir(parents=True)
    for f in fixtures.iterdir():
        if f.suffix in (".mp4", ".mov", ".jpg"):
            shutil.copy(f, album / f.name)
    (album / "brief.txt").write_text("Colorful and fun, end on the sunset")
    shutil.copy(fixtures / "song_sections.wav", data / "demo" / "songs" / "song_sections.wav")
    monkeypatch.setenv("MONTAGE_DATA_ROOT", str(data))
    monkeypatch.setenv("MONTAGE_SCRATCH", str(tmp_path / "scratch"))
    from montage.app import demo_mode
    report = demo_mode.prepare()
    print(report)
    print("\n".join(demo_mode.checklist(report)))
    backups = data / "demo" / "backups"
    assert (backups / "fixtures_album.mp4").exists() and (backups / "fixtures_album_naive.mp4").exists()
    pairs = demo_mode.load_pairs()
    assert pairs and all((data / "demo" / "vote" / p[k]).exists() for p in pairs for k in ("a", "b"))
    assert report["warm_generate_s"] < 30
    assert os.path.getsize(backups / "fixtures_album.mp4") > 100_000
