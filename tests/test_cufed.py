"""Phase 15 (spec 15): CUFED download/layout checks, label mapping, Flickr fill, album features.

Everything here runs without a GPU, network, or the real dataset: a tiny synthetic CUFED tree is written to a
temp DATA_ROOT, the network is mocked, and the feature tests stub the pipeline. test_cufed_parity runs the real
pipeline on Colab (gpu + media).
"""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _jpeg(path: Path, seed: int, size=(48, 36)) -> Path:
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, "JPEG", quality=80)
    return path


def _jpeg_bytes(seed: int = 0) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(seed).random((20, 20, 3)) * 255).astype(np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def make_cufed(root: Path, albums: dict[str, tuple[list[float], set[int] | str]],
               events: dict[str, list[str]] | None = None, nest: str | None = None) -> Path:
    """Write a CUFED-shaped root: image_importance.json [[url, "album/photo", score]], event_type.json and
    images/<album>/<photo>.jpg for the photos whose index is in `present` ("all" = every photo).
    nest="Wedding" puts albums under images/<nest>/<album> (event-grouped archives)."""
    root.mkdir(parents=True, exist_ok=True)
    imp = {}
    for a_i, (album, (scores, present)) in enumerate(albums.items()):
        folder = root / "images" / (nest or "") / album
        entries = []
        for i, s in enumerate(scores):
            pid = str(100000 + 100 * a_i + i)
            entries.append([f"https://farm1.staticflickr.com/1/{pid}_abc.jpg", f"{album}/{pid}", s])
            if present == "all" or i in present:
                _jpeg(folder / f"{pid}.jpg", seed=100 * a_i + i)
        imp[album] = entries
    (root / "image_importance.json").write_text(json.dumps(imp))
    if events is not None:
        (root / "event_type.json").write_text(json.dumps(events))
    return root


def _cfg(**update):
    from montage.config import get_config
    c = get_config().importance.cufed
    flickr = update.pop("flickr", {})
    return c.model_copy(update={**update, "flickr": c.flickr.model_copy(update=flickr)})


SCORES12 = [-1.6, -0.8, 0.0, 0.4, 1.2, 2.0, -0.4, 0.8, 1.6, -2.0, 0.2, -1.2]


# ---------- layout ----------

def test_parse_importance_formats():
    from training.importance.cufed import parse_importance
    got = parse_importance({
        "0_1@N00": [["u1", "0_1@N00/111", 1.5], ["u2", "0_1@N00/222.jpg", "-0.5"], ["u3", "0_1@N00/111", 0.0]],
        "1_2@N00": [{"url": "u4", "key": "1_2@N00/333", "score": 2}],
    })
    a = got["0_1@N00"]
    assert [(p.photo, p.score, p.url) for p in a] == [("111", 1.5, "u1"), ("222", -0.5, "u2")]  # dup id dropped
    assert got["1_2@N00"][0].photo == "333" and got["1_2@N00"][0].score == 2.0
    for bad in ({}, {"a": "x"}, {"a": [["u", "a/1"]]}, {"a": [["u", "a/1", "high"]]}):
        with pytest.raises(ValueError):
            parse_importance(bad)


def test_find_root_nested_and_rejects_superres(fresh_root, caplog):
    from montage import config
    from training.importance.cufed import ensure_dataset, find_root, raw_dir
    sr = raw_dir() / "CUFED"  # super-resolution CUFED: images, no importance labels
    _jpeg(sr / "input" / "000_0.png", 1)
    with config.override({"importance.cufed.sources": ["manual"]}):
        root, info = ensure_dataset(_cfg(sources=["manual"]))
    assert root is None and info == {}
    assert "super-resolution" in caplog.text
    real = make_cufed(raw_dir() / "download" / "CUFED_release", {"0_1@N00": (SCORES12, "all")})
    assert find_root(raw_dir()) == real
    root, info = ensure_dataset(_cfg(sources=["manual"]))
    assert root == real and info["source"] == "manual" and info["albums_with_folder"] == 1
    assert info["photo_files"] == 12 and (raw_dir() / "_downloaded.json").exists()


def test_safe_extract_refuses_escapes(tmp_path):
    from training.importance.cufed import safe_extract
    arc = tmp_path / "evil.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("images/a/1.jpg", b"ok")
        zf.writestr("../escape.txt", b"no")
        zf.writestr("/abs.txt", b"no")
        zf.writestr("__MACOSX/images/._1.jpg", b"meta")
    dest = tmp_path / "out"
    assert safe_extract(arc, dest) == 1
    assert (dest / "images/a/1.jpg").read_bytes() == b"ok"
    assert not (tmp_path / "escape.txt").exists() and not Path("/abs.txt").exists()
    assert not (dest / "__MACOSX").exists()
    assert json.loads((dest / "_extracted_evil.zip.json").read_text())["refused"] == 2
    assert safe_extract(arc, dest) == 1  # marker: no second extraction

    tar = tmp_path / "links.tar"
    with tarfile.open(tar, "w") as tf:
        data = b"img"
        ti = tarfile.TarInfo("images/b/2.jpg")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
        link = tarfile.TarInfo("images/b/link.jpg")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(link)
    assert safe_extract(tar, tmp_path / "out2") == 1
    assert not (tmp_path / "out2/images/b/link.jpg").exists()


def test_extract_all_nested(tmp_path):
    from training.importance.cufed import extract_all, find_root
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("images/0_1@N00/1.jpg", b"x")
    outer = tmp_path / "hf" / "CUFED.zip"
    outer.parent.mkdir(parents=True)
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("CUFED/image_importance.json", json.dumps({"0_1@N00": [["u", "0_1@N00/1", 1.0]]}))
        zf.writestr("CUFED/images.zip", inner.getvalue())
    assert extract_all(tmp_path / "hf") == 3
    root = find_root(tmp_path / "hf")
    assert root == tmp_path / "hf" / "CUFED" and (root / "images/0_1@N00/1.jpg").exists()
    assert extract_all(tmp_path / "hf") == 0  # every archive already extracted


# ---------- label mapping ----------

def test_mapping_covers_every_album_once(fresh_root):
    from training.importance.cufed import build_mapping, index_album_dirs, read_events, read_labels
    root = make_cufed(fresh_root / "cufed", {
        "0_a@N00": (SCORES12, "all"),                                     # kept
        "1_b@N00": ([0.0, 1.0, 2.0, -1.0, 0.5], "all"),                   # 5 < min_photos
        "2_c@N00": (SCORES12, set()),                                     # labeled, no folder
        "3_d@N00": ([0.4] * 12, "all"),                                   # all tied
        "4_e@N00": ([3.5] + SCORES12[1:], set(range(1, 12))),             # 11 of 12 on disk; extreme one missing
    }, events={"0_a@N00": ["Wedding", "Show"], "1_b@N00": ["Zoo"]})
    labels = read_labels(root)
    m = build_mapping(root, labels, read_events(root), index_album_dirs(root, set(labels)), _cfg())
    assert set(m["albums"]) == {"0_a@N00", "4_e@N00"}
    assert {a: d["reason"] for a, d in m["dropped"].items()} == {
        "1_b@N00": "few_photos", "2_c@N00": "no_images", "3_d@N00": "tied_scores"}
    assert not set(m["albums"]) & set(m["dropped"])
    files = [p["path"] for a in m["albums"].values() for p in a["photos"]]
    assert len(files) == len(set(files)) == 23
    for a in m["albums"].values():
        y = np.array([p["y"] for p in a["photos"]])
        assert y.min() == 0.0 and y.max() == 1.0
    e = m["albums"]["4_e@N00"]
    assert e["n_missing"] == 1 and len(e["photos"]) == 11
    top = max(e["photos"], key=lambda p: p["score"])
    assert top["score"] == 2.0 and top["y"] == 1.0  # normalized over present photos only, not the missing 3.5
    assert m["albums"]["0_a@N00"]["events"] == ["Wedding", "Show"]
    assert m["albums"]["4_e@N00"]["events"] == ["unknown"]
    s = m["summary"]
    assert s["albums_kept"] == 2 and s["photos_kept"] == 23 and s["multi_label_albums"] == 1
    assert s["scores_out_of_range"] == 1  # the 3.5
    assert json.loads((fresh_root / "datasets/importance/cufed/mapping.json").read_text())["summary"] == s


def test_mapping_finds_event_grouped_folders(fresh_root):
    from training.importance.cufed import build_mapping, index_album_dirs, read_events, read_labels
    root = make_cufed(fresh_root / "cufed", {"0_a@N00": (SCORES12, "all")}, nest="Wedding")
    labels = read_labels(root)
    index = index_album_dirs(root, set(labels))
    assert index["0_a@N00"] == root / "images" / "Wedding" / "0_a@N00"
    m = build_mapping(root, labels, read_events(root), index, _cfg())
    assert len(m["albums"]["0_a@N00"]["photos"]) == 12


# ---------- Flickr fill ----------

class _Resp:
    def __init__(self, data: bytes, url: str):
        self.data, self.url = data, url

    def read(self):
        return self.data

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_flickr_fill(fresh_root, monkeypatch):
    import urllib.error
    import urllib.request
    from training.importance import cufed
    root = make_cufed(fresh_root / "cufed", {"0_a@N00": (SCORES12, set(range(8)))})  # 4 photos missing
    labels = cufed.read_labels(root)
    missing = [p for p in labels["0_a@N00"][8:]]
    png = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (10, 10)).save(png, "PNG")
    plan = {missing[0].url: ("ok", None), missing[1].url: ("dead", None),
            missing[2].url: ("placeholder", None), missing[3].url: ("png", None)}
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        kind = plan[url][0]
        if kind == "dead":
            raise urllib.error.HTTPError(url, 404, "gone", {}, None)
        if kind == "placeholder":
            return _Resp(_jpeg_bytes(), "https://s.yimg.com/pw/images/en-us/photo_unavailable.png")
        if kind == "png":
            return _Resp(png.getvalue(), url)
        return _Resp(_jpeg_bytes(1), url)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cfg = _cfg(flickr={"enabled": True, "delay_s": 0.0, "workers": 2})
    index = cufed.index_album_dirs(root, set(labels))
    stats = cufed.flickr_fill(root, labels, index, cfg)
    assert sorted(calls) == sorted(p.url for p in missing)  # only missing photos are requested
    assert stats == {"tried": 4, "fetched": 1, "failed": 3, "known_dead_total": 3}
    assert (index["0_a@N00"] / f"{missing[0].photo}.jpg").exists()
    for p in missing[1:]:
        assert not (index["0_a@N00"] / f"{p.photo}.jpg").exists()
    failed = json.loads((cufed.raw_dir() / "_flickr_failed.json").read_text())
    assert "placeholder" in failed[f"0_a@N00/{missing[2].photo}"] and "JPEG" in failed[f"0_a@N00/{missing[3].photo}"]

    calls.clear()
    assert cufed.flickr_fill(root, labels, index, cfg)["tried"] == 0 and not calls  # known dead links skipped
    plan[missing[1].url] = ("ok", None)
    stats = cufed.flickr_fill(root, labels, index, cfg, retry_failed=True)
    assert stats["tried"] == 3 and stats["fetched"] == 1 and stats["known_dead_total"] == 2


# ---------- entry point ----------

def test_cufed_main_end_to_end(fresh_root, capsys):
    from montage import config
    from training.importance import cufed
    with config.override({"importance.cufed.sources": ["manual"]}):
        assert cufed.main([]) == 2  # nothing yet: guidance printed
        assert "No CUFED dataset found" in capsys.readouterr().out
        make_cufed(cufed.raw_dir() / "copied" / "CUFED", {"0_a@N00": (SCORES12, "all"), "1_b@N00": (SCORES12, "all")},
                   events={"0_a@N00": ["Birthday"], "1_b@N00": ["Zoo"]})
        assert cufed.main(["--inspect"]) == 0
        assert not cufed.mapping_path().exists() and not (cufed.raw_dir() / "_downloaded.json").exists()
        assert cufed.main([]) == 0
    m = cufed.load_mapping()
    assert m["summary"]["albums_kept"] == 2 and m["summary"]["photos_kept"] == 24
    assert json.loads((cufed.raw_dir() / "_downloaded.json").read_text())["source"] == "manual"


def test_sources_official_and_hf(fresh_root, monkeypatch):
    """official: an HTML answer stops with instructions; an archive is extracted and found. hf: the snapshot is
    pinned to the resolved sha, archives inside are extracted. Network and huggingface_hub are faked."""
    import sys
    import types
    from training.importance import cufed

    def fake_download(url, dest, *a, **k):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "html" in url:
            dest.write_bytes(b"<!DOCTYPE html><html>Google Drive can't scan this file</html>")
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("CUFED/image_importance.json", json.dumps({"0_a@N00": [["u", "0_a@N00/1", 1.0]]}))
                zf.writestr("CUFED/images/0_a@N00/1.jpg", _jpeg_bytes())
            dest.write_bytes(buf.getvalue())
        return dest

    monkeypatch.setattr(cufed, "download", fake_download)
    with pytest.raises(SystemExit, match="web page"):
        cufed.fetch_official("https://example.org/html")
    root, info = cufed.fetch_official("https://example.org/CUFED.zip")
    assert root == cufed.raw_dir() / "official" / "CUFED" and info == {"url": "https://example.org/CUFED.zip"}
    assert cufed.fetch_official(None) == (None, {})

    seen = {}

    class FakeApi:
        def dataset_info(self, repo, revision=None):
            seen["info"] = (repo, revision)
            return types.SimpleNamespace(sha="abc123def")

    def fake_snapshot(repo_id, repo_type, revision, local_dir):
        seen["snapshot"] = (repo_id, repo_type, revision)
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "image_importance.json").write_text(json.dumps({"0_b@N00": [["u", "0_b@N00/7", 0.5]]}))
        with zipfile.ZipFile(d / "images.zip", "w") as zf:
            zf.writestr("images/0_b@N00/7.jpg", _jpeg_bytes(2))

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot))
    root, info = cufed.fetch_hf("someone/CUFED-mirror", "main")
    assert seen == {"info": ("someone/CUFED-mirror", "main"),
                    "snapshot": ("someone/CUFED-mirror", "dataset", "abc123def")}
    assert root == cufed.raw_dir() / "hf" and (root / "images/0_b@N00/7.jpg").exists()
    assert info == {"repo": "someone/CUFED-mirror", "revision": "main", "sha": "abc123def"}

    class Gated(FakeApi):
        def dataset_info(self, repo, revision=None):
            raise PermissionError("401: this dataset is gated")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=Gated, snapshot_download=fake_snapshot))
    assert cufed.fetch_hf("someone/gated", "main") == (None, {})  # logged, next source is tried
