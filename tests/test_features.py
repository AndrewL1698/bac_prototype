"""Phase 3 acceptance: visual features, tags, mood, dedup (GPU tests run on Colab)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import gpu_available

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def feats(fixtures):
    if not gpu_available():
        pytest.skip("no GPU")
    from montage.candidates import build_media
    from montage.features.extract import extract
    ms = build_media([fixtures])
    return ms, extract(ms.assets, ms.candidates, ms.frames)


def _by_file(ms, f, name):
    aid = next(a.id for a in ms.assets if a.filename == name)
    return [i for i, c in enumerate(ms.candidates) if c.asset_id == aid]


def test_clip_text_match():
    if not gpu_available():
        pytest.skip("no GPU")
    from PIL import Image, ImageDraw
    from montage.features import clip_encoder
    red = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(red).rectangle([40, 40, 184, 184], fill="red")
    e = clip_encoder.encode_images([red])
    t = clip_encoder.encode_text(["a red square", "a blue circle"])
    s = (e @ t.T)[0]
    assert s[0] > s[1]


def test_feature_cache(feats):
    from montage.features import clip_encoder
    from montage.features.extract import extract
    ms, _ = feats
    before = clip_encoder.FORWARD_COUNT["image"]
    extract(ms.assets, ms.candidates, ms.frames)
    assert clip_encoder.FORWARD_COUNT["image"] == before


def test_shapes(feats):
    ms, f = feats
    n = len(ms.candidates)
    assert f.arrays["emb"].shape == (n, 768)
    assert np.allclose(np.linalg.norm(f.arrays["emb"], axis=1), 1, atol=1e-4)
    for k in ("aesthetic", "sharpness", "motion", "shake", "face_count", "visual_arousal", "dup_cluster"):
        assert f.arrays[k].shape == (n,), k
        assert not np.isnan(f.arrays[k]).any(), k


def test_quality_blur(feats):
    ms, f = feats
    photos = [i for i, c in enumerate(ms.candidates) if c.kind == "photo"]
    blurry = _by_file(ms, f, "photo_06_blurry.jpg")[0]
    sharp = f.arrays["sharpness"]
    assert sharp[blurry] == min(sharp[i] for i in photos)


def test_motion_order(feats):
    ms, f = feats
    m = f.arrays["motion"]
    hi = np.mean([m[i] for i in _by_file(ms, f, "vid_motion_landscape.mp4")])
    lo = np.mean([m[i] for i in _by_file(ms, f, "vid_static_landscape.mp4")])
    assert hi > lo


def test_dedup_neardup(feats):
    ms, f = feats
    d = f.arrays["dup_cluster"]
    a = _by_file(ms, f, "photo_07_dupA.jpg")[0]
    b = _by_file(ms, f, "photo_08_dupB.jpg")[0]
    assert d[a] == d[b]
    others = [_by_file(ms, f, n)[0] for n in ("photo_01_sunset.jpg", "photo_02_checker.jpg",
                                               "photo_03_circles_4x3.jpg", "photo_05_pano.jpg")]
    assert len({d[i] for i in others} | {d[a]}) == len(others) + 1
    wins = _by_file(ms, f, "vid_motion_landscape.mp4")
    assert all(d[wins[i]] != d[wins[i + 1]] for i in range(len(wins) - 1))


def test_mood_batch_z(feats):
    _, f = feats
    for k in ("visual_arousal", "visual_valence"):
        v = f.arrays[k]
        assert abs(v.mean()) < 1e-3 and abs(v.std() - 1) < 1e-3


def test_tags_shape(feats):
    from montage.features.tags import vocabulary
    _, f = feats
    vocab = set(vocabulary())
    assert set(f.table.tags) == set(f.ids)
    for tl in f.table.tags.values():
        assert len(tl) <= 5 and set(tl) <= vocab


def test_heuristic_prefers_sharp(feats):
    from montage.importance.blend import importance
    ms, f = feats
    imp = importance(f, ranker=False)
    photos = [i for i, c in enumerate(ms.candidates) if c.kind == "photo"]
    blurry = _by_file(ms, f, "photo_06_blurry.jpg")[0]
    order = sorted(photos, key=lambda i: imp.heuristic[i])
    assert blurry in order[:2]
