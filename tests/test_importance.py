"""Importance scoring on synthetic feature arrays (runs anywhere)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from montage.importance.blend import importance, rank_normalize
from montage.importance.heuristic import hard_rejects, heuristic_scores


def synth(n_photo=6, n_video=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_photo + n_video
    arr = {
        "emb": rng.normal(size=(n, 768)).astype(np.float32),
        "aesthetic": rng.normal(5, 1, n), "sharpness": rng.normal(6, 1, n), "brightness": rng.uniform(0.3, 0.7, n),
        "contrast": rng.uniform(0.1, 0.3, n), "exposure_clip": rng.uniform(0, 0.05, n),
        "colorfulness": rng.uniform(0.2, 0.6, n), "face_count": rng.integers(0, 3, n).astype(float),
        "face_area": rng.uniform(0, 0.1, n),
        "motion": np.r_[np.zeros(n_photo), rng.uniform(0, 0.02, n_video)],
        "shake": np.r_[np.zeros(n_photo), rng.uniform(0, 0.01, n_video)],
        "is_video": np.r_[np.zeros(n_photo, bool), np.ones(n_video, bool)],
    }
    ids = [f"c{i}" for i in range(n)]
    return SimpleNamespace(arrays=arr, ids=ids, table=SimpleNamespace(tags={i: [] for i in ids}))


def test_junk_penalty():
    f = synth()
    i = int(np.argmax(heuristic_scores(f.arrays, f.table.tags, f.ids)))
    f.table.tags[f.ids[i]] = ["screenshot"]
    H = heuristic_scores(f.arrays, f.table.tags, f.ids)
    assert H[i] < np.median(H)


def test_rank_normalize_modality():
    x = np.array([3.0, 1.0, 2.0, 10.0, 30.0, 20.0, 5.0])
    g = np.array([0, 0, 0, 1, 1, 1, 2])
    r = rank_normalize(x, g)
    assert r[:3].min() == 0 and r[:3].max() == 1
    assert r[3:6].min() == 0 and r[3:6].max() == 1
    assert r[6] == 0.5
    assert list(r[:3]) == [1.0, 0.0, 0.5]


def test_blend_without_ranker(fresh_root):
    f = synth()
    imp = importance(f)  # no weights file in a fresh DATA_ROOT
    assert imp.source == "heuristic" and imp.learned is None
    groups = np.where(f.arrays["is_video"], 1, 0)
    assert np.allclose(imp.final, rank_normalize(np.array(imp.heuristic), groups))


def test_reject_relaxation():
    f = synth(10, 0)
    arr = f.arrays
    arr["sharpness"] = np.array([0.5, 0.6, 0.7, 6, 6, 6, 6, 6, 6, 6], float)
    arr["exposure_clip"][3] = 0.9
    H = heuristic_scores(arr, f.table.tags, f.ids)
    rej = hard_rejects(arr, H, n_needed=0)
    assert rej[0] and rej[3] and not rej[5]
    rej2 = hard_rejects(arr, H, n_needed=int((~rej).sum()) + 1)
    assert int((~rej2).sum()) == int((~rej).sum()) + 1
    back = np.where(rej & ~rej2)[0]
    assert len(back) == 1 and H[back[0]] == max(H[np.where(rej)[0]])
