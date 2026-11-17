"""Track B: ranker training utilities, deployed-ranker blend, DEAM splits, feature parity."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import gpu_available


def _synthetic(n_groups=24, per=60, seed=0):
    """Feature scales like the real ranker_features: L2-normalized 768-d embedding + 8 scalar columns."""
    rng = np.random.default_rng(seed)
    n = n_groups * per
    emb = rng.normal(size=(n, 768))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    scal = rng.normal(size=(n, 8))
    X = np.concatenate([emb, scal], 1).astype(np.float32)
    u = rng.normal(size=768)
    u /= np.linalg.norm(u)
    y = 1.5 * scal[:, 0] + 1.0 * scal[:, 2] + 25.0 * emb @ u + rng.normal(scale=1.0, size=n)
    G = np.repeat([f"g{i}" for i in range(n_groups)], per)
    for g in np.unique(G):  # labels only matter within a group: min-max per group
        m = G == g
        y[m] = (y[m] - y[m].min()) / (y[m].max() - y[m].min())
    return X, y.astype(np.float32), G


def test_ranknet_learns_within_group_order():
    pytest.importorskip("torch")
    from training.importance.train import fit, mean_tau
    X, y, G = _synthetic()
    tr = np.isin(G, [f"g{i}" for i in range(18)])
    r = fit(X[tr], y[tr], G[tr], "linear", seed=0, val=(X[~tr], y[~tr], G[~tr]), epochs=15)
    assert mean_tau(r.score(X[~tr]), y[~tr], G[~tr]) > 0.5


def test_blend_with_ranker(fresh_root):
    torch = pytest.importorskip("torch")
    from montage import paths
    from montage.config import get_config
    from montage.importance.blend import importance
    from montage.importance.ranker import Ranker
    from test_importance import synth
    f = synth()
    model = Ranker.build("linear", 776)
    ck = {"type": "linear", "state_dict": model.state_dict(), "feat_mean": [0.0] * 776, "feat_std": [1.0] * 776,
          "blend_alpha": 0.25}
    buf = io.BytesIO()
    torch.save(ck, buf)
    paths.atomic_write(paths.weights_dir() / get_config().importance.ranker_weights, buf.getvalue())
    imp = importance(f)
    assert imp.source == "blend" and imp.learned is not None
    assert all(0 <= v <= 1 for v in imp.final)


def _save_ranker(path, alpha, seed):
    import torch
    from montage import paths
    from montage.importance.ranker import Ranker
    torch.manual_seed(seed)
    model = Ranker.build("linear", 776)
    ck = {"type": "linear", "state_dict": model.state_dict(), "feat_mean": [0.0] * 776, "feat_std": [1.0] * 776,
          "blend_alpha": alpha}
    buf = io.BytesIO()
    torch.save(ck, buf)
    paths.atomic_write(path, buf.getvalue())


def test_photo_ranker_scores_photos_only(fresh_root):
    """Phase 15: photos use the photo ranker when it exists; videos keep the video ranker, unchanged."""
    pytest.importorskip("torch")
    from montage import paths
    from montage.config import get_config
    from montage.importance.blend import importance, rank_normalize
    from montage.importance.ranker import Ranker, ranker_features
    from test_importance import synth
    f = synth()
    w, cfg = paths.weights_dir(), get_config().importance
    v = f.arrays["is_video"]
    groups = np.where(v, 1, 0)
    X = ranker_features(f.arrays)

    _save_ranker(w / cfg.ranker_weights, alpha=0.25, seed=0)
    before = importance(f)  # no photo ranker: the video ranker scores both, as before phase 15
    video = Ranker.load()
    H = np.array(before.heuristic)
    expect = 0.25 * rank_normalize(video.score(X), groups) + 0.75 * rank_normalize(H, groups)
    assert np.allclose(before.final, expect)

    _save_ranker(w / cfg.photo_ranker_weights, alpha=1.0, seed=1)
    after = importance(f)
    photo = Ranker.load(w / cfg.photo_ranker_weights)
    assert np.allclose(np.array(after.final)[v], np.array(before.final)[v])  # videos unchanged
    assert np.allclose(np.array(after.learned)[~v], photo.score(X[~v]), atol=1e-5)
    assert np.allclose(np.array(after.final)[~v], rank_normalize(photo.score(X[~v]), groups[~v]))
    assert importance(f, photo_ranker=False).final == before.final
    assert importance(f, ranker=False).source == "heuristic"

    (w / cfg.ranker_weights).unlink()  # photo ranker only: videos fall back to the heuristic
    only = importance(f)
    assert only.source == "blend"
    assert np.allclose(np.array(only.final)[v], rank_normalize(H, groups)[v])


def test_deam_splits_disjoint():
    from training.emotion.dataset import ccc, split_ids
    ids = list(range(2, 1001)) + list(range(2001, 2059))
    sp = split_ids(ids, seed=0)
    sets = {k: set(v) for k, v in sp.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert sets["full"] == set(range(2001, 2059)) and not (sets["full"] & sets["train"])
    assert abs(len(sp["test"]) - 100) <= 1 and abs(len(sp["val"]) - 100) <= 1
    x = np.linspace(-1, 1, 50)
    assert ccc(x, x) == pytest.approx(1.0) and ccc(x, -x) == pytest.approx(-1.0)


@pytest.mark.gpu
@pytest.mark.media
def test_parity(fixtures):
    """Training features for a dataset video equal the app's features for the same file (same code, same cache)."""
    if not gpu_available():
        pytest.skip("no GPU")
    from montage import config
    from montage.candidates import build_media
    from montage.features.extract import extract
    from montage.importance.ranker import ranker_features
    from training.importance.extract_clip import LIFT, video_features
    video = str(fixtures / "vid_motion_landscape.mp4")
    cands, f = video_features(video)
    with config.override(LIFT):
        ms = build_media([video])
        g = extract(ms.assets, ms.candidates, ms.frames)
    assert [c.id for c in cands] == [c.id for c in ms.candidates]
    assert np.allclose(ranker_features(f.arrays), ranker_features(g.arrays))
