"""Phase 15 days 8-10 (spec 15 section 6): personalization from pins and excludes."""
import json
from types import SimpleNamespace

import numpy as np

from montage import config


def feats(seed=0):
    """6 photos + 6 videos; in each modality the first 3 look like concept A, the last 3 like concept B."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=768), rng.normal(size=768)
    emb = np.stack([(a if i % 6 < 3 else b) + 0.3 * rng.normal(size=768) for i in range(12)]).astype(np.float32)
    ids = [f"c{i}" for i in range(12)]
    return SimpleNamespace(arrays={"emb": emb, "is_video": np.r_[np.zeros(6, bool), np.ones(6, bool)]}, ids=ids,
                           index=lambda: {c: i for i, c in enumerate(ids)})


def test_record_replace_put_back_and_cap(fresh_root):
    from montage.importance import personalize as P
    f = feats()
    P.record("s1", f, pins=["c0", "c1"], excluded=["c3"])
    s = P.load()["sessions"]["s1"]
    assert (s["pos_n"], s["neg_n"]) == (2, 1) and len(s["pos_sum"]) == 768
    P.record("s1", f, pins=["c0"], excluded=["c3", "c4", "missing"])  # re-record replaces; unknown ids ignored
    s = P.load()["sessions"]["s1"]
    assert (s["pos_n"], s["neg_n"]) == (1, 2)
    P.record("s1", f, pins=[], excluded=[])  # everything put back: the session is dropped
    assert "s1" not in P.load()["sessions"]
    with config.override({"importance.personalize.max_sessions": 2}):
        for sid in ("a", "b", "c"):
            P.record(sid, f, pins=["c0"], excluded=[])
    assert sorted(P.load()["sessions"]) == ["b", "c"]
    stored = json.loads(P.prefs_path().read_text())
    assert "c0" not in json.dumps(stored)  # only sums and counts are stored, no candidate ids


def test_no_evidence_leaves_scores_unchanged(fresh_root):
    from montage.importance import personalize as P
    f = feats()
    final = np.linspace(0, 1, 12)
    assert np.array_equal(P.apply(final, f)[0], final)  # no file
    P.record("s1", f, pins=["c0"], excluded=["c3"])     # 2 shots < min_events (3)
    out, n = P.apply(final, f)
    assert n == 2 and np.array_equal(out, final)
    P.record("s1", f, pins=["c0"], excluded=["c3", "c4"])
    with config.override({"importance.personalize.enabled": False}):
        assert np.array_equal(P.apply(final, f)[0], final)


def test_nudge_moves_lookalikes_within_modality(fresh_root):
    from montage.importance import personalize as P
    from montage.importance.blend import rank_normalize
    f = feats()
    groups = np.r_[np.zeros(6), np.ones(6)]
    final = rank_normalize(np.r_[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], groups)  # B-like shots start on top
    P.record("s1", f, pins=["c0"], excluded=["c3", "c6"])
    with config.override({"importance.personalize.beta_max": 0.9, "importance.personalize.half_evidence": 0.1}):
        out, n = P.apply(final, f)
    assert n == 3
    for lo in (0, 6):  # photos, then videos: each modality still spans [0, 1] and A-like shots now lead
        part = out[lo:lo + 6]
        assert part.min() == 0 and part.max() == 1
        assert part[:3].mean() > part[3:].mean()


def test_beta_follows_formula(fresh_root):
    from montage.importance import personalize as P
    with config.override({"importance.personalize.beta_max": 0.4, "importance.personalize.half_evidence": 10,
                          "importance.personalize.min_events": 3}):
        assert P.beta(2) == 0
        assert np.isclose(P.beta(10), 0.2) and np.isclose(P.beta(30), 0.3)
        assert P.beta(10_000) < 0.4


def test_cli_show_and_reset(fresh_root, capsys):
    from montage.importance import personalize as P
    P.record("s1", feats(), pins=["c0"], excluded=["c3", "c4"])
    assert P.main(["--show"]) == 0
    assert "1 kept and 2 removed" in capsys.readouterr().out
    assert P.main(["--reset"]) == 0
    assert not P.prefs_path().exists()
