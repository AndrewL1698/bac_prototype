"""Phase 15 days 4-7 (spec 15 section 5): CV with CUFED, photo ranker selection, deploy decision, safe deploy.

Synthetic windows.npz + cufed_photos.npz where importance is learnable from one feature column and the heuristic
is noise, written to a temp DATA_ROOT; the real trainer runs with few epochs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

FAST = {"importance.cufed.eval.epochs": 5, "importance.cufed.eval.bootstrap": 300,
        "importance.cufed.eval.kinds": ["linear"]}


def _rows(rng, n_groups, per, dataset, signal):
    n = n_groups * per
    X = rng.normal(size=(n, 776)).astype(np.float32)
    X[:, :768] /= np.linalg.norm(X[:, :768], axis=1, keepdims=True)  # CLIP embeddings are unit vectors
    raw = signal * X[:, 768] + rng.normal(size=n)
    G = np.repeat([f"{dataset}/g{i:02d}" for i in range(n_groups)], per)
    y = np.zeros(n, np.float32)
    for g in np.unique(G):
        m = G == g
        y[m] = (raw[m] - raw[m].min()) / (raw[m].max() - raw[m].min())
    return {"X": X, "y": y, "heuristic": rng.normal(size=n).astype(np.float32),
            "aesthetic": rng.normal(size=n).astype(np.float32), "group": G, "dataset": np.full(n, dataset)}


def write_data(root, seed=0, signal=2.0):
    from montage import cache
    rng = np.random.default_rng(seed)
    vids = [_rows(rng, 8, 20, "summe", signal), _rows(rng, 8, 20, "tvsum", signal)]
    w = {k: np.concatenate([v[k] for v in vids]) for k in vids[0]}
    w["t_mid"] = np.tile(np.arange(20, dtype=float) * 1.5, 16)
    alb = _rows(rng, 20, 15, "cufed", signal)
    alb["group"] = np.char.replace(alb["group"].astype(str), "cufed/g", "cufed/album")
    alb["event"] = np.repeat(["Wedding", "Birthday"] * 10, 15)
    alb["photo"] = np.array([f"p{i}" for i in range(len(alb["y"]))])
    d = root / "datasets" / "importance"
    cache.save_npz(d / "windows.npz", **w)
    cache.save_npz(d / "cufed_photos.npz", **alb)


def test_folds_match_phase_10a():
    from training.importance.evaluate import folds
    from training.importance.evaluate_cufed import make_folds
    G = np.repeat([f"summe/v{i}" for i in range(7)] + [f"tvsum/v{i}" for i in range(11)], 3)
    D = np.array([g.split("/")[0] for g in G])
    old = folds(G, D, seed=0)
    assert make_folds(G, D, 5, 0).tolist() == [old[g] for g in G]  # video folds identical to phase 10A
    G2 = np.concatenate([G, np.repeat([f"cufed/a{i}" for i in range(9)], 4)])
    D2 = np.array([g.split("/")[0] for g in G2])
    F2 = make_folds(G2, D2, 5, 0)
    assert F2[: len(G)].tolist() == [old[g] for g in G]  # adding albums does not reshuffle the videos
    assert sorted(set(F2[len(G):])) == [0, 1, 2, 3, 4]


def test_choose_best_on_cufed():
    """The chosen model only scores photos, so it is chosen on CUFED alone, whatever it does on the videos."""
    from training.importance.evaluate_cufed import choose

    def row(summe, cufed):
        return {"summe": {"tau": summe}, "tvsum": {"tau": 0.4}, "cufed": {"tau": cufed}}

    cands = {("learned_cufed_mlp", 1.0): row(0.05, 0.35),   # best on CUFED, worse on SumMe: still chosen
             ("learned_tvsum_mlp", 0.75): row(0.17, 0.20)}
    assert choose(cands) == ("learned_cufed_mlp", 1.0)
    cands[("learned_all_mlp", 0.5)] = row(0.17, 0.35)      # tie on CUFED: the smaller alpha wins
    assert choose(cands) == ("learned_all_mlp", 0.5)


def test_bootstrap_gain():
    from training.importance.evaluate_cufed import bootstrap_gain
    old = {f"a{i}": (0.1, 0.0) for i in range(40)}
    new = {f"a{i}": (0.1 + (0.2 if i % 4 else -0.05), 0.0) for i in range(40)}
    g = bootstrap_gain(new, old, sorted(old), 500, seed=0)
    assert g["n_groups"] == 40 and abs(g["gain"] - (30 * 0.2 - 10 * 0.05) / 40) < 1e-9
    assert 0 < g["ci"][0] < g["gain"] < g["ci"][1] and g["share_improved"] == 0.75
    flat = bootstrap_gain(old, old, sorted(old), 500, seed=0)
    assert flat["gain"] == 0 and flat["ci"] == [0.0, 0.0]


def test_eval_deploy_and_rerun(fresh_root, monkeypatch, capsys):
    pytest.importorskip("torch")
    from montage import config, paths
    from montage.importance.ranker import Ranker
    from training.importance import evaluate_cufed as ev
    write_data(fresh_root)
    with config.override(FAST):
        assert ev.main([]) == 0
        m = json.loads((ev.eval_dir() / "metrics.json").read_text())
        assert m["baseline"] == "heuristic" and m["groups"] == {"summe": 8, "tvsum": 8, "cufed": 20}
        t = m["table"]
        assert {"random", "aesthetic", "heuristic", "learned_cufed_linear", "learned_all_linear"} <= set(t)
        assert "deployed_as_is" not in t  # nothing deployed yet
        assert t["learned_cufed_linear"]["cufed"]["tau"] > t["heuristic"]["cufed"]["tau"] + 0.2  # it learns
        assert m["decision"]["deploy"] and m["gain_vs_baseline"]["ci"][0] > 0
        assert set(m["per_event"]) == {"Wedding", "Birthday"}
        for f in ("report.md", "candidate.pt", "per_album_tau.png"):
            assert (ev.eval_dir() / f).exists()
        live = paths.weights_dir() / "importance_ranker_photo.pt"
        assert not live.exists()  # an evaluation never touches the live weights

        assert ev.main(["--deploy"]) == 0
        r = Ranker.load(live)
        assert r is not None and r.blend_alpha == m["chosen"]["alpha"] and r.kind == "linear"
        assert not live.with_suffix(".prev.pt").exists()
        assert not (paths.weights_dir() / "importance_ranker.pt").exists()  # the video ranker is never written

        calls = []
        real_fit = ev.fit
        monkeypatch.setattr(ev, "fit", lambda *a, **k: calls.append(1) or real_fit(*a, **k))
        assert ev.main([]) == 0
        assert len(calls) == 1  # CV reused from the cache; only the final retrain ran
        m2 = json.loads((ev.eval_dir() / "metrics.json").read_text())
        assert "deployed_as_is" in m2["table"] and "deployed_config_cv" in m2["table"]
        assert m2["deployed"]["file"] == "importance_ranker_photo.pt"
        saw_cufed = m["chosen"]["trained_on"] in ("cufed", "all")  # then its as-is CUFED score is in sample
        assert m2["baseline"] == ("deployed_config_cv" if saw_cufed else "deployed_as_is")
        assert m2["deployed"]["trained_on"] == m["chosen"]["trained_on"]
        if not m2["decision"]["deploy"]:  # same model again: no clear gain, so it must not go live
            before = live.read_bytes()
            assert ev.main(["--deploy"]) == 1 and live.read_bytes() == before
            assert "not deploying" in capsys.readouterr().out


def test_deploy_refuses_mismatched_candidate(fresh_root):
    pytest.importorskip("torch")
    from montage import config, paths
    from training.importance import evaluate_cufed as ev
    assert ev.main(["--deploy"]) == 2  # nothing evaluated yet
    write_data(fresh_root, seed=1)
    with config.override(FAST):
        assert ev.main([]) == 0
    cand = ev.eval_dir() / "candidate.pt"
    cand.write_bytes(cand.read_bytes() + b"x")
    assert ev.main(["--deploy"]) == 2
    assert not (paths.weights_dir() / "importance_ranker_photo.pt").exists()


def test_video_ranker_is_todays_photo_baseline(fresh_root):
    """Today's situation: only the video ranker (trained on TVSum) is live, so it also scores photos. It never saw
    CUFED, so the candidate is compared with it as is; deploying writes the photo ranker and leaves it alone."""
    pytest.importorskip("torch")
    from montage import config, paths
    from training.importance import evaluate_cufed as ev
    write_data(fresh_root, seed=2)
    with config.override({**FAST, "importance.cufed.eval.sources": ["tvsum"]}):
        assert ev.main([]) == 0
    video = paths.weights_dir() / "importance_ranker.pt"
    paths.atomic_write(video, (ev.eval_dir() / "candidate.pt").read_bytes())  # stands in for the phase 10A ranker
    before = video.read_bytes()
    with config.override(FAST):
        assert ev.main([]) == 0
    m = json.loads((ev.eval_dir() / "metrics.json").read_text())
    assert m["deployed"]["file"] == "importance_ranker.pt" and m["deployed"]["trained_on"] == "tvsum"
    assert m["baseline"] == "deployed_as_is"
    assert "deployed_config_cv" in m["table"] and m["table"]["deployed_config_cv"]["summe"]["n_groups"] == 8
    assert m["chosen"]["trained_on"] in ("cufed", "all")  # the signal in the albums is learnable
    report = (ev.eval_dir() / "report.md").read_text()
    assert "deployed_as_is" in report and "**Decision:**" in report and "| Wedding |" in report
    assert "videos keep their ranker" in report
    if m["decision"]["deploy"]:
        assert ev.main(["--deploy"]) == 0
        assert (paths.weights_dir() / "importance_ranker_photo.pt").exists()
    assert video.read_bytes() == before  # the video ranker is never touched
