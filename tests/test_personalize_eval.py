"""Spec 15 section 6: the offline personalization check on synthetic albums."""
import json

import numpy as np
import pytest


def albums(seed=0, n_events=3, per_event=4, n=20):
    """Each event type has its own 'what people keep' direction in embedding space; base scores are noisy."""
    rng = np.random.default_rng(seed)
    emb, y, G, E = [], [], [], []
    for e in range(n_events):
        taste = rng.normal(size=768)
        for a in range(per_event):
            score = rng.normal(size=n)
            emb.append(rng.normal(size=(n, 768)) / np.sqrt(768) + np.outer(score, taste) / np.linalg.norm(taste))
            y += score.tolist()
            G += [f"e{e}a{a}"] * n
            E += [f"event{e}"] * n
    y = np.array(y)
    base = y + rng.normal(scale=2.0, size=len(y))
    return base, np.concatenate(emb).astype(np.float32), y, np.array(G), np.array(E)


def test_same_type_history_helps_and_control_does_not(fresh_root):
    from training.importance.evaluate_personalize import run
    m = run(*albums())
    assert m["n_albums"] == 12 and [r["beta_max"] for r in m["rows"]] == [0.1, 0.2, 0.3, 0.5]
    same = [r["same"]["gain"] for r in m["rows"]]
    assert same == sorted(same) and m["rows"][-1]["same"]["ci"][0] > 0  # more weight, more gain here
    assert np.mean(same) > np.mean([r["control"]["gain"] for r in m["rows"]])
    assert m["recommended_beta_max"] == 0.5  # the largest beta whose CI is above 0 and control is not hurt


def test_album_without_same_type_history_is_left_alone(fresh_root):
    from training.importance.evaluate_personalize import personalized
    base, emb, y, G, E = albums(n_events=2, per_event=2)
    E = np.where(G == "e0a0", "lonely", E)  # its only same-type album is now another type
    out = personalized(base, emb, y, G, E, beta_max=0.5, k=3, seed=0)
    rows = G == "e0a0"
    assert np.array_equal(out["same"][rows], base[rows]) and not np.array_equal(out["same"][G == "e1a0"],
                                                                                  base[G == "e1a0"])


def test_main_end_to_end(fresh_root):
    """After evaluate_cufed has run, the check reads its out-of-fold scores and writes the report."""
    pytest.importorskip("torch")
    from montage import config
    from test_cufed_eval import FAST, write_data
    from training.importance import evaluate_cufed as ev
    from training.importance import evaluate_personalize as evp
    write_data(fresh_root, seed=2)
    with config.override(FAST):
        assert ev.main([]) == 0
        assert evp.main([]) == 0
    m = json.loads((evp.out_dir() / "metrics.json").read_text())
    assert m["n_albums"] > 0 and len(m["rows"]) == 4 and "out of fold" in m["base"]
    assert "**Recommendation:**" in (evp.out_dir() / "report.md").read_text()
