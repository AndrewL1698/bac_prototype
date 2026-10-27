"""Phase 11 evaluation metrics on hand-built EDLs (runs anywhere)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from montage.eval import metrics
from montage.types import EDL, EditPlan, Excerpt, MusicProfile, Section, Shot, ShotScores

SC = ShotScores(importance=0, mood=0, director=0, unary_total=0, pairwise_in=0)


def profile(L=24.0) -> MusicProfile:
    beats = list(np.arange(0, L, 0.5))
    grid = list(np.arange(0, L, 0.5))
    inten = list(np.linspace(-1, 1, len(grid)))
    return MusicProfile(song_hash="h", song_duration=100, tempo=120, beat_mode="normal",
                        excerpt=Excerpt(song_start=0, song_end=L, fade_out=1.5), beats=beats, downbeats=beats[::4],
                        grid_times=grid, intensity=inten, arousal=inten, valence=[0.0] * len(grid),
                        novelty=[0.0] * len(grid),
                        sections=[Section(idx=0, start=0, end=12, label="low", mean_intensity=-0.5),
                                  Section(idx=1, start=12, end=L, label="peak", mean_intensity=0.5)],
                        peak_time=L - 1, emotion_source="heuristic", beat_source="librosa")


def edl(cut_times, L=24.0) -> EDL:
    t = [0.0, *cut_times, L]
    shots = [Shot(slot_idx=j, start=t[j], end=t[j + 1], frame_start=round(t[j] * 30), frame_end=round(t[j + 1] * 30),
                  candidate_id=f"c{j}", asset_id=f"a{j}", kind="photo", src_in=None, src_out=None, framing="native",
                  kenburns=None, scores=SC, reason="") for j in range(len(t) - 1)]
    plan = EditPlan(vibe="", vibe_queries=[], avoid_queries=[], pacing=0, energy_bias=0, chronological=False,
                    target_length_s=0, excerpt_hint="auto", excerpt_start_s=0, opening_query="", closing_query="",
                    segments=[], must_include=[], exclude=[], rationale="")
    return EDL(run_id="r", created=datetime.now(), system="full", seed=0, fps=30, width=1280, height=720,
               song_hash="h", excerpt=Excerpt(song_start=0, song_end=L, fade_out=1.5), plan=plan, shots=shots,
               total_cost=0)


def feats(n):
    rng = np.random.default_rng(0)
    e = rng.normal(size=(n, 768)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return SimpleNamespace(arrays={"emb": e, "dup_cluster": np.arange(n), "visual_arousal": np.linspace(-1, 1, n)},
                           index=lambda: {f"c{j}": j for j in range(n)})


def test_metrics_known():
    pr = profile()
    on = edl([2.0, 4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 20.0])
    m = metrics.compute(on, pr, feats(len(on.shots)), np.linspace(0, 1, len(on.shots)), n_assets=20)
    assert m["beat_hit_rate"] == 1.0 and m["downbeat_hit_rate"] == 1.0 and m["mean_offset_ms"] == 0.0
    assert m["boundary_hit_rate"] == 1.0
    shifted = edl([t + 0.2 for t in [2.0, 4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 20.0]])
    m2 = metrics.compute(shifted, pr, feats(len(shifted.shots)), np.linspace(0, 1, len(shifted.shots)), n_assets=20)
    assert m2["beat_hit_rate"] == 0.0
    assert m2["mean_offset_ms"] == pytest.approx(200, abs=1)
    assert m["coverage"] == pytest.approx(len(on.shots) / 20)
    assert m["same_source_adjacent"] == 0 and m["dup_violations"] == 0


def test_pace_coupling_sign():
    pr = profile()  # intensity rises over time
    rising = edl([4.0, 8.0, 11.0, 13.0, 15.0, 16.5, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0])
    falling = edl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 14.0, 18.0])
    assert metrics.pace_coupling(metrics.internal_cuts(rising), pr) > 0.5
    assert metrics.pace_coupling(metrics.internal_cuts(falling), pr) < -0.5


def test_report_builds(fresh_root):
    import json
    from montage import paths
    from montage.eval import report
    rows = []
    rng = np.random.default_rng(0)
    for system in ("full", "naive", "no_mood"):
        for seed in range(3):
            rows.append({"album": "a", "song": "s", "system": system, "seed": seed,
                         **{m: float(rng.random()) for m in report.METRICS}})
    root = paths.data_root() / "eval"
    root.mkdir(parents=True, exist_ok=True)
    (root / "grid.json").write_text(json.dumps({"rows": rows, "fallbacks": []}))
    out = report.build()
    assert (root / "report.md").exists() and len(out["figures"]) == 3
    assert out["takeaways"] and "full" in out["summary"]


def test_render_check_matching():
    from montage.eval.render_check import match, unmatched
    assert unmatched([10, 50, 101], [10, 100]) == [50]
    assert match([10, 50, 101], [10, 100]) == (pytest.approx(2 / 3), 1.0)
