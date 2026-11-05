"""Phase 14: temporal emotion head (model, loss, crops, training on a synthetic temporal task, inference fallbacks)."""

from __future__ import annotations

import io

import numpy as np
import pytest

ARCH = {"layers": 13, "dim": 768, "proj_dim": 32, "hidden": 16, "dropout": 0.0}


def _split(n_songs: int, steps: int, rng, lag: int = 4):
    """Synthetic DEAM-like split. Each window's features carry a white-noise signal s_t along one direction; the
    target is a function of the PREVIOUS `lag` signals (like an annotator reacting late). A per-window head sees
    only s_t, which is independent of the target, so only a model that looks across windows can predict it."""
    from training.emotion.dataset import Split
    u = np.random.default_rng(123).normal(size=768)
    u /= np.linalg.norm(u)
    feats, ys, songs, times = [], [], [], []
    for sid in range(n_songs):
        s = rng.normal(size=steps + lag)
        x = s[lag:, None] * u[None] * 4 + rng.normal(scale=0.3, size=(steps, 768))
        y = np.tanh(0.5 * np.array([s[t:t + lag].sum() for t in range(steps)]))
        feats.append(np.repeat(x[:, None], 13, 1))
        ys.append(np.stack([0.5 * y, y], 1))           # valence, arousal
        songs.append(np.full(steps, sid))
        times.append(15 + 0.5 * np.arange(steps))
    return Split(np.concatenate(feats).astype(np.float16), np.concatenate(ys).astype(np.float32),
                 np.concatenate(songs), np.concatenate(times))


def test_temporal_head_padding_matches_unpadded():
    torch = pytest.importorskip("torch")
    from montage.music.emotion import make_temporal_head
    torch.manual_seed(0)
    head = make_temporal_head(**ARCH).eval()
    a, b = torch.randn(1, 12, 13, 768), torch.randn(1, 7, 13, 768)
    padded = torch.cat([a, torch.cat([b, torch.zeros(1, 5, 13, 768)], 1)])
    with torch.no_grad():
        out = head(padded, torch.tensor([12, 7]))
        alone = head(b)
    assert out.shape == (2, 12, 2)
    assert torch.allclose(out[1, :7], alone[0], atol=1e-5)   # padding does not leak into real steps
    assert out.abs().max() <= 1


def test_seq_loss_prefers_matching_shape():
    torch = pytest.importorskip("torch")
    from training.emotion.temporal import seq_loss
    t = torch.linspace(0, 6, 40)
    y = torch.stack([0.3 * torch.sin(t), 0.5 * torch.cos(t)], -1)[None].repeat(3, 1, 1)
    mask = torch.ones(3, 40, dtype=torch.bool)
    mask[2, 30:] = False
    assert float(seq_loss(y, y, mask)) == pytest.approx(0.0, abs=1e-3)
    flat = y.mean(1, keepdim=True).expand_as(y)            # right level per song, no movement
    assert float(seq_loss(flat, y, mask)) > float(seq_loss(y + 0.05, y, mask)) + 0.5


def test_crops_stay_inside_songs():
    from training.emotion.temporal import crop_index, full_index, song_runs
    from training.emotion.dataset import Split
    song = np.repeat([5, 9, 11], [30, 8, 50])
    s = Split(np.zeros((len(song), 1, 1), np.float16), np.zeros((len(song), 2), np.float32), song, np.zeros(len(song)))
    ids, starts, lengths = song_runs(s)
    assert list(ids) == [5, 9, 11] and list(starts) == [0, 30, 38] and list(lengths) == [30, 8, 50]
    idx, lens = crop_index(starts, lengths, np.random.default_rng(0), 20, 40)
    for b in range(3):
        row = idx[b, :lens[b]]
        assert np.all(song[row] == ids[b]) and np.all(np.diff(row) == 1)
        assert np.all(idx[b, lens[b]:] == -1)
    assert lens[1] == 8 and 20 <= lens[0] <= 30
    fidx, flens = full_index(starts, lengths)
    assert fidx.shape == (3, 50) and list(flens) == [30, 8, 50] and fidx[2, 49] == 87


def test_temporal_head_learns_what_window_head_cannot():
    pytest.importorskip("torch")
    from training.emotion import evaluate
    from training.emotion.temporal import predict_seq, train_temporal
    from training.emotion.train import predict, train
    rng = np.random.default_rng(0)
    splits = {"train": _split(64, 30, rng), "val": _split(16, 30, rng)}
    te = _split(16, 30, rng)
    window, _, _ = train(splits, seed=0, epochs=15, batch=256)
    temporal, _, _ = train_temporal(splits, ARCH, seed=0, epochs=40, patience=40, batch=16, min_crop=10, max_crop=30,
                                    lr=3e-3)
    w = evaluate.metrics(predict(window, te.feats), te.y, te.song)["arousal"]["ccc_per_song"]
    t = evaluate.metrics(predict_seq(temporal, te), te.y, te.song)["arousal"]["ccc_per_song"]
    assert w < 0.15, w                 # about 0 over seeds 0-2: the window head cannot see the past
    assert t > 0.35 and t > w + 0.3, t  # 0.52-0.59 over seeds 0-2


def _save(path, ckpt):
    import torch
    from montage import paths
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    paths.atomic_write(path, buf.getvalue())


def test_inference_prefers_temporal_then_window_then_heuristic(fresh_root, monkeypatch):
    pytest.importorskip("torch")
    from montage.music import emotion
    for name in ("_temporal", "_head"):
        monkeypatch.setattr(emotion, name, None)
    grid = np.arange(0, 30, 0.5)
    monkeypatch.setattr(emotion, "mert_window_features",
                        lambda w: np.random.default_rng(0).normal(size=(len(w), 13, 768)).astype(np.float16))
    curves = {k: np.random.default_rng(1).normal(size=len(grid)).astype(np.float32)
              for k in ("rms_db", "onset_density", "centroid")}
    y24 = np.zeros(24000 * 30, np.float32)

    assert emotion.emotion_model_id() == "heuristic"
    _save(emotion.head_path(), {"state_dict": emotion.make_head().state_dict()})
    a, v, src, mid = emotion.predict_emotion_detail(y24, grid, curves)
    assert src == "mert" and mid.startswith("window:") and mid == emotion.emotion_model_id()
    assert len(a) == len(v) == len(grid)

    _save(emotion.temporal_path(), {"state_dict": emotion.make_temporal_head(**ARCH).state_dict(), "arch": ARCH})
    a, v, src, mid = emotion.predict_emotion_detail(y24, grid, curves)
    assert src == "mert" and mid.startswith("temporal:") and mid == emotion.emotion_model_id()
    assert len(a) == len(grid) and np.all(np.abs(a) <= 1) and np.all(np.abs(v) <= 1)

    monkeypatch.setattr(emotion, "_temporal", None)          # a broken temporal checkpoint falls back
    emotion.temporal_path().write_bytes(b"not a checkpoint")
    a, v, src, mid = emotion.predict_emotion_detail(y24, grid, curves)
    assert src == "mert" and mid.startswith("window:")

    from montage import config
    with config.override({"music.emotion.use_temporal": False}):
        assert emotion.emotion_model_id().startswith("window:")


def test_training_job_end_to_end(fresh_root, monkeypatch):
    """main() on tiny synthetic DEAM shards: trains both heads, writes the report, deploys, and the app's
    inference then picks up the deployed temporal head."""
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from montage import cache, config
    from montage.music import emotion
    from training.emotion import temporal
    from training.emotion.extract_mert import shard_dir
    rng = np.random.default_rng(0)
    for sid in [*range(2, 32), 2001, 2002]:          # 30 clips (80/10/10) + 2 full songs
        s = _split(1, 50 if sid > 2000 else 24, rng)
        cache.save_npz(shard_dir() / f"{sid}.npz", times=s.times.astype(np.float32), feats=s.feats,
                       valence=s.y[:, 0], arousal=s.y[:, 1])
    with config.override({"music.emotion.temporal.proj_dim": 16, "music.emotion.temporal.hidden": 8}):
        assert temporal.main(["--seeds", "0", "--epochs", "3", "--force-deploy"]) == 0
    report = (temporal.out_dir() / "report.md").read_text()
    assert "temporal head" in report and "window head (smoothed)" in report and "deployed" in report
    assert len(list(temporal.out_dir().glob("curves_*.png"))) >= 3
    monkeypatch.setattr(emotion, "_temporal", None)
    monkeypatch.setattr(emotion, "mert_window_features", lambda w: np.zeros((len(w), 13, 768), np.float16))
    grid = np.arange(0, 20, 0.5)
    a, v, src, mid = emotion.predict_emotion_detail(np.zeros(24000 * 20, np.float32), grid)
    assert src == "mert" and mid.startswith("temporal:") and len(a) == len(grid)


def test_model_id_changes_when_weights_change(fresh_root):
    pytest.importorskip("torch")
    from montage.music import emotion
    _save(emotion.temporal_path(), {"state_dict": emotion.make_temporal_head(**ARCH).state_dict(), "arch": ARCH})
    first = emotion.emotion_model_id()
    _save(emotion.temporal_path(), {"state_dict": emotion.make_temporal_head(**ARCH).state_dict(), "arch": ARCH,
                                    "note": "retrained"})
    assert first.startswith("temporal:") and emotion.emotion_model_id() != first
