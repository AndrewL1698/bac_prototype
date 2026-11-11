"""Temporal emotion head (spec 14): a BiGRU over each song's MERT window sequence, trained on DEAM.

    python scripts/run_job.py --name emo_temporal -- python -m training.emotion.temporal

Uses the phase 9B MERT shards and song split (training/emotion/dataset.py), so test songs are the same as 9B's.
Baselines, all trained here on the same split and seeds: predict-mean, the 9B window head, and the window head
smoothed the way the app smooths (montage.music.profile). Main metric: CCC within each song, averaged over songs.
Writes DATA_ROOT/eval/emotion_temporal/{report.md, metrics.json, curves_<song>.png} and, only if the temporal head
beats the window head on validation (or --force-deploy), DATA_ROOT/weights/emotion_temporal.pt (atomic).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from montage import jobs, paths
from montage.config import get_config
from montage.jobs import git_info
from montage.log import get_logger
from montage.music.emotion import make_temporal_head, temporal_path
from montage.music.energy import gauss_smooth
from training.emotion import evaluate
from training.emotion.dataset import Split, load_all
from training.emotion.extract_mert import shard_dir

log = get_logger("emo_temporal")
DIMS = evaluate.DIMS
SMOOTH_S = 1.0          # the app smooths arousal/valence with gauss_smooth(x, 1.0) (montage.music.profile)
STEP_S = 0.5            # DEAM annotations and our windows are on a 2 Hz grid


def out_dir() -> Path:
    d = paths.data_root() / "eval" / "emotion_temporal"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------- sequences ----------------

def song_runs(s: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(song ids, start index, length) of each song's contiguous run of windows in a Split."""
    if not len(s.song):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    starts = np.flatnonzero(np.r_[True, s.song[1:] != s.song[:-1]])
    lengths = np.diff(np.r_[starts, len(s.song)])
    return s.song[starts], starts, lengths


def shift_split(s: Split, k: int) -> Split:
    """Pair each song's window i with its annotation i + k (k grid steps later); drops the last k windows per song."""
    if k == 0:
        return s
    _, starts, lengths = song_runs(s)
    x = np.concatenate([np.arange(st, st + n - k) for st, n in zip(starts, lengths) if n > k]).astype(int)
    return Split(s.feats[x], s.y[x + k], s.song[x], s.times[x])


def crop_index(starts: np.ndarray, lengths: np.ndarray, rng: np.random.Generator,
               min_crop: int, max_crop: int) -> tuple[np.ndarray, np.ndarray]:
    """One random crop per song: (B, L) window indices padded with -1, and (B,) crop lengths.
    Random lengths and offsets keep the model from learning where a clip starts (DEAM clips all start at 15 s)."""
    lens = np.minimum(lengths, rng.integers(min_crop, max_crop + 1, size=len(lengths)))
    offs = rng.integers(0, lengths - lens + 1)
    idx = np.full((len(lengths), int(lens.max())), -1, np.int64)
    for b, (st, o, n) in enumerate(zip(starts, offs, lens)):
        idx[b, :n] = np.arange(st + o, st + o + n)
    return idx, lens


def full_index(starts: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whole songs: (B, L) indices padded with -1, and (B,) lengths."""
    idx = np.full((len(lengths), int(lengths.max()) if len(lengths) else 0), -1, np.int64)
    for b, (st, n) in enumerate(zip(starts, lengths)):
        idx[b, :n] = np.arange(st, st + n)
    return idx, lengths.copy()


# ---------------- loss ----------------

def seq_loss(p, y, mask, w_ccc: float = 0.5, w_song: float = 0.5, eps: float = 1e-4):
    """MSE + w_ccc * (1 - pooled CCC) + w_song * (1 - within-song CCC), summed over valence and arousal.
    p, y: (B, T, 2); mask: (B, T) bool. Within-song CCC compares per-song-centered curves, so it rewards following
    changes inside a song, which pooled CCC barely sees (song-level means dominate it)."""
    m = mask.unsqueeze(-1).to(p.dtype)
    pv, yv = p[mask], y[mask]
    mse = ((pv - yv) ** 2).mean()
    pm, ym = pv.mean(0), yv.mean(0)
    cov = ((pv - pm) * (yv - ym)).mean(0)
    pooled = 2 * cov / (pv.var(0, unbiased=False) + yv.var(0, unbiased=False) + (pm - ym) ** 2 + 1e-8)
    n = m.sum(1).clamp(min=1)
    dp = (p - (p * m).sum(1, keepdim=True) / n[:, None]) * m
    dy = (y - (y * m).sum(1, keepdim=True) / n[:, None]) * m
    within = 2 * (dp * dy).sum(1) / n / ((dp ** 2).sum(1) / n + (dy ** 2).sum(1) / n + eps)   # (B, 2)
    return mse + w_ccc * (1 - pooled).sum() + w_song * (1 - within.mean(0)).sum()


# ---------------- train / predict ----------------

def predict_seq(head, s: Split, batch: int = 16) -> np.ndarray:
    """(n, 2) predictions for every window of a Split, each song run as one whole sequence."""
    import torch
    dev = next(head.parameters()).device
    _, starts, lengths = song_runs(s)
    out = np.zeros((len(s.song), 2), np.float32)
    head.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            idx, lens = full_index(starts[i:i + batch], lengths[i:i + batch])
            x = torch.from_numpy(s.feats[np.maximum(idx, 0)].astype(np.float32)).to(dev)
            p = head(x, torch.from_numpy(lens)).cpu().numpy()
            out[idx[idx >= 0]] = p[idx >= 0]
    return out


def val_score(m: dict) -> float:
    """Model selection: half pooled CCC (level), half per-song CCC (shape), averaged over valence and arousal."""
    return float(np.mean([0.5 * m[d]["ccc"] + 0.5 * m[d]["ccc_per_song"] for d in DIMS]))


def train_temporal(splits, arch: dict, seed: int = 0, epochs: int = 60, patience: int = 8, batch: int = 32,
                   min_crop: int = 20, max_crop: int = 60, lr: float = 1e-3, progress=None):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = splits["train"], splits["val"]
    X = torch.from_numpy(tr.feats).to(dev)                 # fp16 on the GPU, as in 9B
    Y = torch.from_numpy(tr.y).to(dev)
    _, starts, lengths = song_runs(tr)
    head = make_temporal_head(**arch).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state, bad, hist = -1e9, None, 0, []
    for ep in range(epochs):
        head.train()
        order = rng.permutation(len(starts))
        tot, nb = 0.0, 0
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            idx, lens = crop_index(starts[sel], lengths[sel], rng, min_crop, max_crop)
            it = torch.from_numpy(np.maximum(idx, 0)).to(dev)
            mask = torch.from_numpy(idx >= 0).to(dev)
            p = head(X[it].float(), torch.from_numpy(lens))
            loss = seq_loss(p, Y[it], mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        vm = evaluate.metrics(predict_seq(head, va), va.y, va.song)
        score = val_score(vm)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_score": score,
                     "val_arousal_ccc_per_song": vm["arousal"]["ccc_per_song"]})
        log.info("seed %d epoch %d loss %.4f val score %.4f (arousal per-song CCC %.3f)", seed, ep, tot / max(nb, 1),
                 score, vm["arousal"]["ccc_per_song"])
        if progress:
            progress((ep + 1) / epochs, f"seed {seed} epoch {ep + 1}: val {score:.3f}")
        if score > best + 1e-4:
            best, best_state, bad = score, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head.cpu(), hist, best


def smooth_per_song(pred: np.ndarray, song: np.ndarray) -> np.ndarray:
    out = pred.copy()
    for s in np.unique(song):
        m = song == s
        for k in range(pred.shape[1]):
            out[m, k] = gauss_smooth(pred[m, k], SMOOTH_S)
    return out


# ---------------- report ----------------

def _mean_std(ms: list[dict]) -> dict:
    """Metrics dicts (one per seed) -> {dim: {metric: [mean, std]}}."""
    return {d: {k: [float(np.mean([m[d][k] for m in ms])), float(np.std([m[d][k] for m in ms]))]
                for k in ms[0][d]} for d in DIMS}


def write_report(res: dict) -> None:
    lines = ["# Temporal music emotion head (MERT-v1-95M + DEAM, spec 14)", "",
             f"Songs: {res['songs']}; seeds: {res['seeds']}; annotation lag: {res['lag_s']} s. Mean +- std over seeds. "
             f"Main metric: CCC within each song (per song), averaged over songs. 'smoothed' = gauss_smooth({SMOOTH_S} s) per song, as the app does.", ""]
    for split in ("test", "full"):
        if split not in res:
            continue
        lines += [f"## {split} set", "",
                  "| Model | Arousal CCC per song | Valence CCC per song | Arousal CCC | Valence CCC | Arousal RMSE | "
                  "Valence RMSE |", "|---|---|---|---|---|---|---|"]
        for name, m in res[split].items():
            def f(d, k):
                mu, sd = m[d][k]
                return f"{mu:.3f} +- {sd:.3f}" if sd > 0 else f"{mu:.3f}"
            lines.append(f"| {name} | {f('arousal', 'ccc_per_song')} | {f('valence', 'ccc_per_song')} | "
                         f"{f('arousal', 'ccc')} | {f('valence', 'ccc')} | {f('arousal', 'rmse')} | "
                         f"{f('valence', 'rmse')} |")
        lines.append("")
    lines += [f"Deploy: {res['deploy']['decision']} ({res['deploy']['reason']}).", "",
              "Figures: `curves_<song>.png` (annotated vs window head vs temporal head).", ""]
    (out_dir() / "report.md").write_text("\n".join(lines))
    (out_dir() / "metrics.json").write_text(json.dumps(res, indent=1))


def plots(splits, preds: dict[str, dict[str, np.ndarray]], ids: dict[str, list[int]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split, n in (("test", 3), ("full", 2)):
        s = splits[split]
        for sid in ids[split][:n]:
            m = s.song == sid
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=110, sharex=True)
            for k, (d, ax) in enumerate(zip(DIMS, axes)):
                ax.plot(s.times[m], s.y[m, k], color="#212121", lw=2, label="annotated")
                ax.plot(s.times[m], preds[split]["window head (smoothed)"][m, k], color="#1565c0", lw=1.5,
                        label="window head (smoothed)")
                ax.plot(s.times[m], preds[split]["temporal head"][m, k], color="#c62828", lw=1.5, label="temporal head")
                ax.set_ylim(-1, 1)
                ax.set_ylabel(d, fontsize=12)
            axes[0].legend(fontsize=9, ncol=3)
            axes[0].set_title(f"DEAM {split} song {sid}", fontsize=13)
            axes[1].set_xlabel("seconds", fontsize=12)
            fig.tight_layout()
            fig.savefig(out_dir() / f"curves_{sid}.png")
            plt.close(fig)


# ---------------- main ----------------

def main(argv: list[str] | None = None) -> int:
    import torch
    from training.emotion.train import predict, train
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--force-deploy", action="store_true", help="deploy even if it does not beat the window head")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--lag", type=float, default=0.0,
                    help="pair each window with the annotation this many seconds later, so curves follow the music, "
                         "not the annotators' delayed reaction (14.3)")
    args = ap.parse_args(argv)
    paths.load_colab_env()
    cfg = get_config()
    arch = {"layers": 13, "dim": 768, **cfg.music.emotion.temporal.model_dump()}
    jobs.report(0.02, "loading MERT shards", force=True)
    ids, splits = load_all(shard_dir(), seed=cfg.seed)
    splits = {k: shift_split(v, round(args.lag / STEP_S)) for k, v in splits.items()}
    log.info("songs: %s", {k: len(v) for k, v in ids.items()})
    n = len(args.seeds)
    window, temporal = [], []
    for i, seed in enumerate(args.seeds):
        jobs.report(0.05 + 0.3 * i / n, f"window head, seed {seed}", force=True)
        head, _, _ = train(splits, seed=seed)
        window.append(head)
        lo = 0.35 + 0.5 * i / n
        head, hist, _ = train_temporal(splits, arch, seed=seed, epochs=args.epochs,
                                       progress=lambda f, msg, lo=lo: jobs.report(lo + 0.5 / n * f, msg))
        temporal.append((head, hist))
    jobs.report(0.88, "evaluating", force=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    window = [h.to(dev) for h in window]
    temporal = [(h.to(dev), hist) for h, hist in temporal]
    mean = splits["train"].y.mean(0)
    res: dict = {"songs": {k: len(v) for k, v in ids.items()}, "seeds": args.seeds, "arch": arch, "lag_s": args.lag,
                 "history": [h for _, h in temporal]}
    per_seed = {}
    for split in ("val", "test", "full"):
        s = splits[split]
        if not len(s.y):
            continue
        rows: dict[str, list[dict]] = {"predict-mean": [], "window head": [], "window head (smoothed)": [],
                                       "temporal head": [], "temporal head (smoothed)": []}
        for wh, (th, _) in zip(window, temporal):
            pw, pt = predict(wh, s.feats), predict_seq(th, s)
            rows["predict-mean"].append(evaluate.metrics(np.tile(mean, (len(s.y), 1)), s.y, s.song))
            rows["window head"].append(evaluate.metrics(pw, s.y, s.song))
            rows["window head (smoothed)"].append(evaluate.metrics(smooth_per_song(pw, s.song), s.y, s.song))
            rows["temporal head"].append(evaluate.metrics(pt, s.y, s.song))
            rows["temporal head (smoothed)"].append(evaluate.metrics(smooth_per_song(pt, s.song), s.y, s.song))
        per_seed[split] = rows
        res[split] = {k: _mean_std(v) for k, v in rows.items()}
    # Deploy the best temporal seed if, on validation, it beats the best window head (smoothed, as the app runs it).
    v_t = [val_score(m) for m in per_seed["val"]["temporal head (smoothed)"]]
    v_w = [val_score(m) for m in per_seed["val"]["window head (smoothed)"]]
    bi = int(np.argmax(v_t))
    better = v_t[bi] > max(v_w)
    reason = f"val score temporal {v_t[bi]:.3f} (seed {args.seeds[bi]}) vs window {max(v_w):.3f}"
    deploy = not args.no_deploy and (better or args.force_deploy)
    res["deploy"] = {"decision": "deployed" if deploy else "not deployed", "reason": reason,
                     "seed": args.seeds[bi]}
    best_pred = {sp: {"window head (smoothed)": smooth_per_song(predict(window[bi], splits[sp].feats), splits[sp].song),
                      "temporal head": predict_seq(temporal[bi][0], splits[sp])}
                 for sp in ("test", "full") if len(splits[sp].y)}
    plots(splits, best_pred, ids)
    if deploy:
        head = temporal[bi][0].cpu()
        _, commit = git_info()
        ckpt = {"state_dict": head.state_dict(), "arch": arch, "arch_type": "temporal",
                "layer_logits": head.layer_logits.detach().numpy().tolist(), "outputs": ["valence", "arousal"],
                "window_s": cfg.music.emotion.window_s, "align": "centered", "metrics": res["test"],
                "feature_version": f"{cfg.music.emotion.model}@{cfg.music.emotion.revision}",
                "created": datetime.now().isoformat(timespec="seconds"), "trained_by": paths.user(),
                "git_commit": commit, "seed": args.seeds[bi], "lag_s": args.lag}
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        paths.atomic_write(temporal_path(), buf.getvalue())
    write_report(res)
    t = res["test"]["temporal head (smoothed)"]
    w = res["test"]["window head (smoothed)"]
    print(f"{res['deploy']['decision']} ({reason}). Test arousal CCC per song: temporal "
          f"{t['arousal']['ccc_per_song'][0]:.3f} vs window {w['arousal']['ccc_per_song'][0]:.3f}. "
          f"Report: {out_dir() / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
