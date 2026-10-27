"""Emotion head evaluation vs predict-mean and a best-layer ridge probe (spec 09 section 5).

Metrics per dimension on test (45 s clips) and full (58 full songs): RMSE, Pearson r, CCC pooled over windows,
plus CCC averaged per song. Outputs in DATA_ROOT/eval/emotion/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from montage import paths
from montage.log import get_logger
from training.emotion.dataset import ccc

log = get_logger("emo_eval")
DIMS = ("valence", "arousal")


def out_dir() -> Path:
    d = paths.data_root() / "eval" / "emotion"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mean_ccc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([ccc(pred[:, k], y[:, k]) for k in range(2)]))


def metrics(pred: np.ndarray, y: np.ndarray, song: np.ndarray) -> dict:
    out = {}
    for k, d in enumerate(DIMS):
        p, t = pred[:, k], y[:, k]
        per_song = [ccc(p[song == s], t[song == s]) for s in np.unique(song)]
        out[d] = {"rmse": float(np.sqrt(np.mean((p - t) ** 2))),
                  "pearson": float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-9 else 0.0,
                  "ccc": ccc(p, t), "ccc_per_song": float(np.nanmean(per_song))}
    return out


def ridge_probe(splits) -> tuple[dict, dict[str, np.ndarray]]:
    from sklearn.linear_model import Ridge
    tr, va = splits["train"], splits["val"]
    best = (-1e9, None, None)
    for layer in range(tr.feats.shape[1]):
        Xtr, Xva = tr.feats[:, layer].astype(np.float32), va.feats[:, layer].astype(np.float32)
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=alpha).fit(Xtr, tr.y)
            score = mean_ccc(m.predict(Xva), va.y)
            if score > best[0]:
                best = (score, (layer, alpha), m)
    (layer, alpha), m = best[1], best[2]
    log.info("ridge probe: best layer %d alpha %g (val mean CCC %.3f)", layer, alpha, best[0])
    preds = {k: np.clip(m.predict(s.feats[:, layer].astype(np.float32)), -1, 1) for k, s in splits.items() if len(s.y)}
    return {"layer": layer, "alpha": alpha, "val_mean_ccc": best[0]}, preds


def plots(head, splits, ids) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from training.emotion.train import predict
    w = torch.softmax(head.layer_logits.detach(), 0).numpy()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
    ax.bar(range(len(w)), w, color="#1565c0")
    ax.set_xlabel("MERT layer", fontsize=12)
    ax.set_ylabel("learned weight", fontsize=12)
    ax.set_title("Which MERT layers the emotion head uses", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir() / "layer_weights.png")
    plt.close(fig)
    te = splits["test"]
    for sid in ids["test"][:3]:
        m = te.song == sid
        p = predict(head, te.feats[m])
        fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
        for k, (d, c) in enumerate(zip(DIMS, ("#2e7d32", "#c62828"))):
            ax.plot(te.times[m], te.y[m, k], color=c, lw=2, label=f"{d} (annotated)")
            ax.plot(te.times[m], p[:, k], color=c, lw=2, ls="--", label=f"{d} (predicted)")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("seconds", fontsize=12)
        ax.set_title(f"DEAM test song {sid}", fontsize=13)
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir() / f"curves_{sid}.png")
        plt.close(fig)


def run(head, splits, ids, hist) -> dict:
    from training.emotion.train import predict
    probe_info, probe_preds = ridge_probe(splits)
    mean = splits["train"].y.mean(0)
    res: dict = {"songs": {k: len(v) for k, v in ids.items()},
                 "windows": {k: int(len(s.y)) for k, s in splits.items()}, "probe": probe_info, "history": hist}
    for split in ("test", "full"):
        s = splits[split]
        if not len(s.y):
            continue
        res[split] = {"predict_mean": metrics(np.tile(mean, (len(s.y), 1)), s.y, s.song),
                      "ridge_probe": metrics(probe_preds[split], s.y, s.song),
                      "head": metrics(predict(head, s.feats), s.y, s.song)}
    plots(head, splits, ids)
    (out_dir() / "metrics.json").write_text(json.dumps(res, indent=1))
    lines = ["# Music emotion head (MERT-v1-95M + DEAM)", "",
             f"Songs: {res['songs']}; windows: {res['windows']}. Ridge probe: layer {probe_info['layer']}, "
             f"alpha {probe_info['alpha']:g}.", ""]
    for split in ("test", "full"):
        if split not in res:
            continue
        lines += [f"## {split} set", "", "| Model | Valence CCC | Arousal CCC | Valence RMSE | Arousal RMSE | "
                  "Valence r | Arousal r | Arousal CCC per song |", "|---|---|---|---|---|---|---|---|"]
        for name in ("predict_mean", "ridge_probe", "head"):
            m = res[split][name]
            lines.append(f"| {name} | {m['valence']['ccc']:.3f} | {m['arousal']['ccc']:.3f} | {m['valence']['rmse']:.3f} | "
                         f"{m['arousal']['rmse']:.3f} | {m['valence']['pearson']:.3f} | {m['arousal']['pearson']:.3f} | "
                         f"{m['arousal']['ccc_per_song']:.3f} |")
        lines.append("")
    lines += ["Figures: `layer_weights.png`, `curves_<song>.png` (3 test songs).", ""]
    (out_dir() / "report.md").write_text("\n".join(lines))
    return res
