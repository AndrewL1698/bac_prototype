"""Cross-validate, select, retrain and deploy the importance ranker (spec 10 sections 5-6).

    python scripts/run_job.py --name rank_train -- python -m training.importance.evaluate

5-fold CV by video (folds stratified so each mixes SumMe and TVSum). Per held-out video: Kendall tau and
Spearman rho between scores and window labels, averaged per dataset. Rows: random, aesthetic only,
heuristic H, learned {summe, tvsum, both} x {linear, mlp}, and blends of the best learned model with H at
alpha in {0, .25, .5, .75, 1} (rank-normalized within video). Selection: best mean tau on SumMe unless clearly
worse on TVSum. The chosen model is retrained on all videos and saved to DATA_ROOT/weights/importance_ranker.pt
with its blend_alpha; if no learned blend beats the heuristic, alpha = 0 ships (an honest result).
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from montage import cache, jobs, paths
from montage.config import get_config
from montage.jobs import git_info
from montage.log import get_logger
from training.common import datasets_dir
from training.importance.train import fit

log = get_logger("rank_eval")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CLEARLY_WORSE = 0.03  # tau points on TVSum


def out_dir() -> Path:
    d = paths.data_root() / "eval" / "importance"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rank01_within(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores))
    for g in np.unique(groups):
        m = groups == g
        out[m] = (rankdata(scores[m]) - 1) / max(m.sum() - 1, 1)
    return out


def per_video(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, datasets: np.ndarray) -> dict:
    res: dict[str, list] = {"summe": [], "tvsum": []}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 3 or np.std(y[m]) == 0:
            continue
        t = kendalltau(scores[m], y[m]).statistic
        r = spearmanr(scores[m], y[m]).statistic
        res[datasets[m][0]].append((0.0 if np.isnan(t) else t, 0.0 if np.isnan(r) else r))
    return {d: {"tau": float(np.mean([a for a, _ in v])) if v else float("nan"),
                "rho": float(np.mean([b for _, b in v])) if v else float("nan"), "n_videos": len(v)}
            for d, v in res.items()}


def folds(groups: np.ndarray, datasets: np.ndarray, k: int = 5, seed: int = 0) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    out = {}
    for d in ("summe", "tvsum"):
        vids = sorted(set(groups[datasets == d]))
        for i, v in enumerate(rng.permutation(vids)):
            out[str(v)] = i % k
    return out


def main() -> int:
    import torch
    paths.load_colab_env()
    cfg = get_config()
    z = cache.load_npz(datasets_dir("importance") / "windows.npz")
    X, y, G, D = z["X"], z["y"], z["group"].astype(str), z["dataset"].astype(str)
    H, A = z["heuristic"], z["aesthetic"]
    fold_of = folds(G, D, seed=cfg.seed)
    F = np.array([fold_of[g] for g in G])
    rng = np.random.default_rng(cfg.seed)
    oof = {"random": rng.random(len(y)), "aesthetic": A.astype(float), "heuristic": H.astype(float)}
    configs = [(src, kind) for src in ("summe", "tvsum", "both") for kind in ("linear", "mlp")]
    for ci, (src, kind) in enumerate(configs):
        pred = np.zeros(len(y))
        for f in range(5):
            tr = (F != f) & ((D == src) if src != "both" else True)
            tr_groups = sorted(set(G[tr]))
            val_g = set(np.random.default_rng(cfg.seed + f).permutation(tr_groups)[: max(1, len(tr_groups) // 10)])
            inner_val = tr & np.isin(G, list(val_g))
            inner_tr = tr & ~inner_val
            r = fit(X[inner_tr], y[inner_tr], G[inner_tr], kind, seed=cfg.seed,
                    val=(X[inner_val], y[inner_val], G[inner_val]))
            te = F == f
            pred[te] = r.score(X[te])
            jobs.report(0.05 + 0.75 * (ci * 5 + f + 1) / (len(configs) * 5), f"CV {src}/{kind} fold {f + 1}")
        oof[f"learned_{src}_{kind}"] = pred
    table = {name: per_video(s, y, G, D) for name, s in oof.items()}
    learned = [n for n in table if n.startswith("learned_")]
    best = max(learned, key=lambda n: table[n]["summe"]["tau"])
    alt = max(learned, key=lambda n: table[n]["tvsum"]["tau"])
    if table[alt]["tvsum"]["tau"] - table[best]["tvsum"]["tau"] > CLEARLY_WORSE and \
            table[best]["summe"]["tau"] - table[alt]["summe"]["tau"] < CLEARLY_WORSE:
        best = alt  # best on SumMe was clearly worse on TVSum
    rH = rank01_within(H, G)
    rL = rank01_within(oof[best], G)
    for a in ALPHAS:
        table[f"blend_alpha_{a:g}"] = per_video(a * rL + (1 - a) * rH, y, G, D)
    alpha = max(ALPHAS, key=lambda a: (table[f"blend_alpha_{a:g}"]["summe"]["tau"], -a))
    src, kind = best.split("_")[1], best.split("_")[2]
    reason = (f"best SumMe tau among learned models: {best} ({table[best]['summe']['tau']:.3f}); blend alpha {alpha:g} "
              f"has the best SumMe tau ({table[f'blend_alpha_{alpha:g}']['summe']['tau']:.3f}) vs heuristic "
              f"({table['heuristic']['summe']['tau']:.3f})")
    log.info(reason)
    jobs.report(0.85, "retraining the chosen model on all videos", force=True)
    tr = (D == src) if src != "both" else np.ones(len(y), bool)
    final = fit(X[tr], y[tr], G[tr], kind, seed=cfg.seed, epochs=30)
    _, commit = git_info()
    metrics = {"cv": table, "chosen": best, "blend_alpha": alpha, "reason": reason,
               "n_windows": int(len(y)), "n_videos": int(len(set(G)))}
    ckpt = {"type": kind, "state_dict": final.model.state_dict(), "feat_mean": final.mean.tolist(),
            "feat_std": final.std.tolist(), "blend_alpha": alpha, "metrics": metrics,
            "feature_version": cfg.features.version, "trained_on": src,
            "created": datetime.now().isoformat(timespec="seconds"), "trained_by": paths.user(), "git_commit": commit}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    paths.atomic_write(paths.weights_dir() / cfg.importance.ranker_weights, buf.getvalue())
    write_report(table, metrics, oof, y, G, D, z["t_mid"])
    print(f"deployed ranker: {best}, blend_alpha {alpha:g}; SumMe tau {table[best]['summe']['tau']:.3f} vs heuristic "
          f"{table['heuristic']['summe']['tau']:.3f}", flush=True)
    return 0


def write_report(table, metrics, oof, y, G, D, t_mid) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out_dir() / "metrics.json").write_text(json.dumps(metrics, indent=1))
    lines = ["# Importance ranker (SumMe + TVSum, 5-fold CV by video)", "",
             f"{metrics['n_videos']} videos, {metrics['n_windows']} labeled 3 s windows. Chosen: **{metrics['chosen']}**, "
             f"blend alpha **{metrics['blend_alpha']:g}**. {metrics['reason']}.", "",
             "| Model | SumMe Kendall tau | SumMe Spearman rho | TVSum Kendall tau | TVSum Spearman rho |",
             "|---|---|---|---|---|"]
    for name, r in table.items():
        lines.append(f"| {name} | {r['summe']['tau']:.3f} | {r['summe']['rho']:.3f} | {r['tvsum']['tau']:.3f} | "
                     f"{r['tvsum']['rho']:.3f} |")
    lines += ["", "Figures: `heldout_<video>.png` (predicted vs label over time).", ""]
    (out_dir() / "report.md").write_text("\n".join(lines))
    best = oof[metrics["chosen"]]
    for d in ("summe", "tvsum"):
        g = sorted(set(G[D == d]))[0]
        m = G == g
        o = np.argsort(t_mid[m])
        fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
        ax.plot(t_mid[m][o], y[m][o], color="#424242", lw=2, label="annotated importance")
        s = best[m][o]
        ax.plot(t_mid[m][o], (s - s.min()) / (s.max() - s.min() + 1e-9), color="#c62828", lw=2, ls="--",
                label="ranker (held out, scaled)")
        ax.set_xlabel("seconds", fontsize=12)
        ax.set_title(f"{g}: held-out prediction vs label", fontsize=13)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir() / f"heldout_{g.replace('/', '_')}.png")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
