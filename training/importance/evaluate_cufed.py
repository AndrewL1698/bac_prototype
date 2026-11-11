"""Phase 15 days 4-7 (spec 15 section 5): train and evaluate the importance ranker with CUFED photo albums next
to the SumMe/TVSum videos, compare with the deployed ranker, and decide whether to deploy.

    python scripts/run_job.py --name cufed_eval -- python -m training.importance.evaluate_cufed
    python -m training.importance.evaluate_cufed --deploy      # after reading the report: make the candidate live

evaluate.py (phase 10A, videos only) is left unchanged so its published numbers stay reproducible; this module
reuses its ranking helper and the same RankNet trainer (train.fit).

Data: windows.npz (video windows) + cufed_photos.npz (album photos); same ranker_features columns.
CV: `folds` folds by group (a video or an album), each dataset split on its own; video folds are drawn exactly as
evaluate.py draws them. Per held-out group: Kendall tau-b and Spearman rho (tie-aware: CUFED scores are discrete),
averaged per dataset.
Rows: random, aesthetic, heuristic H, the deployed ranker as is (out of sample on CUFED only: it was trained on
the videos), the deployed configuration re-run in this CV (the fair reference on the videos), learned
{sources} x {kinds}, and blends of the selected model with H at alpha in {0, .25, .5, .75, 1}.
Selection: the (model, alpha) with the best CUFED tau among those whose SumMe tau is not more than
`clearly_worse` below the deployed configuration's. Deploy only if it beats today's ranker on CUFED (with
`require_ci`: the 95% bootstrap CI of the per-album tau gain is above 0). Today's ranker is the live checkpoint as
is when it never saw CUFED; once a CUFED-trained ranker is live, its as-is score is in sample, so its
configuration re-run in this CV is the baseline instead. No live checkpoint: the heuristic.
A run always writes the report and a candidate checkpoint, never the live weights: every app loads those, so
going live is the separate --deploy step, which refuses unless the decision says deploy and keeps the previous
file as <name>.prev.pt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from montage import cache, config, jobs, paths
from montage.config import get_config
from montage.jobs import git_info
from montage.log import get_logger
from training.common import datasets_dir
from training.importance.evaluate import ALPHAS, rank01_within
from training.importance.train import fit

log = get_logger("cufed_eval")
KEYS = ("X", "y", "heuristic", "aesthetic", "group", "dataset")
VIDEO_SETS = ("summe", "tvsum")
CANDIDATE = "candidate.pt"


def eval_dir() -> Path:
    d = paths.data_root() / "eval" / "importance_cufed"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- data ----------

def load_data() -> dict[str, np.ndarray]:
    """windows.npz + cufed_photos.npz concatenated on the shared keys (t_mid: NaN for photos)."""
    parts = []
    for name, hint in (("windows.npz", "training.importance.extract_clip"),
                       ("cufed_photos.npz", "training.importance.cufed_features")):
        z = cache.load_npz(datasets_dir("importance") / name)
        if z is None:
            raise SystemExit(f"{name} not found: run python -m {hint} first")
        part = {k: z[k] for k in KEYS}
        part["t_mid"] = z["t_mid"].astype(float) if "t_mid" in z else np.full(len(z["y"]), np.nan)
        part["event"] = z["event"].astype(str) if "event" in z else np.full(len(z["y"]), "")
        parts.append(part)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["group"], data["dataset"] = data["group"].astype(str), data["dataset"].astype(str)
    return data


def datasets_in(D: np.ndarray) -> list[str]:
    present = set(D.tolist())
    return [d for d in VIDEO_SETS if d in present] + sorted(present - set(VIDEO_SETS))


def source_mask(src: str, D: np.ndarray) -> np.ndarray:
    if src == "all":
        return np.ones(len(D), bool)
    if src == "both":
        return np.isin(D, VIDEO_SETS)
    return D == src


def make_folds(G: np.ndarray, D: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold per row; groups of each dataset dealt round-robin after a seeded shuffle. SumMe then TVSum are drawn
    first with the same generator as evaluate.folds, so the video folds match phase 10A exactly."""
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for d in datasets_in(D):
        for i, g in enumerate(rng.permutation(sorted(set(G[D == d])))):
            fold_of[str(g)] = i % k
    return np.array([fold_of[g] for g in G])


# ---------- metrics ----------

def group_scores(scores: np.ndarray, y: np.ndarray, G: np.ndarray) -> dict[str, tuple[float, float]]:
    """{group: (Kendall tau-b, Spearman rho)} for groups with at least 3 items and label variance."""
    out = {}
    for g in np.unique(G):
        m = G == g
        if m.sum() < 3 or np.std(y[m]) == 0:
            continue
        t, r = kendalltau(scores[m], y[m]).statistic, spearmanr(scores[m], y[m]).statistic
        out[str(g)] = (0.0 if np.isnan(t) else float(t), 0.0 if np.isnan(r) else float(r))
    return out


def summarize(per_group: dict[str, tuple[float, float]], D_of: dict[str, str], datasets: list[str]) -> dict:
    acc: dict[str, list] = defaultdict(list)
    for g, v in per_group.items():
        acc[D_of[g]].append(v)
    return {d: {"tau": float(np.mean([a for a, _ in acc[d]])) if acc[d] else float("nan"),
                "rho": float(np.mean([b for _, b in acc[d]])) if acc[d] else float("nan"),
                "n_groups": len(acc[d])} for d in datasets}


def bootstrap_gain(new: dict[str, tuple], old: dict[str, tuple], groups: list[str], n: int,
                   seed: int) -> dict:
    """Mean per-group tau gain (new - old) over `groups`, with a 95% percentile bootstrap CI over groups."""
    diffs = np.array([new[g][0] - old[g][0] for g in groups if g in new and g in old])
    if len(diffs) == 0:
        return {"gain": float("nan"), "ci": [float("nan"), float("nan")], "n_groups": 0, "share_improved": float("nan")}
    rng = np.random.default_rng(seed)
    boots = diffs[rng.integers(0, len(diffs), (n, len(diffs)))].mean(1)
    return {"gain": float(diffs.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_groups": int(len(diffs)), "share_improved": float((diffs > 0).mean())}


# ---------- cross-validation ----------

def fingerprint(data: dict[str, np.ndarray], *extra) -> str:
    h = hashlib.sha1()
    for k in ("X", "y", "group"):
        h.update(np.ascontiguousarray(data[k]).tobytes())
    h.update(repr(extra).encode())
    return h.hexdigest()[:16]


def cv_predictions(data: dict, F: np.ndarray, src: str, kind: str, k: int, epochs: int, seed: int,
                   fp: str) -> np.ndarray:
    """Out-of-fold scores for every row from a model trained on `src` rows of the other folds. Cached per
    config (keyed by the data fingerprint), so a rerun after a Colab disconnect skips finished configs."""
    path = eval_dir() / "oof" / f"{src}_{kind}.npz"
    z = cache.load_npz(path)
    if z is not None and str(z["fingerprint"]) == fp:
        return z["pred"]
    X, y, G, D = data["X"], data["y"], data["group"], data["dataset"]
    pred = np.zeros(len(y))
    for f in range(k):
        tr = (F != f) & source_mask(src, D)
        tr_groups = sorted(set(G[tr]))
        val_g = set(np.random.default_rng(seed + f).permutation(tr_groups)[: max(1, len(tr_groups) // 10)])
        inner_val = tr & np.isin(G, list(val_g))
        inner_tr = tr & ~inner_val
        r = fit(X[inner_tr], y[inner_tr], G[inner_tr], kind, seed=seed, epochs=epochs,
                val=(X[inner_val], y[inner_val], G[inner_val]))
        te = F == f
        pred[te] = r.score(X[te])
    cache.save_npz(path, pred=pred, fingerprint=np.array(fp))
    return pred


# ---------- deployed ranker ----------

def deployed() -> tuple[object | None, dict]:
    """(Ranker, checkpoint metadata) for the live weights, or (None, {})."""
    from montage.importance.ranker import Ranker
    path = paths.weights_dir() / get_config().importance.ranker_weights
    r = Ranker.load(path)
    if r is None:
        return None, {}
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    alpha = r.blend_alpha if r.blend_alpha is not None else get_config().importance.blend_alpha
    return r, {"type": ck.get("type"), "trained_on": ck.get("trained_on"), "blend_alpha": float(alpha),
               "created": ck.get("created"), "git_commit": ck.get("git_commit")}


# ---------- selection ----------

def choose(cands: dict[tuple[str, float], dict], ref_summe: float | None, clearly_worse: float,
           target: str = "cufed") -> tuple[str, float]:
    """(model, alpha) with the best `target` tau among candidates whose SumMe tau is at least
    ref_summe - clearly_worse (no SumMe reference: all eligible). Ties prefer the smaller alpha."""
    def summe(c):
        return cands[c].get("summe", {}).get("tau", float("nan"))
    eligible = [c for c in cands if ref_summe is None or np.isnan(summe(c)) or summe(c) >= ref_summe - clearly_worse]
    if not eligible:
        raise ValueError("no candidate is within clearly_worse of the deployed SumMe tau")
    return max(eligible, key=lambda c: (cands[c][target]["tau"], -c[1]))


def blend(L: np.ndarray, H: np.ndarray, G: np.ndarray, alpha: float) -> np.ndarray:
    """What the app ranks by: alpha * rank(learned) + (1 - alpha) * rank(heuristic), within each group."""
    return alpha * rank01_within(L, G) + (1 - alpha) * rank01_within(H, G)


# ---------- main ----------

def run() -> dict:
    cfg = get_config()
    ec = cfg.importance.cufed.eval
    seed = cfg.seed
    data = load_data()
    X, y, G, D, H, A = data["X"], data["y"], data["group"], data["dataset"], data["heuristic"], data["aesthetic"]
    sets = datasets_in(D)
    if "cufed" not in sets:
        raise SystemExit("no CUFED rows in the data")
    D_of = {g: d for g, d in zip(G.tolist(), D.tolist())}
    F = make_folds(G, D, ec.folds, seed)
    fp = fingerprint(data, seed, ec.folds, ec.epochs, cfg.features.version)
    log.info("%d rows, %d groups (%s)", len(y), len(D_of),
             ", ".join(f"{d}: {sum(1 for v in D_of.values() if v == d)}" for d in sets))

    scores: dict[str, np.ndarray] = {"random": np.random.default_rng(seed).random(len(y)),
                                     "aesthetic": A.astype(float), "heuristic": H.astype(float)}
    dep, dep_meta = deployed()
    if dep is not None:
        scores["deployed_as_is"] = blend(dep.score(X), H, G, dep_meta["blend_alpha"])
    configs = [(s, k) for s in ec.sources for k in ec.kinds if source_mask(s, D).any()]
    oof: dict[str, np.ndarray] = {}
    for i, (src, kind) in enumerate(configs):
        jobs.report(0.02 + 0.78 * i / len(configs), f"CV {src}/{kind} ({i + 1}/{len(configs)})", force=True)
        oof[f"learned_{src}_{kind}"] = cv_predictions(data, F, src, kind, ec.folds, ec.epochs, seed, fp)
    scores.update(oof)

    per_group = {name: group_scores(s, y, G) for name, s in scores.items()}
    table = {name: summarize(pg, D_of, sets) for name, pg in per_group.items()}

    # every (learned model, alpha) blend is a candidate; alpha 0 is the heuristic alone
    cands, cand_pg = {}, {}
    for name, s in oof.items():
        for a in ALPHAS:
            pg = group_scores(blend(s, H, G, a), y, G)
            cands[(name, a)], cand_pg[(name, a)] = summarize(pg, D_of, sets), pg
    ref_key = None
    if dep is not None:
        name = f"learned_{dep_meta['trained_on']}_{dep_meta['type']}"
        if name in oof:
            ref_key = (name, dep_meta["blend_alpha"])
            if ref_key not in cands:
                pg = group_scores(blend(oof[name], H, G, ref_key[1]), y, G)
                cands[ref_key], cand_pg[ref_key] = summarize(pg, D_of, sets), pg
            table["deployed_config_cv"] = cands[ref_key]
    ref_summe = cands[ref_key]["summe"]["tau"] if ref_key and "summe" in sets else None
    chosen = choose(cands, ref_summe, ec.clearly_worse)
    for a in ALPHAS:
        table[f"blend_{chosen[0]}_alpha_{a:g}"] = cands[(chosen[0], a)]

    # deploy decision: chosen blend vs what the app does today on CUFED albums. A live model that was trained on
    # CUFED has seen these albums, so its as-is score is in sample; then its configuration re-run in this CV is
    # the fair baseline.
    if dep is None:
        baseline_name, base_pg = "heuristic", per_group["heuristic"]
    elif dep_meta.get("trained_on") in ("cufed", "all") and ref_key is not None:
        baseline_name, base_pg = "deployed_config_cv", cand_pg[ref_key]
    else:
        baseline_name, base_pg = "deployed_as_is", per_group["deployed_as_is"]
    albums = sorted(g for g, d in D_of.items() if d == "cufed")
    gain = bootstrap_gain(cand_pg[chosen], base_pg, albums, ec.bootstrap, seed)
    beats = gain["gain"] > 0 and (gain["ci"][0] > 0 if ec.require_ci else True)
    summe_note = (f"SumMe tau {cands[chosen]['summe']['tau']:.3f} vs deployed configuration "
                  f"{ref_summe:.3f} (allowed down to {ref_summe - ec.clearly_worse:.3f})"
                  if ref_summe is not None else "no SumMe reference")
    reason = (f"chosen {chosen[0]} at alpha {chosen[1]:g}: CUFED tau {cands[chosen]['cufed']['tau']:.3f} vs "
              f"{baseline_name} {table[baseline_name]['cufed']['tau']:.3f}, gain {gain['gain']:+.3f} "
              f"(95% CI {gain['ci'][0]:+.3f} to {gain['ci'][1]:+.3f}, "
              f"{gain['share_improved']:.0%} of albums improved); "
              f"{summe_note}. " + ("Deploy." if beats else "Keep the deployed ranker: " + (
                  "the gain is not clearly above 0." if gain["gain"] > 0 else "no gain on CUFED.")))
    log.info(reason)

    jobs.report(0.85, "retraining the chosen model on all its data", force=True)
    src, kind = chosen[0].split("_")[1], chosen[0].split("_")[2]
    tr = source_mask(src, D)
    final = fit(X[tr], y[tr], G[tr], kind, seed=seed, epochs=ec.epochs)
    _, commit = git_info()
    metrics = {"table": table, "chosen": {"model": chosen[0], "alpha": chosen[1], "trained_on": src, "type": kind},
               "baseline": baseline_name, "deployed": dep_meta, "gain_vs_baseline": gain,
               "decision": {"deploy": bool(beats), "reason": reason},
               "n_rows": int(len(y)), "groups": {d: table["heuristic"][d]["n_groups"] for d in sets},
               "per_event": per_event(cand_pg[chosen], base_pg, data),
               "config": ec.model_dump(), "seed": seed, "created": datetime.now().isoformat(timespec="seconds")}
    import torch
    ckpt = {"type": kind, "state_dict": final.model.state_dict(), "feat_mean": final.mean.tolist(),
            "feat_std": final.std.tolist(), "blend_alpha": float(chosen[1]), "metrics": metrics,
            "feature_version": cfg.features.version, "trained_on": src, "phase": 15,
            "created": metrics["created"], "trained_by": paths.user(), "git_commit": commit}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    payload = buf.getvalue()
    paths.atomic_write(eval_dir() / CANDIDATE, payload)
    metrics["candidate_sha1"] = hashlib.sha1(payload).hexdigest()
    cache.write_json(eval_dir() / "metrics.json", metrics)
    write_report(metrics, base_pg, cand_pg[chosen], baseline_name, albums)
    jobs.report(1.0, "done", force=True)
    return metrics


def per_event(new: dict, old: dict, data: dict) -> dict:
    """Mean CUFED tau per event type: chosen blend vs baseline."""
    ev_of = {}
    for g, d, e in zip(data["group"].tolist(), data["dataset"].tolist(), data["event"].tolist()):
        if d == "cufed":
            ev_of[g] = e
    out: dict[str, dict] = {}
    for ev in sorted(set(ev_of.values())):
        gs = [g for g, e in ev_of.items() if e == ev and g in new and g in old]
        if gs:
            out[ev] = {"n_albums": len(gs), "chosen_tau": float(np.mean([new[g][0] for g in gs])),
                       "baseline_tau": float(np.mean([old[g][0] for g in gs]))}
    return out


def write_report(metrics: dict, base_pg: dict, chosen_pg: dict, baseline_name: str, albums: list[str]) -> None:
    sets = list(metrics["groups"])
    head = " | ".join(f"{d} tau | {d} rho" for d in sets)
    lines = ["# Importance ranker with CUFED photo albums (phase 15)", "",
             f"{metrics['n_rows']} labeled rows; groups: "
             + ", ".join(f"{d} {n}" for d, n in metrics["groups"].items())
             + f". {len(metrics['config']['sources'])} training sets x {len(metrics['config']['kinds'])} model types, "
             f"{metrics['config']['folds']}-fold CV by group; Kendall tau-b / Spearman rho within each "
             "held-out group.", "",
             f"**Decision:** {metrics['decision']['reason']}", "",
             f"| Model | {head} |", "|---" * (1 + 2 * len(sets)) + "|"]
    for name, r in metrics["table"].items():
        lines.append(f"| {name} | " + " | ".join(f"{r[d]['tau']:.3f} | {r[d]['rho']:.3f}" for d in sets) + " |")
    lines += ["", "`deployed_as_is` is the live checkpoint as it is: out of sample only on data it was not trained on "
              "(a ranker trained on the videos has never seen CUFED). `deployed_config_cv` re-runs its configuration "
              "in this CV: the fair reference on the videos, and on CUFED once a CUFED-trained ranker is live.",
              "", "Per event type (CUFED tau, chosen vs baseline):", "", "| Event | Albums | Chosen | Baseline |",
              "|---|---|---|---|"]
    for ev, r in sorted(metrics["per_event"].items(), key=lambda kv: -kv[1]["n_albums"]):
        lines.append(f"| {ev} | {r['n_albums']} | {r['chosen_tau']:.3f} | {r['baseline_tau']:.3f} |")
    lines += ["", "Figure: `per_album_tau.png` (each dot is a held-out album).", ""]
    paths.atomic_write(eval_dir() / "report.md", "\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        both = [g for g in albums if g in chosen_pg and g in base_pg]
        xs, ys = [base_pg[g][0] for g in both], [chosen_pg[g][0] for g in both]
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
        ax.axline((0, 0), slope=1, color="#9e9e9e", lw=1, ls="--")
        ax.scatter(xs, ys, s=18, color="#1565c0", alpha=0.7)
        ax.set_xlabel(f"{baseline_name}: Kendall tau per album", fontsize=11)
        ax.set_ylabel("chosen: Kendall tau per album", fontsize=11)
        ax.set_title("Held-out CUFED albums (above the line = better)", fontsize=12)
        fig.tight_layout()
        fig.savefig(eval_dir() / "per_album_tau.png")
        plt.close(fig)
    except Exception as e:  # the figure is a nicety; the numbers are in report.md
        log.warning("figure skipped (%s)", e)


def deploy() -> int:
    """Make the candidate live, only if the last run decided to deploy. Keeps the previous file as .prev.pt."""
    m = cache.read_json(eval_dir() / "metrics.json")
    cand = eval_dir() / CANDIDATE
    if not m or not cand.exists():
        print("no evaluation found: run python -m training.importance.evaluate_cufed first", flush=True)
        return 2
    payload = cand.read_bytes()
    if hashlib.sha1(payload).hexdigest() != m.get("candidate_sha1"):
        print("candidate.pt does not match metrics.json (another run in between?); rerun the evaluation", flush=True)
        return 2
    if not m["decision"]["deploy"]:
        print("not deploying: " + m["decision"]["reason"], flush=True)
        return 1
    live = paths.weights_dir() / get_config().importance.ranker_weights
    if live.exists():
        paths.atomic_write(live.with_suffix(".prev.pt"), live.read_bytes())
    paths.atomic_write(live, payload)
    print(f"deployed {m['chosen']['model']} at alpha {m['chosen']['alpha']:g} to {live.name} "
          f"(previous kept as {live.with_suffix('.prev.pt').name})", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15: ranker CV with CUFED, comparison, deploy decision")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--deploy", action="store_true", help="make the last run's candidate live (if it decided so)")
    a = ap.parse_args(argv)
    paths.load_colab_env()
    if a.set:
        config.set_overrides(a.set)
    if a.deploy:
        return deploy()
    m = run()
    print("decision:", m["decision"]["reason"], flush=True)
    print(f"report: {eval_dir() / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
