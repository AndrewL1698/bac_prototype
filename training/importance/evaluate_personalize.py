"""Phase 15 days 8-10 (spec 15 section 6): does personalization from Keep/Remove help? Offline check on CUFED.

A simulated user edits albums of one event type: in each history album they keep the k photos with the highest
CUFED score and remove the k lowest. Each album is held out in turn, with every other album of its event type as
history ("same"), and with as many albums of other event types ("control": a user whose taste is unrelated).
Base scores: the chosen photo ranker's out-of-fold predictions from evaluate_cufed (the held-out album was never
trained on) at the chosen alpha. For each beta_max: mean CUFED tau gain with a 95% bootstrap CI over albums.

    python scripts/run_job.py --name personalize_eval -- python -m training.importance.evaluate_personalize
"""
from __future__ import annotations

import numpy as np

from montage import cache, config, jobs, paths
from montage.config import get_config
from montage.importance import personalize as P
from montage.log import get_logger
from training.importance import evaluate_cufed as ev

log = get_logger("personalize_eval")


def out_dir():
    d = paths.data_root() / "eval" / "personalize"
    d.mkdir(parents=True, exist_ok=True)
    return d


def history(emb: np.ndarray, y: np.ndarray, G: np.ndarray, albums: list[str], k: int):
    """(mean kept embedding, mean removed embedding, shots) for a user who kept the k best and removed the k
    worst photos of each album."""
    pos, neg = [], []
    for g in albums:
        rows = np.flatnonzero(G == g)
        order = rows[np.argsort(y[rows], kind="stable")]
        neg += order[:k].tolist()
        pos += order[-k:].tolist()
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return unit[pos].mean(0), unit[neg].mean(0), len(pos) + len(neg)


def personalized(base: np.ndarray, emb: np.ndarray, y: np.ndarray, G: np.ndarray, E: np.ndarray,
                 beta_max: float, k: int, seed: int) -> dict[str, np.ndarray]:
    """{"same": scores, "control": scores} with each album nudged by its history (albums without another
    album of their event type keep their base scores)."""
    rng = np.random.default_rng(seed)
    out = {"same": base.copy(), "control": base.copy()}
    event_of = {g: E[G == g][0] for g in np.unique(G)}
    for g in sorted(event_of):
        same = [a for a in event_of if a != g and event_of[a] == event_of[g]]
        if not same:
            continue
        others = sorted(a for a in event_of if event_of[a] != event_of[g])
        control = rng.choice(others, size=min(len(same), len(others)), replace=False).tolist()
        rows = G == g
        for name, albums in (("same", same), ("control", control)):
            pos, neg, n = history(emb, y, G, albums, k)
            with config.override({"importance.personalize.beta_max": beta_max}):
                b = P.beta(n)
            out[name][rows] = P.nudge(base[rows], P.preference(emb[rows], pos, neg), np.zeros(rows.sum()), b)
    return out


def run(base: np.ndarray, emb: np.ndarray, y: np.ndarray, G: np.ndarray, E: np.ndarray, seed: int = 0) -> dict:
    ec = get_config().importance.personalize.eval
    base_pg = ev.group_scores(base, y, G)
    event_of = {g: E[G == g][0] for g in np.unique(G)}
    n_of_type = {e: list(event_of.values()).count(e) for e in set(event_of.values())}
    albums = sorted(g for g in base_pg if n_of_type[event_of[g]] > 1)  # albums with same-type history
    rows = []
    for i, bm in enumerate(ec.beta_grid):
        new = personalized(base, emb, y, G, E, bm, ec.k, seed)
        row = {"beta_max": bm}
        for name, s in new.items():
            row[name] = ev.bootstrap_gain(ev.group_scores(s, y, G), base_pg, albums, ec.bootstrap, seed)
        rows.append(row)
        jobs.report((i + 1) / len(ec.beta_grid), f"beta_max {bm}")
    ok = [r["beta_max"] for r in rows if r["control"]["gain"] >= ec.control_floor and r["same"]["ci"][0] > 0]
    return {"rows": rows, "recommended_beta_max": max(ok) if ok else None, "n_albums": len(albums),
            "base_tau": float(np.mean([base_pg[g][0] for g in albums])) if albums else float("nan"), "k": ec.k}


def write_report(m: dict, base_name: str) -> None:
    def cell(r):
        return f"{r['gain']:+.3f} ({r['ci'][0]:+.3f} to {r['ci'][1]:+.3f}), {r['share_improved']:.0%} better"
    rec = m["recommended_beta_max"]
    lines = [f"Held-out CUFED albums with same-type history: {m['n_albums']}; base ({base_name}) mean tau "
             f"{m['base_tau']:.3f}. The simulated user keeps the {m['k']} best and removes the {m['k']} worst "
             "photos per history album.", "",
             f"**Recommendation:** beta_max {rec}" if rec is not None else
             "**Recommendation:** keep personalization off (no beta_max helps without hurting the control).", "",
             "| beta_max | same event type: tau gain (95% CI) | unrelated user (control): tau gain (95% CI) |",
             "|---|---|---|"]
    lines += [f"| {r['beta_max']:g} | {cell(r['same'])} | {cell(r['control'])} |" for r in m["rows"]]
    paths.atomic_write(out_dir() / "report.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    data = ev.load_data()
    cm = cache.read_json(ev.eval_dir() / "metrics.json")
    if not cm:
        raise SystemExit("run python -m training.importance.evaluate_cufed first")
    ch = cm["chosen"]
    z = cache.load_npz(ev.eval_dir() / "oof" / f"{ch['trained_on']}_{ch['type']}.npz")
    if z is None or len(z["pred"]) != len(data["y"]):
        raise SystemExit("out-of-fold predictions missing or stale: rerun evaluate_cufed")
    G = data["group"]
    base = ev.blend(z["pred"], data["heuristic"], G, ch["alpha"])
    c = data["dataset"] == "cufed"
    m = run(base[c], data["X"][c, :768], data["y"][c], G[c], data["event"][c], seed=get_config().seed)
    m["base"] = f"{ch['model']} at alpha {ch['alpha']:g}, out of fold"
    cache.write_json(out_dir() / "metrics.json", m)
    write_report(m, m["base"])
    log.info("recommended beta_max: %s", m["recommended_beta_max"])
    print(f"recommended beta_max: {m['recommended_beta_max']}\nreport: {out_dir() / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
