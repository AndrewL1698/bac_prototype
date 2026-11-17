"""Personalization from pins and excludes (spec 15 section 6).

Each recut that applies Keep/Remove stores, per session, the sum and count of the CLIP embeddings of the kept
(pos) and removed (neg) shots in DATA_ROOT/prefs/<user>.json. Later cuts nudge importance toward shots that look
like the kept ones and away from shots like the removed ones, within each modality, with a weight that grows
with the evidence and never exceeds beta_max.

    python -m montage.importance.personalize --show     # evidence counts for the current user
    python -m montage.importance.personalize --reset    # forget this user's preferences
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from montage import paths
from montage.config import get_config
from montage.importance.blend import rank_normalize
from montage.log import get_logger

log = get_logger("personalize")
VERSION = 1


def prefs_path(user: str | None = None):
    return paths.data_root() / "prefs" / f"{user or paths.user()}.json"


def load(user: str | None = None) -> dict:
    p = prefs_path(user)
    if not p.exists():
        return {"version": VERSION, "sessions": {}}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("unreadable prefs file %s: ignored", p)
        return {"version": VERSION, "sessions": {}}
    return d if d.get("version") == VERSION else {"version": VERSION, "sessions": {}}


def _side(emb: np.ndarray, rows: list[int]) -> tuple[list[float], int]:
    if not rows:
        return [], 0
    e = emb[rows].astype(np.float64)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    return [round(float(v), 6) for v in e.sum(0)], len(rows)


def record(session_id: str, features, pins: list[str], excluded: list[str], user: str | None = None) -> dict:
    """Replace this session's evidence with its current pins and excludes (empty: the session is dropped)."""
    idx = features.index()
    emb = np.asarray(features.arrays["emb"])
    pos_sum, pos_n = _side(emb, [idx[c] for c in dict.fromkeys(pins) if c in idx])
    neg_sum, neg_n = _side(emb, [idx[c] for c in dict.fromkeys(excluded) if c in idx and c not in set(pins)])
    d = load(user)
    sessions = d["sessions"]
    sessions.pop(session_id, None)
    if pos_n or neg_n:
        sessions[session_id] = {"updated": time.time(), "pos_sum": pos_sum, "pos_n": pos_n,
                                "neg_sum": neg_sum, "neg_n": neg_n}
    keep = sorted(sessions, key=lambda k: sessions[k]["updated"])[-get_config().importance.personalize.max_sessions:]
    d["sessions"] = {k: sessions[k] for k in keep}
    paths.atomic_write(prefs_path(user), json.dumps(d))
    return d


def evidence(d: dict) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """(mean kept embedding or None, mean removed embedding or None, number of kept + removed shots)."""
    def mean(side: str):
        n = sum(s[f"{side}_n"] for s in d["sessions"].values())
        if not n:
            return None, 0
        tot = np.sum([np.asarray(s[f"{side}_sum"]) for s in d["sessions"].values() if s[f"{side}_n"]], axis=0)
        return tot / n, n
    pos, n_pos = mean("pos")
    neg, n_neg = mean("neg")
    return pos, neg, n_pos + n_neg


def preference(emb: np.ndarray, pos: np.ndarray | None, neg: np.ndarray | None) -> np.ndarray:
    """cos(e, mean kept) - cos(e, mean removed), each term only when that side has evidence."""
    e = np.asarray(emb, np.float64)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    out = np.zeros(len(e))
    for m, sign in ((pos, 1.0), (neg, -1.0)):
        if m is not None:
            out += sign * (e @ (m / max(np.linalg.norm(m), 1e-8)))
    return out


def beta(n: int) -> float:
    c = get_config().importance.personalize
    return 0.0 if n < c.min_events else c.beta_max * n / (n + c.half_evidence)


def nudge(final: np.ndarray, pref: np.ndarray, groups: np.ndarray, b: float) -> np.ndarray:
    """rank01((1 - b) * final + b * rank01(pref)) within each group (modality)."""
    if b <= 0:
        return np.asarray(final, float)
    return rank_normalize((1 - b) * np.asarray(final, float) + b * rank_normalize(pref, groups), groups)


def apply(final, features, user: str | None = None) -> tuple[np.ndarray, int]:
    """(final scores, number of kept + removed shots used). Unchanged, with 0, when disabled, without a prefs
    file, or with fewer than min_events shots."""
    final = np.asarray(final, float)
    if not get_config().importance.personalize.enabled:
        return final, 0
    pos, neg, n = evidence(load(user))
    b = beta(n)
    if b <= 0:
        return final, 0
    arr = features.arrays
    groups = np.where(np.asarray(arr["is_video"], bool), 1, 0)
    return nudge(final, preference(arr["emb"], pos, neg), groups, b), n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--show", action="store_true", help="print this user's evidence counts")
    g.add_argument("--reset", action="store_true", help="delete this user's preferences")
    a = ap.parse_args(argv)
    p = prefs_path()
    if a.reset:
        p.unlink(missing_ok=True)
        print(f"preferences for {paths.user()} deleted")
        return 0
    d = load()
    pos_n = sum(s["pos_n"] for s in d["sessions"].values())
    neg_n = sum(s["neg_n"] for s in d["sessions"].values())
    print(f"user {paths.user()}: {len(d['sessions'])} sessions, {pos_n} kept and {neg_n} removed shots; "
          f"beta {beta(pos_n + neg_n):.3f} (file {p})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
