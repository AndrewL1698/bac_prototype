"""Live A/B vote: one CSV per user under DATA_ROOT/eval/votes/, analysis with an exact binomial test.

    python -m montage.eval.vote --analyze
"""

from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime
from math import sqrt
from pathlib import Path

from montage import paths

FIELDS = ["timestamp", "rater", "pair_id", "left_system", "right_system", "choice"]


def votes_dir() -> Path:
    d = paths.data_root() / "eval" / "votes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_vote(rater: str, pair_id: str, left_system: str, right_system: str, choice: str) -> None:
    """choice: 'left', 'right' or 'tie'. Appends atomically to this user's file (no shared-file writes)."""
    if choice not in ("left", "right", "tie"):
        raise ValueError(choice)
    p = votes_dir() / f"{paths.user()}.csv"
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    if not old:
        w.writeheader()
    w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "rater": (rater or "").strip()[:40],
                "pair_id": pair_id, "left_system": left_system, "right_system": right_system, "choice": choice})
    paths.atomic_write(p, old + buf.getvalue())


def load_votes() -> list[dict]:
    rows = []
    for p in sorted(votes_dir().glob("*.csv")):
        with open(p, encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def analyze(rows: list[dict] | None = None, system: str = "full") -> dict:
    from scipy.stats import binomtest
    rows = load_votes() if rows is None else rows
    wins = losses = ties = 0
    for r in rows:
        if system not in (r["left_system"], r["right_system"]):
            continue
        if r["choice"] == "tie":
            ties += 1
            continue
        picked = r["left_system"] if r["choice"] == "left" else r["right_system"]
        wins += picked == system
        losses += picked != system
    n = wins + losses
    p = binomtest(wins, n, 0.5).pvalue if n else 1.0
    lo, hi = wilson(wins, n)
    return {"wins": wins, "losses": losses, "ties": ties, "win_rate": wins / n if n else 0.0, "p_value": p,
            "ci95": (lo, hi)}


def summary_line(res: dict, system: str = "full") -> str:
    lo, hi = res["ci95"]
    return (f"'{system}' preferred in {res['wins']}/{res['wins'] + res['losses']} votes "
            f"({res['win_rate']:.0%}, 95% CI {lo:.0%}-{hi:.0%}, binomial p={res['p_value']:.3g}; {res['ties']} ties)")


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--system", default="full")
    args = ap.parse_args(argv)
    if args.analyze:
        print(summary_line(analyze(system=args.system), args.system))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
