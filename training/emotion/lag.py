"""Annotator-lag study (spec 14, open question 1): how much of the temporal head's gain is just learning the lag?

    python scripts/run_job.py --name emo_lag -- python -m training.emotion.lag

DEAM annotators react a few seconds late. For each lag (0-4 s), the annotation is shifted that much later than the
audio (window at t is paired with the annotation at t + lag), and the window head and the temporal head are retrained
and scored on the shifted targets. If the window head at its best lag catches up with the temporal head, the gain was
mostly lag; if the temporal head keeps its lead, it comes from context. Writes DATA_ROOT/eval/emotion_lag/report.md
and metrics.json. Never deploys.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from montage import jobs, paths
from montage.config import get_config
from montage.log import get_logger
from training.emotion import evaluate
from training.emotion.dataset import load_all
from training.emotion.extract_mert import shard_dir
from training.emotion.temporal import STEP_S, predict_seq, shift_split, smooth_per_song, train_temporal

log = get_logger("emo_lag")
def main(argv: list[str] | None = None) -> int:
    from training.emotion.train import predict, train
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], help="seconds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args(argv)
    paths.load_colab_env()
    cfg = get_config()
    arch = {"layers": 13, "dim": 768, **cfg.music.emotion.temporal.model_dump()}
    jobs.report(0.02, "loading MERT shards", force=True)
    _, base = load_all(shard_dir(), seed=cfg.seed)
    rows = []
    for i, lag in enumerate(args.lags):
        splits = {k: shift_split(v, round(lag / STEP_S)) for k, v in base.items()}
        test = splits["test"]
        scores: dict[str, list[dict]] = {"window head (smoothed)": [], "temporal head (smoothed)": []}
        for seed in args.seeds:
            jobs.report(0.05 + 0.95 * i / len(args.lags), f"lag {lag} s, seed {seed}", force=True)
            wh, _, _ = train(splits, seed=seed)
            th, _, _ = train_temporal(splits, arch, seed=seed)
            for name, p in (("window head (smoothed)", predict(wh, test.feats)),
                            ("temporal head (smoothed)", predict_seq(th, test))):
                scores[name].append(evaluate.metrics(smooth_per_song(p, test.song), test.y, test.song))
        for name, ms in scores.items():
            row = {"lag_s": lag, "model": name}
            for d in evaluate.DIMS:
                for key in ("ccc_per_song", "ccc"):
                    row[f"{d}_{key}"] = [float(np.mean([m[d][key] for m in ms])), float(np.std([m[d][key] for m in ms]))]
            rows.append(row)
            log.info("lag %.1f s %s: arousal per song %.3f, valence per song %.3f", lag, name,
                     row["arousal_ccc_per_song"][0], row["valence_ccc_per_song"][0])
    write_report(rows, args.seeds)
    return 0


def write_report(rows: list[dict], seeds: list[int]) -> None:
    out = paths.data_root() / "eval" / "emotion_lag"
    out.mkdir(parents=True, exist_ok=True)
    f = lambda v: f"{v[0]:.3f} +- {v[1]:.3f}"
    lines = ["# Annotator-lag study (spec 14)", "",
             f"DEAM test set, seeds {seeds}, mean +- std. Lag = how much later than the audio the annotation is taken. "
             "Both heads retrained per lag and smoothed as the app does.", "",
             "| Lag (s) | Model | Arousal CCC per song | Valence CCC per song | Arousal CCC | Valence CCC |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['lag_s']} | {r['model']} | {f(r['arousal_ccc_per_song'])} | {f(r['valence_ccc_per_song'])} | "
              f"{f(r['arousal_ccc'])} | {f(r['valence_ccc'])} |" for r in rows]
    best = {m: max((r for r in rows if r["model"] == m), key=lambda r: r["arousal_ccc_per_song"][0])
            for m in ("window head (smoothed)", "temporal head (smoothed)")}
    lines += ["", "Best lag by arousal CCC per song: " + "; ".join(
        f"{m} {r['lag_s']} s ({r['arousal_ccc_per_song'][0]:.3f})" for m, r in best.items()) + ".", ""]
    (out / "report.md").write_text("\n".join(lines))
    (out / "metrics.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
