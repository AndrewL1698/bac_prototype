"""Does the temporal emotion head change the edits? (spec 14, step 14.4)

    python scripts/run_job.py --name emo_edit_window   -- python -m training.emotion.edit_compare --emotion window
    python scripts/run_job.py --name emo_edit_temporal -- python -m training.emotion.edit_compare --emotion temporal
    python -m training.emotion.edit_compare --compare

Plans the phase 11 eval albums x songs with the full editor (shared cached plans, 3 seeds), once with the 9B window
head and once with the deployed temporal head; nothing else changes. Writes rows and EDLs to
DATA_ROOT/eval/emotion_edit/<emotion>/ (the phase 11 grid files are not touched); --compare writes report.md there.
Mood and pace coupling are measured against each run's own music curves, so they show how well the edit follows its
curves, not whether the curves are right (DEAM scores answer that).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from montage import config, jobs, paths
from montage.log import get_logger

log = get_logger("emo_edit")
METRICS = ["mood_coupling", "pace_coupling", "beat_hit_rate", "boundary_hit_rate", "director_adherence",
           "quality_lift", "n_shots", "arousal_std", "shot_len_cv", "n_sections"]


def out_dir(name: str = "") -> Path:
    d = paths.data_root() / "eval" / "emotion_edit" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(emotion: str) -> None:
    from montage import pipeline
    from montage.eval import metrics
    from montage.eval.run_grid import albums, director_matrix, key, plan_for, songs
    config.set_overrides([f"music.emotion.use_temporal={emotion == 'temporal'}"])
    seeds = config.get_config().eval.seeds
    al, so = albums(), songs()
    rows, total = [], len(al) * len(so) * seeds
    for album in al:
        for song in so:
            s = pipeline.analyze([str(album)], str(song))
            st = pipeline.state(s)
            plan = plan_for(s, album, song)
            profile = pipeline._profile(st, plan)
            imp = np.asarray(st.importance.final)
            for seed in range(seeds):
                edl = pipeline.plan_edit(s, plan, seed, "full", run_id=f"emo_{emotion}_{key(album, song)}_{seed}",
                                         persist=False)
                paths.atomic_write(out_dir(emotion) / "edls" / f"{key(album, song)}__{seed}.json",
                                   edl.model_dump_json(indent=1))
                m = metrics.compute(edl, profile, st.features, imp, n_assets=len(st.media.assets),
                                    D=director_matrix(st, plan, profile, edl))
                lens = np.array([sh.end - sh.start for sh in edl.shots])
                rows.append({"album": album.name, "song": song.stem, "seed": seed, **m,
                             "arousal_std": float(np.std(profile.arousal)), "shot_len_cv": float(lens.std() / lens.mean()),
                             "n_sections": len(profile.sections), "emotion_model": st.music.emotion_model})
                jobs.report(len(rows) / total, f"{len(rows)}/{total} EDLs")
            log.info("%s x %s done (emotion model %s)", album.name, song.stem, st.music.emotion_model)
    paths.atomic_write(out_dir(emotion) / "rows.json", json.dumps(rows, indent=1))
    print(f"{emotion}: {len(rows)} EDLs, emotion model {sorted({r['emotion_model'] for r in rows})}", flush=True)


def changed_shots(a: Path, b: Path) -> float:
    """Share of screen time showing a different candidate in the two EDLs (sampled every 0.1 s)."""
    ea, eb = (json.loads(p.read_text())["shots"] for p in (a, b))
    t = np.arange(0, min(ea[-1]["end"], eb[-1]["end"]), 0.1)
    at = lambda e: [next(s["candidate_id"] for s in e if s["start"] <= x < s["end"]) for x in t]
    return float(np.mean([x != y for x, y in zip(at(ea), at(eb))]))


def compare() -> None:
    rows = {e: json.loads((out_dir(e) / "rows.json").read_text()) for e in ("window", "temporal")}
    f = lambda e, k: np.mean([r[k] for r in rows[e]])
    lines = ["# Edits with the window head vs the temporal head (spec 14, 14.4)", "",
             f"Full editor, phase 11 eval albums x songs, shared plans, {len(rows['window'])} EDLs per head. "
             "Mood/pace coupling are against each run's own curves (see the module docstring).", "",
             "| Metric | Window head | Temporal head |", "|---|---|---|"]
    lines += [f"| {k} | {f('window', k):.3f} | {f('temporal', k):.3f} |" for k in METRICS]
    edls = sorted((out_dir("window") / "edls").glob("*.json"))
    ch = [changed_shots(p, out_dir("temporal") / "edls" / p.name) for p in edls]
    lines += ["", f"Screen time showing a different shot: {np.mean(ch):.0%} (mean over {len(ch)} EDL pairs, "
              f"range {min(ch):.0%}-{max(ch):.0%}).", "", "Per album x song (seed means):", "",
              "| Album x song | Mood coupling (window -> temporal) | Arousal std (window -> temporal) |", "|---|---|---|"]
    for a, s in sorted({(r["album"], r["song"]) for r in rows["window"]}):
        g = lambda e, k: np.mean([r[k] for r in rows[e] if r["album"] == a and r["song"] == s])
        lines.append(f"| {a} x {s} | {g('window', 'mood_coupling'):+.2f} -> {g('temporal', 'mood_coupling'):+.2f} | "
                     f"{g('window', 'arousal_std'):.3f} -> {g('temporal', 'arousal_std'):.3f} |")
    (out_dir() / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion", choices=["window", "temporal"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args(argv)
    paths.load_colab_env()
    if args.emotion:
        run(args.emotion)
    if args.compare:
        compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
