"""Evaluation grid (spec 11 section 1): albums x songs x systems x seeds -> EDLs + metrics (no rendering).

    python scripts/run_job.py --name eval_grid -- python -m montage.eval.run_grid

Albums: DATA_ROOT/eval/albums/<name>/ (optional brief.txt). Songs: DATA_ROOT/eval/songs/. One plan per
(album, song) via the director, shared by every system, so the comparison isolates the editor. EDLs go to
DATA_ROOT/eval/edls/, plans to eval/plans/, metric rows to eval/grid.json; then the report is built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from montage import jobs, paths
from montage.config import get_config
from montage.log import get_logger

log = get_logger("eval_grid")
MEDIA_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".mp4", ".mov", ".m4v", ".avi", ".mkv"}
SONG_EXT = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}


def eval_root() -> Path:
    d = paths.data_root() / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def albums() -> list[Path]:
    d = eval_root() / "albums"
    return sorted(p for p in d.iterdir() if p.is_dir() and any(f.suffix.lower() in MEDIA_EXT for f in p.rglob("*"))) \
        if d.exists() else []


def songs() -> list[Path]:
    d = eval_root() / "songs"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in SONG_EXT) if d.exists() else []


def brief_for(album: Path) -> str:
    f = album / "brief.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else get_config().eval.default_brief


def key(album: Path, song: Path) -> str:
    return f"{album.name}__{song.stem}"


def plan_for(s, album: Path, song: Path):
    """The shared plan for (album, song): made once by the director, then reused (render check too)."""
    from montage import pipeline
    from montage.types import EditPlan
    p = eval_root() / "plans" / f"{key(album, song)}.json"
    if p.exists():
        return EditPlan.model_validate_json(p.read_text(encoding="utf-8"))
    plan = pipeline.make_plan(s, brief_for(album))
    paths.atomic_write(p, plan.model_dump_json(indent=1))
    return plan


def director_matrix(st, plan, profile, edl) -> np.ndarray:
    from montage.director.apply import compile_plan
    from montage.eval.metrics import slots_from_edl
    d = compile_plan(plan, profile, st.features, [c.asset_id for c in st.media.candidates])
    return d.director_scores(slots_from_edl(edl, profile))


def main() -> int:
    from montage import pipeline
    from montage.eval import metrics, report
    paths.load_colab_env()
    cfg = get_config().eval
    al, so = albums(), songs()
    if len(al) < 1 or len(so) < 1:
        raise SystemExit(f"need albums in {eval_root() / 'albums'} and songs in {eval_root() / 'songs'}")
    rows = []
    total = len(al) * len(so) * len(cfg.systems) * cfg.seeds
    done = 0
    for album in al:
        for song in so:
            s = pipeline.analyze([str(album)], str(song))
            st = pipeline.state(s)
            plan = plan_for(s, album, song)
            profile = pipeline._profile(st, plan)
            imp = np.asarray(st.importance.final)
            for system in cfg.systems:
                for seed in range(cfg.seeds):
                    edl = pipeline.plan_edit(s, plan, seed, system, run_id=f"eval_{key(album, song)}_{system}_{seed}",
                                             persist=False)
                    paths.atomic_write(eval_root() / "edls" / f"{key(album, song)}__{system}__{seed}.json",
                                       edl.model_dump_json(indent=1))
                    m = metrics.compute(edl, profile, st.features, imp, n_assets=len(st.media.assets),
                                        D=director_matrix(st, plan, profile, edl))
                    rows.append({"album": album.name, "song": song.stem, "system": system, "seed": seed, **m})
                    done += 1
                    jobs.report(done / total, f"{done}/{total} EDLs")
            log.info("%s x %s done (plan source %s)", album.name, song.stem, plan.source)
    paths.atomic_write(eval_root() / "grid.json", json.dumps({"rows": rows, "fallbacks": sorted(set(
        pipeline.component_fallbacks(st)))}, indent=1))
    print(f"grid: {len(rows)} EDLs ({len(al)} albums x {len(so)} songs x {len(cfg.systems)} systems x {cfg.seeds} seeds)",
          flush=True)
    report.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
