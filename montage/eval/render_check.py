"""Render check (spec 11 section 3): render `full` seed 0 for every (album, song), detect cuts with
PySceneDetect ContentDetector, and match them to the EDL's frame boundaries within +/-1 frame.

    python scripts/run_job.py --name render_check -- python -m montage.eval.render_check

Precision = detected cuts that match an EDL cut (should be ~1.0: no cut where the EDL put none, i.e. no drift).
Recall = EDL cuts that were detected (visually similar neighbouring shots can be missed, so < 1.0 is expected).
"""

from __future__ import annotations

import json

import numpy as np

from montage import jobs, paths
from montage.log import get_logger

log = get_logger("render_check")


def detect_cuts(video: str) -> list[int]:
    from scenedetect import ContentDetector, SceneManager, open_video
    v = open_video(video)
    sm = SceneManager()
    sm.add_detector(ContentDetector(min_scene_len=3))
    sm.detect_scenes(v, show_progress=False)
    return [s[0].get_frames() for s in sm.get_scene_list()][1:]


def match(detected: list[int], expected: list[int], tol: int = 1) -> tuple[float, float]:
    det, exp = np.asarray(detected), np.asarray(expected)
    if len(det) == 0:
        return 1.0, 0.0
    prec = float(np.mean([np.min(np.abs(exp - d)) <= tol for d in det])) if len(exp) else 0.0
    rec = float(np.mean([np.min(np.abs(det - e)) <= tol for e in exp])) if len(exp) else 1.0
    return prec, rec


def main() -> int:
    from montage import pipeline
    from montage.eval.run_grid import albums, eval_root, key, plan_for, songs
    paths.load_colab_env()
    rows = []
    pairs = [(a, s) for a in albums() for s in songs()]
    for i, (album, song) in enumerate(pairs):
        sess = pipeline.analyze([str(album)], str(song))
        plan = plan_for(sess, album, song)
        edl = pipeline.plan_edit(sess, plan, 0, "full")
        video = pipeline.render_edl(edl, preview=True)
        detected = detect_cuts(video)
        expected = [s.frame_start for s in edl.shots[1:]]
        prec, rec = match(detected, expected)
        rows.append({"album": album.name, "song": song.stem, "edl_cuts": len(expected), "detected": len(detected),
                     "precision": prec, "recall": rec, "video": video})
        log.info("%s: precision %.3f recall %.3f (%d detected / %d cuts)", key(album, song), prec, rec,
                 len(detected), len(expected))
        jobs.report((i + 1) / len(pairs), f"{i + 1}/{len(pairs)} rendered")
    summary = {"rows": rows, "precision": float(np.mean([r["precision"] for r in rows])),
               "recall": float(np.mean([r["recall"] for r in rows]))}
    paths.atomic_write(eval_root() / "render_check.json", json.dumps(summary, indent=1))
    print(f"render check: precision {summary['precision']:.3f}, recall {summary['recall']:.3f} over {len(rows)} renders",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
