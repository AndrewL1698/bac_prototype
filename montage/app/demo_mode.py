"""Demo mode: warm every cache, render backups, pre-render the vote pair, time a warm generate.

    python scripts/run_job.py --name demo_prep -- python -m montage.app.demo_mode --prepare

Albums: DATA_ROOT/demo/<album>/ (optional brief.txt, optional song.txt naming a file in demo/songs/).
Outputs: demo/backups/<album>.mp4 (1080p) and <album>_naive.mp4, demo/vote/<album>_{full,naive}.mp4 + pairs.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from montage import jobs, paths
from montage.config import get_config
from montage.log import get_logger

log = get_logger("demo")


MEDIA_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".mp4", ".mov", ".m4v", ".avi", ".mkv"}
SONG_EXT = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}
NOT_ALBUMS = {"songs", "songs_full", "backups", "vote"}


def demo_root() -> Path:
    return paths.data_root() / "demo"


def demo_albums() -> list[str]:
    """Folders in demo/ holding media (possibly in subfolders), except the reserved ones."""
    d = demo_root()
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name not in NOT_ALBUMS
                  and any(f.suffix.lower() in MEDIA_EXT for f in p.rglob("*")))


def demo_songs() -> list[str]:
    d = demo_root() / "songs"
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in SONG_EXT) if d.exists() else []


def albums() -> list[Path]:
    return [demo_root() / a for a in demo_albums()]


def songs() -> list[Path]:
    return [demo_root() / "songs" / s for s in demo_songs()]


def album_brief(a: Path) -> str:
    f = a / "brief.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else get_config().eval.default_brief


def album_song(a: Path, all_songs: list[Path]) -> Path | None:
    f = a / "song.txt"
    if f.exists():
        p = demo_root() / "songs" / f.read_text(encoding="utf-8").strip()
        if p.exists():
            return p
    return all_songs[0] if all_songs else None


def vote_pairs_path() -> Path:
    return demo_root() / "vote" / "pairs.json"


def load_pairs() -> list[dict]:
    p = vote_pairs_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def prepare() -> dict:
    from montage import pipeline
    from montage.music.profile import analyze_song
    report: dict = {"albums": {}, "songs": []}
    al, so = albums(), songs()
    if not al or not so:
        raise SystemExit(f"Need at least one album folder and one song in {demo_root()} (found {len(al)} albums, "
                         f"{len(so)} songs).")
    steps = len(so) + 3 * len(al) + 1
    done = 0

    def tick(msg: str) -> None:
        nonlocal done
        done += 1
        jobs.report(done / steps, msg, force=True)
        log.info(msg)

    for s in so:
        analyze_song(s)
        report["songs"].append(s.name)
        tick(f"analyzed song {s.name}")
    pairs = []
    backups = demo_root() / "backups"
    vote = demo_root() / "vote"
    backups.mkdir(parents=True, exist_ok=True)
    vote.mkdir(parents=True, exist_ok=True)
    from montage.ingest.assets import ingest
    from montage.render.proxy import ensure_proxies
    for a in al:
        n = ensure_proxies(ingest([a]))
        log.info("%s: %d HDR render proxies ready", a.name, n)
        report.setdefault("proxies", {})[a.name] = n
    for a in al:
        song = album_song(a, so)
        res = pipeline.generate([str(a)], str(song), album_brief(a), seed=0)
        tick(f"{a.name}: generated preview")
        x = pipeline.export(res.run_id)
        shutil.copyfile(x, backups / f"{a.name}.mp4")
        shutil.copyfile(res.video_path, vote / f"{a.name}_full.mp4")
        s = pipeline.session_for_run(res.run_id)
        edl = pipeline.plan_edit(s, s.plan_history[-1], seed=0, system="naive")
        naive = pipeline.render_edl(edl, preview=True)
        shutil.copyfile(naive, vote / f"{a.name}_naive.mp4")
        tick(f"{a.name}: backup + naive cut")
        naive_x = pipeline.render_edl(edl, preview=False)
        shutil.copyfile(naive_x, backups / f"{a.name}_naive.mp4")
        pairs.append({"pair_id": a.name, "a": f"{a.name}_full.mp4", "a_system": "full",
                      "b": f"{a.name}_naive.mp4", "b_system": "naive"})
        report["albums"][a.name] = {"song": song.name, "run_id": res.run_id, "fallbacks": res.fallbacks}
        tick(f"{a.name}: naive backup")
    paths.atomic_write(vote_pairs_path(), json.dumps(pairs, indent=1))
    a = al[0]
    t0 = time.time()
    pipeline.generate([str(a)], str(album_song(a, so)), album_brief(a), seed=1)
    report["warm_generate_s"] = round(time.time() - t0, 1)
    tick("timed warm generate")
    return report


def checklist(report: dict | None = None) -> list[str]:
    import torch
    w = paths.weights_dir()
    cfg = get_config()
    url = paths.jobs_dir() / "app.url"
    lines = [
        f"[{'x' if torch.cuda.is_available() else ' '}] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        f"[{'x' if os.environ.get('ANTHROPIC_API_KEY') else ' '}] Claude API key (director: {'Claude' if os.environ.get('ANTHROPIC_API_KEY') else 'rules'})",
        f"[{'x' if (w / cfg.music.emotion.head_weights).exists() else ' '}] emotion head weights (else heuristic arousal)",
        f"[{'x' if (w / cfg.importance.ranker_weights).exists() else ' '}] importance ranker weights (else heuristic)",
        f"[{'x' if (w / 'beat_this-final0.ckpt').exists() else ' '}] Beat This! checkpoint cached on Drive (else librosa beats)",
    ]
    for a in albums():
        b = demo_root() / "backups" / f"{a.name}.mp4"
        lines.append(f"[{'x' if b.exists() else ' '}] backup video for {a.name}")
    lines.append(f"[{'x' if load_pairs() else ' '}] vote pair(s) for the Rate tab")
    if report and "warm_generate_s" in report:
        ok = report["warm_generate_s"] < 30
        lines.append(f"[{'x' if ok else ' '}] warm generate {report['warm_generate_s']} s (target < 30 s)")
    lines.append(f"[{'x' if url.exists() else ' '}] app URL in {url}")
    return lines


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args(argv)
    report = prepare() if args.prepare else None
    if report:
        print(json.dumps(report, indent=1))
    print("\n".join(checklist(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
