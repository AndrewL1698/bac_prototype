"""MERT window features for DEAM (spec 09 section 2). Resumable: one shard per song, finished songs skipped.

    python scripts/run_job.py --name mert_feats -- python -m training.emotion.extract_mert

Per annotated time t: a 5 s window CENTERED on t (montage.music.emotion.make_windows, the exact inference
code), MERT hidden states mean-pooled over time -> (13, 768) fp16. Shard:
DATA_ROOT/datasets/deam_mert/<song_id>.npz with times, feats (n, 13, 768), valence (n,), arousal (n,).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from montage import cache, jobs, paths
from montage.config import get_config
from montage.log import get_logger
from montage.music.emotion import make_windows, mert_window_features
from montage.music.load import load_audio
from training.common import datasets_dir
from training.emotion.download_deam import annotations_path, audio_dir

log = get_logger("mert_feats")


def shard_dir() -> Path:
    return datasets_dir("deam_mert")


def song_items() -> list[tuple[str, Path, np.ndarray, np.ndarray, np.ndarray]]:
    data = json.loads(annotations_path().read_text())
    start, step = data["start_s"], data["step_s"]
    out = []
    for sid, s in sorted(data["songs"].items(), key=lambda kv: int(kv[0])):
        mp3 = audio_dir() / f"{sid}.mp3"
        if not mp3.exists():
            continue
        a, v = np.asarray(s["arousal"], np.float32), np.asarray(s["valence"], np.float32)
        n = min(len(a), len(v))
        out.append((sid, mp3, start + step * np.arange(n, dtype=np.float64), v[:n], a[:n]))
    return out


def main() -> int:
    paths.load_colab_env()
    sr = get_config().music.mert_sr
    items = song_items()
    todo = [it for it in items if not (shard_dir() / f"{it[0]}.npz").exists()]
    log.info("%d songs, %d to extract", len(items), len(todo))

    def load(it):
        sid, mp3, times, v, a = it
        y = load_audio(mp3, sr)
        keep = times <= len(y) / sr  # drop annotation times past the audio end
        return sid, y, times[keep], v[keep], a[keep]

    done = len(items) - len(todo)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for sid, y, times, v, a in ex.map(load, todo):  # decode ahead while the GPU works
            if len(times) == 0:
                log.warning("song %s: no annotated times inside the audio", sid)
                continue
            feats = mert_window_features(make_windows(y, times, sr=sr))
            cache.save_npz(shard_dir() / f"{sid}.npz", times=times.astype(np.float32), feats=feats,
                           valence=v.astype(np.float32), arousal=a.astype(np.float32))
            done += 1
            jobs.report(done / len(items), f"{done}/{len(items)} songs")
    n = len(list(shard_dir().glob("*.npz")))
    print(f"MERT shards: {n}/{len(items)} songs in {shard_dir()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
