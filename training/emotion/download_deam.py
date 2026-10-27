"""DEAM (1,802 songs, dynamic valence/arousal at 2 Hz from 15 s) into DATA_ROOT/datasets/deam/.

    python scripts/run_job.py --name deam_dl -- python -m training.emotion.download_deam

The official host (cvml.unige.ch) was unreachable on 2026-10-27, so this uses public Hugging Face mirrors:
audio from herrjyj/herrjyj-deam-assets (DEAM_audio.zip), dynamic annotations from
Rehead/DEAM_continuous_annotations (only the annotation columns are read; the parquet audio is skipped).
Idempotent: finished steps are skipped.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from montage import jobs, paths
from montage.log import get_logger
from training.common import datasets_dir, download

log = get_logger("deam")
AUDIO_URL = "https://huggingface.co/datasets/herrjyj/herrjyj-deam-assets/resolve/main/DEAM_audio.zip"
ANNOT_REPO = "datasets/Rehead/DEAM_continuous_annotations"
ANNOT_START_S = 15.0
ANNOT_STEP_S = 0.5
N_SONGS = 1802


def audio_dir() -> Path:
    return datasets_dir("deam") / "audio"


def annotations_path() -> Path:
    return datasets_dir("deam") / "annotations.json"


def fetch_audio() -> int:
    out = audio_dir()
    have = len(list(out.glob("*.mp3"))) if out.exists() else 0
    if have >= N_SONGS:
        return have
    z = download(AUDIO_URL, paths.scratch_dir("downloads") / "DEAM_audio.zip", 0.0, 0.6, "DEAM audio")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        members = [m for m in zf.infolist() if m.filename.lower().endswith(".mp3")]
        log.info("audio zip: %d mp3 files, top folders %s", len(members),
                 Counter(m.filename.split("/")[0] for m in members).most_common(3))
        for i, m in enumerate(members):
            target = out / Path(m.filename).name
            if not target.exists() or target.stat().st_size != m.file_size:
                paths.atomic_write(target, zf.read(m))
            if i % 100 == 0:
                jobs.report(0.6 + 0.2 * i / len(members), f"extracting audio {i}/{len(members)}")
    return len(list(out.glob("*.mp3")))


def fetch_annotations() -> dict:
    p = annotations_path()
    if p.exists():
        return json.loads(p.read_text())
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{ANNOT_REPO}/data/*.parquet"))
    log.info("annotation parquet files: %d", len(files))
    ann: dict[str, dict] = {}
    for i, f in enumerate(files):
        with fs.open(f, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["song_id", "arousal_annotations", "valence_annotations"])
        for sid, a, v in zip(t["song_id"].to_pylist(), t["arousal_annotations"].to_pylist(),
                             t["valence_annotations"].to_pylist()):
            ann[str(sid)] = {"arousal": [float(x) for x in a], "valence": [float(x) for x in v]}
        jobs.report(0.8 + 0.2 * (i + 1) / len(files), f"annotations {i + 1}/{len(files)} files")
    paths.atomic_write(p, json.dumps({"start_s": ANNOT_START_S, "step_s": ANNOT_STEP_S, "songs": ann}))
    return json.loads(p.read_text())


def main() -> int:
    paths.load_colab_env()
    n_audio = fetch_audio()
    data = fetch_annotations()
    songs = data["songs"]
    lens = np.array([len(s["arousal"]) for s in songs.values()])
    vals = np.concatenate([np.r_[s["arousal"], s["valence"]] for s in songs.values()])
    ids_audio = {p.stem for p in audio_dir().glob("*.mp3")}
    both = ids_audio & set(songs)
    print(f"DEAM: {n_audio} audio files, {len(songs)} annotated songs, {len(both)} with both; annotation length "
          f"min/median/max {lens.min()}/{int(np.median(lens))}/{lens.max()} (2 Hz from {data['start_s']} s); "
          f"values in [{vals.min():.2f}, {vals.max():.2f}]", flush=True)
    if len(both) < 1700:
        raise SystemExit("fewer than 1700 songs have both audio and annotations; stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
