"""PMEmo 2019 (794 pop choruses; dynamic valence/arousal at 2 Hz for 767) into DATA_ROOT/datasets/pmemo/.

    python scripts/run_job.py --name pmemo_dl -- python -m training.emotion.download_pmemo

The official links are Google Drive / Baidu, so this uses the public Hugging Face mirror csdhbg/PMEmo2019 (the full
release zip; layout and annotation files checked against the official release on 2026-11-11). Only the chorus audio
and dynamic_annotations.csv are kept. The 2019 release starts annotations at 15.5 s (frameTime is seconds into the
chorus clip, step 0.5 s). Values are in [0, 1] and are mapped linearly to [-1, 1] like DEAM. Songs with fewer than
MIN_FRAMES annotated frames, or with gaps in their frame times, are dropped. Idempotent.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from montage import jobs, paths
from montage.log import get_logger
from training.common import datasets_dir, download

log = get_logger("pmemo")
ZIP_URL = "https://huggingface.co/datasets/csdhbg/PMEmo2019/resolve/main/PMEmo2019.zip"
START_S = 15.5         # the 2019 release drops the first 15 s
STEP_S = 0.5
MIN_FRAMES = 10        # 5 s: shorter curves give meaningless within-song CCC


def audio_dir() -> Path:
    return datasets_dir("pmemo") / "audio"


def annotations_path() -> Path:
    return datasets_dir("pmemo") / "annotations.json"


def parse_annotations(csv_bytes: bytes) -> dict:
    """dynamic_annotations.csv -> {"start_s", "step_s", "songs": {id: {"arousal", "valence"}}} in [-1, 1], the
    DEAM annotations.json format. Kept songs start at START_S with contiguous frames."""
    import pandas as pd
    d = pd.read_csv(io.BytesIO(csv_bytes))
    songs = {}
    for sid, g in d.sort_values(["musicId", "frameTime"]).groupby("musicId"):
        t = g["frameTime"].to_numpy()
        if len(t) < MIN_FRAMES or t[0] != START_S or not np.allclose(np.diff(t), STEP_S):
            continue
        songs[str(sid)] = {"arousal": (2 * g["Arousal(mean)"] - 1).round(5).tolist(),
                           "valence": (2 * g["Valence(mean)"] - 1).round(5).tolist()}
    return {"start_s": START_S, "step_s": STEP_S, "songs": songs}


def main() -> int:
    paths.load_colab_env()
    z = zipfile.ZipFile(download(ZIP_URL, paths.scratch_dir("downloads") / "PMEmo2019.zip", 0.0, 0.6, "PMEmo"))
    data = parse_annotations(z.read("PMEmo2019/annotations/dynamic_annotations.csv"))
    paths.atomic_write(annotations_path(), json.dumps(data))
    members = [m for m in z.infolist() if m.filename.startswith("PMEmo2019/chorus/") and m.filename.endswith(".mp3")
               and Path(m.filename).stem in data["songs"]]
    out = audio_dir()
    for i, m in enumerate(members):
        target = out / Path(m.filename).name
        if not target.exists() or target.stat().st_size != m.file_size:
            paths.atomic_write(target, z.read(m))
        if i % 50 == 0:
            jobs.report(0.6 + 0.4 * i / len(members), f"audio {i}/{len(members)}")
    lens = np.array([len(s["arousal"]) for s in data["songs"].values()])
    print(f"PMEmo: {len(data['songs'])} annotated songs kept, {len(list(out.glob('*.mp3')))} audio files; "
          f"annotation length min/median/max {lens.min()}/{int(np.median(lens))}/{lens.max()} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
