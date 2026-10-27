"""Map SumMe/TVSum h5 entries to the raw videos and read per-frame importance labels (spec 10 section 2).

h5 entries give `gtscore` (importance per sampled frame), `picks` (frame indices), `n_frames`, and sometimes
`video_name`. Only labels and frame indices are used; the h5 GoogLeNet features are ignored.
Mapping: by name when `video_name` exists, otherwise by frame count within the dataset (ties by duration).
The full mapping is written to DATA_ROOT/datasets/importance/mapping.json; every entry must map one to one.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from montage import paths
from montage.config import get_config
from montage.log import get_logger
from training.common import datasets_dir
from training.importance.download import VIDEO_EXT, raw_dir

log = get_logger("vsum_labels")


@dataclass
class VideoLabels:
    key: str              # e.g. "summe/video_3"
    dataset: str          # "summe" | "tvsum"
    video: str            # path to the raw video
    fps: float
    pick_times: np.ndarray
    score: np.ndarray     # min-max normalized gtscore in [0, 1]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _count_frames(path: Path) -> tuple[int, float, float]:
    """(frame count, fps, duration) via ffprobe -count_packets (demux only, fast)."""
    c = get_config().render
    r = subprocess.run([c.ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets,avg_frame_rate,r_frame_rate:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    st = (d.get("streams") or [{}])[0]

    def rate(s):
        if not s or s in ("0/0", "0"):
            return 0.0
        n, _, m = s.partition("/")
        return float(n) / float(m or 1)

    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
    return int(st.get("nb_read_packets") or 0), fps, float((d.get("format") or {}).get("duration") or 0)


def dataset_of(path: Path) -> str:
    s = str(path).lower()
    return "summe" if "summe" in s else "tvsum" if "tvsum" in s else "?"


def find_files() -> tuple[dict[str, Path], dict[str, list[Path]]]:
    root = raw_dir()
    h5 = {dataset_of(p): p for p in root.rglob("*.h5") if dataset_of(p) in ("summe", "tvsum")}
    vids: dict[str, list[Path]] = {"summe": [], "tvsum": []}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT and dataset_of(p) in vids:
            vids[dataset_of(p)].append(p)
    return h5, vids


def build_mapping() -> dict:
    import h5py
    out_path = datasets_dir("importance") / "mapping.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    h5_files, vids = find_files()
    log.info("h5 files: %s; videos: %s", {k: str(v.name) for k, v in h5_files.items()},
             {k: len(v) for k, v in vids.items()})
    mapping: dict[str, dict] = {}
    for ds, h5p in sorted(h5_files.items()):
        probes = {str(v): _count_frames(v) for v in vids[ds]}
        by_name = {_norm(Path(v).stem): v for v in probes}
        used: set[str] = set()
        with h5py.File(h5p, "r") as f:
            entries = sorted(f.keys(), key=lambda k: int(re.sub(r"\D", "", k) or 0))
            for k in entries:
                g = f[k]
                n_frames = int(np.array(g["n_frames"]))
                name = g["video_name"][()].decode() if "video_name" in g else None
                match = by_name.get(_norm(name)) if name else None
                how = "name"
                if match is None:
                    cands = [v for v, (n, _, _) in probes.items() if v not in used and abs(n - n_frames) <= 2]
                    if len(cands) > 1:  # break ties by expected duration (n_frames / fps)
                        cands.sort(key=lambda v: abs(probes[v][2] - n_frames / max(probes[v][1], 1e-6)))
                    match, how = (cands[0], "frames") if cands else (None, "none")
                if match is None or match in used:
                    raise SystemExit(f"cannot map {ds}/{k} (n_frames={n_frames}, name={name}) one to one")
                used.add(match)
                n, fps, dur = probes[match]
                mapping[f"{ds}/{k}"] = {"dataset": ds, "h5": str(h5p), "entry": k, "video": match, "how": how,
                                        "n_frames_h5": n_frames, "n_frames_video": n, "fps": fps, "duration": dur}
    counts = {ds: sum(1 for m in mapping.values() if m["dataset"] == ds) for ds in ("summe", "tvsum")}
    log.info("mapped %s", counts)
    paths.atomic_write(out_path, json.dumps(mapping, indent=1))
    return mapping


def load_labels(mapping: dict) -> list[VideoLabels]:
    import h5py
    out = []
    for key, m in sorted(mapping.items()):
        with h5py.File(m["h5"], "r") as f:
            g = f[m["entry"]]
            gts = np.asarray(g["gtscore"], float)
            picks = np.asarray(g["picks"], float)
        n = min(len(gts), len(picks))
        s = gts[:n]
        s = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.zeros(n)
        out.append(VideoLabels(key=key, dataset=m["dataset"], video=m["video"], fps=m["fps"],
                               pick_times=picks[:n] / max(m["fps"], 1e-6), score=s))
    return out
