"""DEAM MERT shards -> song-level splits (spec 09 section 3).

45 s clips (song ids < 2001): seeded 80/10/10 train/val/test by song. The 58 full songs (ids 2001-2058) form a
second test set ("full"). A song never appears in two splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FULL_SONG_MIN_ID = 2001


@dataclass
class Split:
    feats: np.ndarray      # (n, 13, 768) float16
    y: np.ndarray          # (n, 2) valence, arousal
    song: np.ndarray       # (n,) song id per window
    times: np.ndarray      # (n,)


def split_ids(ids: list[int], seed: int = 0) -> dict[str, list[int]]:
    clips = sorted(i for i in ids if i < FULL_SONG_MIN_ID)
    full = sorted(i for i in ids if i >= FULL_SONG_MIN_ID)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(clips))
    n_val = n_test = int(round(0.1 * len(clips)))
    return {"test": sorted(perm[:n_test]), "val": sorted(perm[n_test:n_test + n_val]),
            "train": sorted(perm[n_test + n_val:]), "full": full}


def load_split(shard_dir: Path, ids: list[int]) -> Split:
    fs, ys, ss, ts = [], [], [], []
    for i in ids:
        with np.load(shard_dir / f"{i}.npz") as z:
            fs.append(z["feats"])
            ys.append(np.stack([z["valence"], z["arousal"]], 1))
            ss.append(np.full(len(z["times"]), i))
            ts.append(z["times"])
    if not fs:
        return Split(np.zeros((0, 13, 768), np.float16), np.zeros((0, 2), np.float32), np.zeros(0, int), np.zeros(0))
    return Split(np.concatenate(fs), np.concatenate(ys).astype(np.float32), np.concatenate(ss), np.concatenate(ts))


def load_all(shard_dir: Path, seed: int = 0) -> tuple[dict[str, list[int]], dict[str, Split]]:
    ids = sorted(int(p.stem) for p in shard_dir.glob("*.npz"))
    sp = split_ids(ids, seed)
    return sp, {k: load_split(shard_dir, v) for k, v in sp.items()}


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12))
