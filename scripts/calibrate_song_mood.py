"""Calibrate the song-pick song mood (`songpick.sentiment.song_mood_center` and `song_mood_scale`, phase 12.9).

    python scripts/calibrate_song_mood.py [--song FILE ...] [--no-deam]

Raw song mood = the mean MERT emotion-head (valence, arousal) over a song (`songpick.sentiment.raw_mood` uses the
default excerpt). Songs: the DEAM songs the head never trained on (test clips + full songs, the cached MERT shards
from phase 9B) plus our own songs (default: demo/songs and eval/songs, through the editor's cached song analysis).
Suggested center = the mean over songs, scale = the std, as `calibrate_album_mood.py` does for the photos. Prints the
suggestion and every own song under it. Needs the MERT head (weights/emotion_head.pt); run on Colab.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import paths  # noqa: E402
from montage.app.demo_mode import SONG_EXT  # noqa: E402
from montage.app.demo_mode import songs as demo_songs  # noqa: E402
from montage.config import get_config  # noqa: E402
from montage.songpick.sentiment import raw_mood  # noqa: E402
from montage.songpick.vibe import mood_word  # noqa: E402

Row = tuple[str, str, float, float]  # (song, group label, raw valence, raw arousal)


def deam_songs() -> list[Row]:
    import torch

    from montage.music.emotion import _load_head
    from training.emotion.dataset import load_split, split_ids
    from training.emotion.extract_mert import shard_dir
    ids = sorted(int(p.stem) for p in shard_dir().glob("*.npz"))
    sp = split_ids(ids, get_config().seed)   # the split train.py used, so these songs are held out
    head, rows = _load_head(), []
    for group in ("test", "full"):
        for i in sp[group]:
            s = load_split(shard_dir(), [i])
            with torch.no_grad():
                va = head(torch.from_numpy(s.feats.astype(np.float32))).numpy()
            rows.append((str(i), f"DEAM {group}", float(va[:, 0].mean()), float(va[:, 1].mean())))
    return rows


def own_songs(files: list[Path]) -> list[Row]:
    rows = []
    for f in files:
        va = raw_mood(f)
        if va is None:
            sys.exit("the MERT emotion head is not available (weights/emotion_head.pt); nothing to calibrate")
        rows.append((f.name, f"own: {f.parent.name}", va[0], va[1]))
    return rows


def main() -> None:
    paths.load_colab_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--song", type=Path, action="append", help="song file (repeatable; default demo + eval songs)")
    ap.add_argument("--no-deam", action="store_true")
    args = ap.parse_args()
    eval_dir = paths.data_root() / "eval" / "songs"
    files = args.song or demo_songs() + (sorted(p for p in eval_dir.iterdir() if p.suffix.lower() in SONG_EXT)
                                         if eval_dir.exists() else [])
    deam = [] if args.no_deam else deam_songs()
    own = own_songs(files)
    fit = deam or own
    v, a = np.array([r[2] for r in fit]), np.array([r[3] for r in fit])
    cv, ca, sv, sa = (round(float(x), 2) for x in (v.mean(), a.mean(), v.std(), a.std()))
    print(f"fit on {len(fit)} songs ({'DEAM test + full' if deam else 'own songs'})")
    print(f"song_mood_center: {{valence: {cv}, arousal: {ca}}}")
    print(f"song_mood_scale: {{valence: {sv}, arousal: {sa}}}\n")
    print("| Song | Raw valence | Raw arousal | Valence | Arousal | Mood word |\n|---|---|---|---|---|---|")
    for g in sorted({r[1] for r in deam}):
        rv = float(np.mean([r[2] for r in deam if r[1] == g]))
        ra = float(np.mean([r[3] for r in deam if r[1] == g]))
        print(f"| {g} (mean of {sum(r[1] == g for r in deam)}) | {rv:+.2f} | {ra:+.2f} | | | |")
    for name, _, rv, ra in own:
        mv, ma = math.tanh((rv - cv) / max(sv, 1e-6)), math.tanh((ra - ca) / max(sa, 1e-6))
        print(f"| {name} | {rv:+.2f} | {ra:+.2f} | {mv:+.2f} | {ma:+.2f} | {mood_word(mv, ma) or 'neutral'} |")


if __name__ == "__main__":
    main()
