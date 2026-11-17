"""Calibrate the song-pick album mood (`songpick.vibe.album_mood_center` and `album_mood_scale`, phase 12.8).

    python scripts/calibrate_album_mood.py [--album DIR ...] [--no-cufed]

Raw album mood = the mean over an album's assets of 100 * (cos(high pole) - cos(low pole)) (`features.mood.mood_raw`,
pooled as in `songpick.vibe.album_from_features`). Albums: the CUFED event albums (`cufed_photos.npz`; the CLIP
embedding is the first 768 columns of X, see `importance.ranker.ranker_features`) plus our own albums (default:
demo/* and eval/albums/*, through the editor's cached feature path). Suggested center = the mean over albums,
scale = the std over albums, so one album in three lands beyond tanh(1) = 0.76. Prints the suggestion and every
event type and own album under it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import paths  # noqa: E402
from montage.app.demo_mode import MEDIA_EXT  # noqa: E402
from montage.app.demo_mode import albums as demo_albums  # noqa: E402
from montage.features.mood import mood_raw  # noqa: E402
from montage.songpick.vibe import album_vibe, mood_word  # noqa: E402
from training.common import datasets_dir  # noqa: E402

Row = tuple[str, str, float, float]  # (album, group label, raw valence, raw arousal)


def cufed_albums() -> list[Row]:
    z = np.load(datasets_dir("importance") / "cufed_photos.npz")
    a_raw, v_raw = mood_raw(z["X"][:, :768])
    return [(str(g), str(z["event"][m][0]), float(v_raw[m].mean()), float(a_raw[m].mean()))
            for g in np.unique(z["group"]) for m in [z["group"] == g]]


def own_albums(dirs: list[Path]) -> list[Row]:
    rows = []
    for d in dirs:
        v, a = album_vibe(sorted(str(f) for f in d.rglob("*") if f.suffix.lower() in MEDIA_EXT)).raw
        rows.append((d.name, f"own: {d.name}", v, a))
    return rows


def main() -> None:
    paths.load_colab_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--album", type=Path, action="append", help="album folder (repeatable; default demo + eval)")
    ap.add_argument("--no-cufed", action="store_true")
    args = ap.parse_args()
    eval_root = paths.data_root() / "eval" / "albums"
    dirs = args.album or demo_albums() + (sorted(p for p in eval_root.iterdir() if p.is_dir())
                                         if eval_root.exists() else [])
    rows = ([] if args.no_cufed else cufed_albums()) + own_albums(dirs)
    v, a = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    cv, ca, sv, sa = (round(float(x), 2) for x in (v.mean(), a.mean(), v.std(), a.std()))
    print(f"{len(rows)} albums")
    print(f"album_mood_center: {{valence: {cv}, arousal: {ca}}}")
    print(f"album_mood_scale: {{valence: {sv}, arousal: {sa}}}\n")
    print("| Albums | n | Raw valence | Raw arousal | Valence | Arousal | Mood word |\n|---|---|---|---|---|---|---|")
    groups = sorted({r[1] for r in rows}, key=lambda g: (g.startswith("own"), g))
    for g in groups:
        rv = float(np.mean([r[2] for r in rows if r[1] == g]))
        ra = float(np.mean([r[3] for r in rows if r[1] == g]))
        mv, ma = math.tanh((rv - cv) / sv), math.tanh((ra - ca) / sa)
        n = sum(r[1] == g for r in rows)
        print(f"| {g} | {n} | {rv:+.2f} | {ra:+.2f} | {mv:+.2f} | {ma:+.2f} | {mood_word(mv, ma) or 'neutral'} |")


if __name__ == "__main__":
    main()
