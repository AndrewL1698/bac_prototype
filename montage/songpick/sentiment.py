"""Song sentiment match from audio (phase 12.9; ROADMAP F1 piece 5): the song's own mood vs the photos' mood.

Runs after the stage A tag ranking, on its best `songpick.sentiment.max_songs` songs, only when the user opted in to
the YouTube fetch. Per song: fetch the audio (`youtube.find_and_fetch`, cached by video id) -> the editor's song
analysis (`music.profile.analyze_song`, cached by content hash, so Generate is warm for the chosen song) -> mean MERT
valence/arousal over the default excerpt -> centered and squashed like the album side (`song_mood_center/scale`).

vibe_sent = closeness of that mood to the photos' mood (`vibe.mood_match`). For the songs with audio it replaces the
tag-lexicon mood term: vibe = weighted mean of the CLIP tag match (`vibe.w_clip`) and vibe_sent (`w_sent`), each
rank-normalized over those songs. A song whose audio could not be fetched or analyzed keeps its stage A vibe.
Logs carry counts and error types only, never song names.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from montage.config import get_config
from montage.log import get_logger
from montage.songpick import vibe
from montage.types import TrackCandidate

log = get_logger("songpick.sentiment")

Mood = tuple[float, float]                          # (valence, arousal)
Fetch = Callable[[str, str, int | None], tuple[Path, dict]]


def raw_mood(path: str | Path) -> Mood | None:
    """Mean head (valence, arousal) over the song's default excerpt, before center/scale. None without the MERT head
    (the heuristic emotion has no valence)."""
    from montage.music.excerpt import choose_excerpt
    from montage.music.profile import analyze_song
    fa = analyze_song(path)
    if fa.emotion_source == "heuristic":
        return None
    ex = choose_excerpt(fa)
    grid = fa.a("grid")
    m = (grid >= ex.song_start) & (grid < ex.song_end)
    if not m.any():
        m = np.ones_like(grid, bool)
    return float(fa.a("valence")[m].mean()), float(fa.a("arousal")[m].mean())


def standardize(raw: Mood) -> Mood:
    cfg = get_config().songpick.sentiment
    c, s = cfg.song_mood_center, cfg.song_mood_scale
    return (math.tanh((raw[0] - c["valence"]) / s["valence"]), math.tanh((raw[1] - c["arousal"]) / s["arousal"]))


def song_mood(path: str | Path) -> Mood | None:
    raw = raw_mood(path)
    return standardize(raw) if raw is not None else None


def _default_fetch(track: str, artist: str, duration_ms: int | None) -> tuple[Path, dict]:
    from montage.songpick import youtube
    return youtube.find_and_fetch(track, artist, duration_ms)


def _strip_vibe_line(c: TrackCandidate, album: vibe.AlbumVibe) -> None:
    """Remove the stage A vibe sentence from the reason (it is rewritten with the audio mood)."""
    for good in (True, False):
        line = vibe.reason_line(c, album, good)
        if line and c.reason.startswith(line):
            c.reason = c.reason[len(line):].lstrip()
            return


class Scorer:
    """Callable hook for `pick.suggest(sentiment_scorer=...)`: re-scores the head of the stage A ranking in place.
    Returns True if any song got an audio mood."""

    def __init__(self, album: vibe.AlbumVibe, fetch: Fetch | None = None,
                 mood: Callable[[Path], Mood | None] | None = None):
        self.album, self.fetch, self.mood = album, fetch or _default_fetch, mood or song_mood
        self.n_heard = 0
        self.n_tried = 0

    def _moods(self, cands: list[TrackCandidate]) -> dict[str, Mood]:
        cfg = get_config().songpick.sentiment
        start = time.monotonic()
        out: dict[str, Mood] = {}
        errors: dict[str, int] = {}
        pool = ThreadPoolExecutor(max(1, cfg.workers))
        futs = {pool.submit(self.fetch, c.track, c.artist, c.duration_ms): c for c in cands}
        try:
            for f in as_completed(futs):
                if time.monotonic() - start > cfg.budget_s:
                    errors["budget"] = errors.get("budget", 0) + 1
                    break
                try:
                    va = self.mood(f.result()[0])   # analysis stays on this thread: one song at a time on the GPU
                except Exception as e:  # noqa: BLE001  one song failing must not stop the others
                    errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1
                    continue
                if va is not None:
                    out[futs[f].key] = va
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        log.info("songpick: audio mood for %d of %d songs (%.0f s)%s", len(out), len(cands),
                 time.monotonic() - start, f"; failures {errors}" if errors else "")
        return out

    def __call__(self, cands: list[TrackCandidate]) -> bool:
        cfg = get_config().songpick
        cands = cands[: cfg.sentiment.max_songs]
        self.n_tried += len(cands)
        moods = self._moods(cands)
        if not moods:
            return False
        self.n_heard += len(moods)
        heard = [c for c in cands if c.key in moods]
        sent = vibe._ranks({c.key: vibe.mood_match(moods[c.key], self.album) for c in heard})
        clip = vibe._ranks({c.key: c.vibe_clip for c in heard if c.vibe_clip is not None})
        for c in heard:
            if c.tags:
                _strip_vibe_line(c, self.album)
            c.valence, c.arousal = (round(x, 4) for x in moods[c.key])
            c.mood_source = "audio"
            c.vibe_sent, c.vibe_clip = sent[c.key], clip.get(c.key)
            terms = [(cfg.vibe.w_clip, c.vibe_clip), (cfg.sentiment.w_sent, c.vibe_sent)]
            terms = [(w, x) for w, x in terms if x is not None and w > 0]
            if terms:
                    c.vibe = round(sum(w * x for w, x in terms) / sum(w for w, _ in terms), 6)
        med = float(np.median([c.vibe for c in heard if c.vibe is not None] or [0.5]))
        for c in cands:
            if c.vibe is None:   # no tags and no audio: the median, as stage A does for untagged songs
                c.vibe = round(med, 6)
        for c in heard:
            line = vibe.reason_line(c, self.album, c.vibe >= med)
            c.reason = f"{line} {c.reason}" if line else c.reason
        return True
