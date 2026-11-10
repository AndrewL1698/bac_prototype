"""Song vibe match, stage A (phase 12.8): the song's tags vs the photos, before any audio exists.

Two terms, each in [0, 1], combined as a weighted mean of the ones available (`songpick.vibe.w_clip`, `w_mood`):
- vibe_clip: CLIP text embedding of the song's tags (prompt ensemble `vibe.templates`, weighted by tag weight)
  vs the album's pooled CLIP image embedding; min-max normalized over the scored songs.
- vibe_mood: 1 - distance between the song's valence/arousal (tag lexicon `configs/tag_mood.tsv`) and the photos'
  absolute mood (mean raw CLIP mood-pole scores, `features/mood.py`), scaled to [0, 1].
Songs without tags get the median vibe of the scored songs, so a missing tag neither helps nor hurts.

The album side reuses the editor's own feature path (`candidates.build_media` + `features.extract`, cached by
content hash), so Generate is warm after Suggest. `score` is pure: tests pass a fake `encode_text`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from montage.config import get_config, repo_path
from montage.log import get_logger
from montage.types import TrackCandidate

log = get_logger("songpick.vibe")

Tags = list[tuple[str, float]]          # (tag, weight 0-100), best first
EncodeText = Callable[[Sequence[str]], np.ndarray]
MAX_VA_DIST = 2.0 * math.sqrt(2.0)       # farthest two points in [-1, 1]^2


@dataclass
class AlbumVibe:
    emb: np.ndarray                      # (768,) pooled CLIP image embedding, L2-normalized
    valence: float                       # [-1, 1] absolute photo mood
    arousal: float
    n_assets: int
    taken_at: list[datetime | None] = field(default_factory=list)


def norm_tag(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", str(t).lower())).strip()


@lru_cache(maxsize=4)
def _lexicon(path: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for line in repo_path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if not line.strip() or line.startswith("#") or len(parts) != 3 or parts[0] == "tag":
            continue
        out[norm_tag(parts[0])] = (float(parts[1]), float(parts[2]))
    return out


def lexicon() -> dict[str, tuple[float, float]]:
    return _lexicon(get_config().songpick.vibe.tag_mood_file)


# ---------- album side ----------

def album_from_features(emb: np.ndarray, asset_ids: Sequence[str], arousal_raw: np.ndarray, valence_raw: np.ndarray,
                        taken_at: list[datetime | None] | None = None) -> AlbumVibe:
    """Pool candidate rows per asset first (a long video's many windows count once), then over assets."""
    cfg = get_config().songpick.vibe
    order = list(dict.fromkeys(asset_ids))
    idx = {a: [i for i, x in enumerate(asset_ids) if x == a] for a in order}
    per_emb = np.stack([np.asarray(emb)[idx[a]].mean(0) for a in order])
    pooled = per_emb.mean(0)
    pooled = (pooled / max(float(np.linalg.norm(pooled)), 1e-8)).astype(np.float32)
    a_raw = float(np.mean([np.asarray(arousal_raw)[idx[a]].mean() for a in order]))
    v_raw = float(np.mean([np.asarray(valence_raw)[idx[a]].mean() for a in order]))
    s = cfg.album_mood_scale
    return AlbumVibe(emb=pooled, valence=math.tanh(v_raw / s["valence"]), arousal=math.tanh(a_raw / s["arousal"]),
                     n_assets=len(order), taken_at=list(taken_at or []))


def album_vibe(media_paths: list[str], progress: Callable[[float, str], None] | None = None) -> AlbumVibe:
    """Ingest + features (the editor's cached path) -> AlbumVibe. Raises UserError on bad input."""
    from montage.candidates import build_media
    from montage.features.extract import extract
    from montage.log import stage
    with stage("songpick_ingest"):
        ms = build_media(media_paths, (lambda f, m: progress(0.4 * f, m)) if progress else None)
    with stage("songpick_features"):
        feats = extract(ms.assets, ms.candidates, ms.frames,
                        (lambda f, m: progress(0.4 + 0.5 * f, m)) if progress else None)
    arr = feats.arrays
    av = album_from_features(arr["emb"], feats.asset_ids, arr["mood_arousal_raw"], arr["mood_valence_raw"],
                             [a.taken_at for a in ms.assets])
    log.info("songpick: album mood from %d assets: valence %.2f, arousal %.2f", av.n_assets, av.valence, av.arousal)
    return av


# ---------- song side ----------

def tag_mood(tags: Tags) -> tuple[float, float] | None:
    """Weighted mean (valence, arousal) of the tags found in the lexicon; None if none is."""
    lex = lexicon()
    hits = [(lex[norm_tag(t)], w) for t, w in tags if norm_tag(t) in lex and w > 0]
    if not hits:
        return None
    tw = sum(w for _, w in hits)
    return (sum(va[0] * w for va, w in hits) / tw, sum(va[1] * w for va, w in hits) / tw)


def track_embedding(tags: Tags, encode_text: EncodeText) -> np.ndarray | None:
    templates = get_config().songpick.vibe.templates
    tags = [(t, w) for t, w in tags if w > 0]
    if not tags:
        return None
    texts = [tpl.format(t) for t, _ in tags for tpl in templates]
    e = np.asarray(encode_text(texts), np.float32).reshape(len(tags), len(templates), -1).mean(1)
    w = np.array([wt for _, wt in tags], np.float32)
    v = (e * w[:, None]).sum(0) / w.sum()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None


def mood_match(song: tuple[float, float], album: AlbumVibe) -> float:
    d = math.hypot(song[0] - album.valence, song[1] - album.arousal)
    return max(0.0, 1.0 - d / MAX_VA_DIST)


def mood_word(valence: float, arousal: float) -> str | None:
    m = get_config().songpick.vibe.mood_word_min
    v = 1 if valence >= m else -1 if valence <= -m else 0
    a = 1 if arousal >= m else -1 if arousal <= -m else 0
    return {(1, 1): "upbeat", (1, -1): "warm and calm", (-1, 1): "intense", (-1, -1): "melancholic",
            (0, 1): "energetic", (0, -1): "calm", (1, 0): "bright", (-1, 0): "moody"}.get((v, a))


def _reason(c: TrackCandidate, album: AlbumVibe, good: bool) -> str:
    n = get_config().songpick.vibe.reason_tags
    named = ", ".join(c.tags[:n])
    if not good:
        return f"Tagged {named}."
    aw = mood_word(album.valence, album.arousal)
    sw = mood_word(c.valence, c.arousal) if c.valence is not None and c.arousal is not None else None
    if aw and sw == aw:
        return f"{aw.capitalize()}, like your photos (tagged {named})."
    return f"Fits your photos (tagged {named})."


def score(cands: list[TrackCandidate], album: AlbumVibe, tags_by_key: dict[str, Tags],
          encode_text: EncodeText | None = None) -> int:
    """Fill vibe fields and prepend a vibe reason, in place. Returns how many songs had usable tags."""
    cfg = get_config().songpick.vibe
    if encode_text is None:
        from montage.features.clip_encoder import encode_text
    raw_clip: dict[str, float] = {}
    for c in cands:
        tags = (tags_by_key.get(c.key) or [])[: cfg.top_tags]
        c.tags = [t for t, _ in tags]
        e = track_embedding(tags, encode_text)
        if e is not None:
            raw_clip[c.key] = float(e @ album.emb)
        va = tag_mood(tags)
        if va is not None:
            c.valence, c.arousal = round(va[0], 4), round(va[1], 4)
            c.vibe_mood = round(mood_match(va, album), 6)
    if raw_clip:
        lo, hi = min(raw_clip.values()), max(raw_clip.values())
        for c in cands:
            if c.key in raw_clip:
                c.vibe_clip = round((raw_clip[c.key] - lo) / (hi - lo), 6) if hi - lo > 1e-9 else 0.5
    for c in cands:
        terms = [(cfg.w_clip, c.vibe_clip), (cfg.w_mood, c.vibe_mood)]
        terms = [(w, x) for w, x in terms if x is not None and w > 0]
        if terms:
            c.vibe = round(sum(w * x for w, x in terms) / sum(w for w, _ in terms), 6)
    scored = [c.vibe for c in cands if c.vibe is not None]
    if not scored:
        return 0
    med = float(np.median(scored))
    for c in cands:
        if c.vibe is None:
            c.vibe = round(med, 6)
        elif c.tags:
            c.reason = f"{_reason(c, album, c.vibe >= med)} {c.reason}"
    return len(scored)


class Scorer:
    """Callable hook for `pick.suggest(vibe_scorer=...)`: tag lookup + `score` for one album."""

    def __init__(self, album: AlbumVibe, lookup: Callable[[list[tuple[str, str, str]]], dict[str, Tags]],
                 encode_text: EncodeText | None = None):
        self.album, self.lookup, self.encode_text = album, lookup, encode_text
        self.n_scored = 0
        self.n_looked_up = 0

    def __call__(self, cands: list[TrackCandidate]) -> bool:
        tags = self.lookup([(c.key, c.artist, c.track) for c in cands])
        self.n_looked_up += len(cands)
        n = score(cands, self.album, tags, self.encode_text)
        self.n_scored += n
        log.info("songpick: vibe for %d of %d songs", n, len(cands))
        return n > 0
