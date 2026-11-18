"""Song-pick study (phase 12, offline): what the tag ranker, the audio ranker and both pick for each kind of album.

    python scripts/run_job.py --name songpick_study -- python -m montage.eval.songpick_study

Albums: the CUFED event albums (`cufed_photos.npz`, phase 15; album mood and embedding as `vibe.album_from_features`).
Songs: the public list `configs/study_songs.tsv` (10 per mood quadrant, hand-labelled), so no one's listening data
is used. Per song: Last.fm/MusicBrainz tags (memory only, never written) and the audio mood (YouTube fetch +
`sentiment.song_mood`; the opt-in fetch, approved for this study by Andrew on 2026-11-18). Needs the GPU, the
MERT head and network; resumable (the song moods are cached in the output folder).

Rankers per album, all through the app's own scoring: tags (`vibe.score`, stage A), audio (`vibe.mood_match` of the
audio mood alone), both (`sentiment.Scorer` on top of stage A, as the app does with the YouTube box ticked).
Outputs in DATA_ROOT/eval/songpick/: `report.md` (tables for RESULTS.md) and `metrics.json`:
- mood agreement: share of songs whose tag mood / audio mood falls in the hand-labelled quadrant;
- Spearman of tag vs audio valence and arousal over the songs that have both;
- per event type, the songs each ranker puts on top (mean rank over that type's albums);
- Kendall tau between the tags and both orders, averaged over albums (how much the audio moves the list).
Evaluation only: nothing here tunes a weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from montage import config, jobs, paths
from montage.config import repo_path
from montage.log import get_logger
from montage.songpick import sentiment, tags, vibe
from montage.songpick.affinity import track_key
from montage.types import TrackCandidate

log = get_logger("songpick_study")
SONGS = "configs/study_songs.tsv"
QUADRANT = {(1, 1): "upbeat", (-1, 1): "intense", (1, -1): "warm", (-1, -1): "sad"}


def out_dir() -> Path:
    d = paths.data_root() / "eval" / "songpick"
    d.mkdir(parents=True, exist_ok=True)
    return d


def study_songs() -> list[tuple[str, str, str]]:
    rows = [line.split("\t") for line in repo_path(SONGS).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    return [(a, t, q) for a, t, q in rows[1:]]


def quadrant(va: tuple[float, float] | None) -> str | None:
    return None if va is None else QUADRANT[(1 if va[0] >= 0 else -1, 1 if va[1] >= 0 else -1)]


def cand(artist: str, track: str) -> TrackCandidate:
    return TrackCandidate(key=track_key(artist, track), track=track, artist=artist, album="", uri=None,
                          plays_window=0, plays_total=0, n_plays_window=0, n_skips_window=0, lift=1, affinity=0,
                          score=0, reason="")


def audio_moods(songs: list[tuple[str, str, str]]) -> dict[str, tuple[float, float]]:
    """Standardized audio mood per song key; raw moods cached in moods.json, so a rerun fetches nothing new."""
    from montage.songpick import youtube
    path = out_dir() / "moods.json"
    raw: dict[str, list[float] | None] = json.loads(path.read_text()) if path.exists() else {}
    for i, (a, t, _) in enumerate(songs):
        k = track_key(a, t)
        if k not in raw:
            try:
                audio, _ = youtube.find_and_fetch(t, a)
                va = sentiment.raw_mood(audio)
            except Exception as e:  # noqa: BLE001  one song failing must not end the study
                log.warning("study song %d: %s", i, type(e).__name__)
                va = None
            raw[k] = list(va) if va is not None else None
            paths.atomic_write(path, lambda p: Path(p).write_text(json.dumps(raw, indent=1)))
        jobs.report(0.05 + 0.6 * (i + 1) / len(songs), f"audio mood {i + 1}/{len(songs)}")
    return {k: sentiment.standardize(tuple(v)) for k, v in raw.items() if v is not None}


def cufed_albums() -> list[tuple[str, vibe.AlbumVibe]]:
    from montage.features.mood import mood_raw
    from training.common import datasets_dir
    z = np.load(datasets_dir("importance") / "cufed_photos.npz")
    emb = z["X"][:, :768].astype(np.float32)   # see importance.ranker.ranker_features
    a_raw, v_raw = mood_raw(emb)
    out = []
    for g in np.unique(z["group"]):
        m = z["group"] == g
        ids = [str(i) for i in range(int(m.sum()))]
        out.append((str(z["event"][m][0]), vibe.album_from_features(emb[m], ids, a_raw[m], v_raw[m])))
    return out


def rank(album: vibe.AlbumVibe, songs, tag_map, moods, encode_text=None) -> dict[str, list[str]]:
    """Song keys best first, per ranker."""
    cs = [cand(a, t) for a, t, _ in songs]
    vibe.score(cs, album, tag_map, encode_text)
    tags_order = [c.key for c in sorted(cs, key=lambda c: (-(c.vibe or 0), c.key))]
    audio_order = sorted(moods, key=lambda k: (-vibe.mood_match(moods[k], album), k))
    fake = sentiment.Scorer(album, fetch=lambda t, a, d: (Path(track_key(a, t)), {}),
                            mood=lambda p: moods.get(str(p)))
    with config.override({"songpick.sentiment.max_songs": len(cs), "songpick.sentiment.budget_s": 1e9}):
        fake(cs)
    both_order = [c.key for c in sorted(cs, key=lambda c: (-(c.vibe or 0), c.key))]
    return {"tags": tags_order, "audio": audio_order, "both": both_order}


def kendall(a: list[str], b: list[str]) -> float:
    from scipy.stats import kendalltau
    common = [k for k in a if k in b]
    return float(kendalltau([a.index(k) for k in common], [b.index(k) for k in common]).statistic)


def main() -> int:
    paths.load_colab_env()
    from scipy.stats import spearmanr
    songs = study_songs()
    jobs.report(0.01, "song tags", force=True)
    tag_map = tags.lookup([(track_key(a, t), a, t) for a, t, _ in songs])
    moods = audio_moods(songs)
    tag_moods = {track_key(a, t): vibe.tag_mood(tag_map.get(track_key(a, t)) or []) for a, t, _ in songs}
    names = {track_key(a, t): f"{a} - {t}" for a, t, _ in songs}
    expected = {track_key(a, t): q for a, t, q in songs}

    agree = {src: float(np.mean([quadrant(m.get(k)) == expected[k] for k in expected if m.get(k) is not None]))
             if any(m.get(k) is not None for k in expected) else float("nan")
             for src, m in (("tags", tag_moods), ("audio", moods))}
    both = [k for k in expected if tag_moods.get(k) is not None and k in moods]
    corr = {d: float(spearmanr([tag_moods[k][i] for k in both], [moods[k][i] for k in both]).statistic)
            if len(both) > 2 else float("nan") for i, d in enumerate(("valence", "arousal"))}

    jobs.report(0.7, "ranking albums")
    by_event: dict[str, dict[str, dict[str, list[int]]]] = {}
    taus = []
    for event, av in cufed_albums():
        orders = rank(av, songs, tag_map, moods)
        taus.append(kendall(orders["tags"], orders["both"]))
        for r, order in orders.items():
            for pos, k in enumerate(order):
                by_event.setdefault(event, {}).setdefault(r, {}).setdefault(k, []).append(pos)

    metrics = {"n_songs": len(songs), "n_tags": sum(v is not None for v in tag_moods.values()),
               "n_audio": len(moods), "quadrant_agreement": agree, "tag_vs_audio_spearman": corr,
               "kendall_tags_vs_both": float(np.nanmean(taus)), "n_albums": len(taus)}
    lines = ["# Song-pick study (offline)", "",
             f"{len(songs)} public songs (10 per hand-labelled mood quadrant), {len(taus)} CUFED event albums. "
             f"Song mood found: tags {metrics['n_tags']}, audio {metrics['n_audio']}.", "",
             "| Mood source | Agrees with hand label |", "|---|---|"]
    lines += [f"| {src} | {v:.0%} |" for src, v in agree.items()]
    lines += ["", f"Spearman tag vs audio mood (songs with both, n={len(both)}): valence {corr['valence']:.2f}, "
              f"arousal {corr['arousal']:.2f}. Kendall tau tags vs both, mean over albums: "
              f"{metrics['kendall_tags_vs_both']:.2f}.", "",
              "| Event type | Tags: top 3 | Audio: top 3 | Both: top 3 |", "|---|---|---|---|"]
    for event in sorted(by_event):
        cells = []
        for r in ("tags", "audio", "both"):
            mean_pos = {k: np.mean(v) for k, v in by_event[event].get(r, {}).items()}
            cells.append(", ".join(names[k] for k in sorted(mean_pos, key=lambda k: (mean_pos[k], k))[:3]))
        lines.append(f"| {event} | " + " | ".join(cells) + " |")
    (out_dir() / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir() / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    jobs.report(1.0, "done", force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
