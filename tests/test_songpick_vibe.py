"""Phase 12.8 acceptance (stage A vibe match): tag lexicon, album pooling, vibe scoring, ranking hook, tag lookup.
Pure Python: a fake CLIP text encoder and canned Last.fm/MusicBrainz JSON (made-up artists); no network, no weights."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from montage import config
from montage.app import songpick_ui
from montage.songpick import tags as tags_mod
from montage.songpick import vibe
from montage.songpick.pick import NO_TAGS, suggest
from montage.types import ListeningEvent, TopTrack, TrackCandidate

NOW = datetime(2026, 11, 5, 12, 0, 0)
AXES = {"party": 0, "dance": 0, "happy": 0, "summer": 0, "sad": 1, "ambient": 1, "melancholy": 1, "calm": 1}


def fake_encode(texts):
    """4-d stand-in for CLIP text: party-ish words on axis 0, quiet/sad words on axis 1, the rest on axis 3."""
    out = []
    for t in texts:
        v = np.zeros(4, np.float32)
        hits = [AXES[w] for w in AXES if w in t]
        for h in hits:
            v[h] += 1.0
        if not hits:
            v[3] = 1.0
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


def album(kind: str) -> vibe.AlbumVibe:
    if kind == "party":
        return vibe.AlbumVibe(emb=np.array([1, 0, 0, 0], np.float32), valence=0.7, arousal=0.6, n_assets=10)
    return vibe.AlbumVibe(emb=np.array([0, 1, 0, 0], np.float32), valence=-0.5, arousal=-0.6, n_assets=10)


HAPPY = [("happy", 100.0), ("dance", 80.0), ("summer", 60.0)]
SAD = [("sad", 100.0), ("ambient", 70.0), ("melancholy", 50.0)]


def top_lists() -> dict[str, list[TopTrack]]:
    """Three songs with clear listening order: Fixture Bloc (#1) > Glum Owls (#2) > Neon Kites (#3)."""
    songs = [("Fixture Bloc", "Steady Song"), ("Glum Owls", "Rainy Window"), ("Neon Kites", "Beach Party")]
    return {"short_term": [TopTrack(track=t, artist=a, time_range="short_term", rank=i + 1)
                           for i, (a, t) in enumerate(songs)]}


TAGS = {"neon kites|beach party": HAPPY, "glum owls|rainy window": SAD}


def scorer(kind: str, tags=TAGS):
    return vibe.Scorer(album(kind), lambda triples: {k: tags[k] for k, _, _ in triples if k in tags}, fake_encode)


# ---------- lexicon, album, song side ----------

def test_lexicon_and_tag_mood():
    assert vibe.norm_tag(" Hip-Hop_  Classics ") == "hip hop classics"
    lex = vibe.lexicon()
    assert lex["happy"][0] > 0.5 and lex["sad"][0] < -0.5 and lex["ambient"][1] < -0.5 and lex["k pop"] == lex["k-pop".replace("-", " ")]
    assert all(-1 <= v <= 1 and -1 <= a <= 1 for v, a in lex.values())
    v, a = vibe.tag_mood([("happy", 100), ("sad", 100), ("unknown words", 100)])
    assert v == pytest.approx((lex["happy"][0] + lex["sad"][0]) / 2) and a == pytest.approx((lex["happy"][1] + lex["sad"][1]) / 2)
    assert vibe.tag_mood([("unknown words", 100)]) is None and vibe.tag_mood([]) is None


def test_album_pools_per_asset_and_mood_is_absolute():
    # asset "vid" has three identical windows; it must count once, like the photo
    emb = np.array([[1, 0, 0, 0]] * 3 + [[0, 1, 0, 0]], np.float32)
    s = config.get_config().songpick.vibe.album_mood_scale
    av = vibe.album_from_features(emb, ["vid", "vid", "vid", "photo"], np.array([1.0, 1.0, 1.0, 3.0]),
                                  np.array([-2.0, -2.0, -2.0, 0.0]), [datetime(2026, 7, 3)])
    assert np.allclose(av.emb, np.array([1, 1, 0, 0]) / np.sqrt(2)) and av.n_assets == 2
    assert av.arousal == pytest.approx(np.tanh(2.0 / s["arousal"])) and av.valence == pytest.approx(np.tanh(-1.0 / s["valence"]))
    assert av.taken_at == [datetime(2026, 7, 3)]


def test_mood_word_quadrants():
    assert vibe.mood_word(0.7, 0.6) == "upbeat" and vibe.mood_word(-0.6, -0.5) == "melancholic"
    assert vibe.mood_word(0.0, 0.0) is None and vibe.mood_word(0.05, -0.8) == "calm"


# ---------- scoring ----------

def cand(key: str, aff: float = 1.0) -> TrackCandidate:
    artist, track = key.split("|")
    return TrackCandidate(key=key, track=track, artist=artist, album="", uri=None, plays_window=0, plays_total=0,
                          n_plays_window=0, n_skips_window=0, lift=1, affinity=aff, score=aff, reason="#1 in your top songs.")


@pytest.mark.parametrize("kind,winner", [("party", "neon kites|beach party"), ("calm", "glum owls|rainy window")])
def test_score_prefers_the_matching_vibe(kind, winner):
    cs = [cand("neon kites|beach party"), cand("glum owls|rainy window"), cand("fixture bloc|steady song")]
    n = vibe.score(cs, album(kind), TAGS, fake_encode)
    assert n == 2
    by = {c.key: c for c in cs}
    loser = ({"neon kites|beach party", "glum owls|rainy window"} - {winner}).pop()
    assert by[winner].vibe > by[loser].vibe and by[winner].vibe_clip == 1.0 and by[loser].vibe_clip == 0.0
    assert by[winner].vibe_mood > by[loser].vibe_mood
    untagged = by["fixture bloc|steady song"]  # no tags: the median, so it is neither rewarded nor punished
    assert untagged.vibe == pytest.approx(float(np.median([by[winner].vibe, by[loser].vibe])))
    assert untagged.vibe_clip is None and untagged.tags == [] and untagged.reason == "#1 in your top songs."
    assert by[winner].reason.endswith("#1 in your top songs.") and "tagged" in by[winner].reason
    assert by[winner].tags[0] in ("happy", "sad")


def test_score_reason_names_the_shared_mood():
    cs = [cand("neon kites|beach party"), cand("glum owls|rainy window")]
    vibe.score(cs, album("party"), TAGS, fake_encode)
    assert cs[0].reason.startswith("Upbeat, like your photos (tagged happy, dance, summer).")
    assert cs[1].reason.startswith("Tagged sad, ambient, melancholy.")


def test_score_without_tags_leaves_vibe_unset():
    cs = [cand("a|b"), cand("c|d")]
    assert vibe.score(cs, album("party"), {}, fake_encode) == 0
    assert all(c.vibe is None for c in cs)


# ---------- ranking hook (pick.suggest) ----------

def test_vibe_mode_reranks_by_vibe_and_blends_with_alpha():
    alpha = config.get_config().songpick.vibe.alpha_by_mode["vibe"]
    base = suggest([NOW - timedelta(days=2)], mode="vibe", top=top_lists(), now=NOW)
    assert [c.artist for c in base.candidates] == ["Fixture Bloc", "Glum Owls", "Neon Kites"]
    party = suggest([NOW - timedelta(days=2)], mode="vibe", top=top_lists(), now=NOW, vibe_scorer=scorer("party"))
    calm = suggest([NOW - timedelta(days=2)], mode="vibe", top=top_lists(), now=NOW, vibe_scorer=scorer("calm"))
    assert party.candidates[0].artist == "Neon Kites" and calm.candidates[0].artist == "Glum Owls"
    assert party.vibe_source == "tags" and not any("photos' mood" in n for n in party.notes)
    for c in party.candidates:
        assert c.score == pytest.approx(alpha * c.affinity + (1 - alpha) * c.vibe, abs=1e-6)
    again = suggest([NOW - timedelta(days=2)], mode="vibe", top=top_lists(), now=NOW, vibe_scorer=scorer("party"))
    assert again.model_dump() == party.model_dump()  # deterministic


def test_scorer_without_tags_keeps_affinity_order():
    base = suggest([NOW], mode="vibe", top=top_lists(), now=NOW)
    none = suggest([NOW], mode="vibe", top=top_lists(), now=NOW, vibe_scorer=scorer("party", tags={}))
    assert [c.key for c in none.candidates] == [c.key for c in base.candidates]
    assert none.vibe_source == "none" and NO_TAGS in none.notes and all(c.vibe is None for c in none.candidates)


def test_scorer_only_sees_the_best_max_tracks():
    seen = []

    def spy(cands):
        seen.extend(c.key for c in cands)
        return vibe.score(cands, album("party"), TAGS, fake_encode) > 0
    with config.override({"songpick.vibe.max_tracks": 2}):
        pick = suggest([NOW], mode="vibe", top=top_lists(), now=NOW, vibe_scorer=spy)
    assert seen == ["fixture bloc|steady song", "glum owls|rainy window"] and len(pick.candidates) == 2


def test_release_eligibility_is_unchanged_by_vibe():
    from test_songpick_connect import listening
    recent, top = listening()
    dates = [datetime(2021, 7, 3), datetime(2021, 7, 10)]
    with config.override({"songpick.shortlist_size": 50}):
        base = suggest(dates, mode="release", recent=recent, top=top, now=NOW)
        tagged = {c.key: HAPPY for c in base.candidates[1:]}
        withv = suggest(dates, mode="release", recent=recent, top=top, now=NOW, vibe_scorer=scorer("party", tagged))
    assert {c.key for c in withv.candidates} == {c.key for c in base.candidates}
    assert withv.period_start == base.period_start and withv.mode == "release" and withv.vibe_source == "tags"
    assert any(c.reason.startswith("Upbeat, like your photos") and "Came out" in c.reason for c in withv.candidates)


def test_export_mode_uses_the_scorer():
    evs = [ListeningEvent(ts=datetime(2026, 7, 3 + i % 5, 12), ms_played=200_000, track=t, artist=a)
           for i in range(12) for a, t in [("Glum Owls", "Rainy Window"), ("Neon Kites", "Beach Party")]]
    with config.override({"songpick.min_window_events": 1}):
        pick = suggest([datetime(2026, 7, 3), datetime(2026, 7, 8)], mode="export", export=evs, now=NOW,
                       vibe_scorer=scorer("calm"))
    assert pick.vibe_source == "tags" and pick.candidates[0].artist == "Glum Owls"


def test_ui_suggest_sets_album_mood(monkeypatch):
    av = album("party")
    av.taken_at = [NOW - timedelta(days=2)]
    monkeypatch.setattr(songpick_ui, "_album", lambda media, progress=None: (av.taken_at, av, []))
    monkeypatch.setattr(tags_mod, "lookup", lambda triples: {k: TAGS[k] for k, _, _ in triples if k in TAGS})
    from montage.features import clip_encoder
    monkeypatch.setattr(clip_encoder, "encode_text", fake_encode)
    choices, pick, md = songpick_ui.suggest(["album"], None, {"top": top_lists()}, "vibe")
    assert pick.vibe_source == "tags" and pick.album_valence == 0.7 and pick.album_arousal == 0.6
    assert "Neon Kites" in choices[0][0] and md.startswith("Your songs that best fit")


def test_ui_album_mood_failure_falls_back(monkeypatch):
    monkeypatch.setattr(vibe, "album_vibe", lambda media, progress=None: (_ for _ in ()).throw(RuntimeError("no GPU")))
    monkeypatch.setattr(songpick_ui, "_album_dates", lambda media: [NOW])
    taken, av, notes = songpick_ui._album(["x.jpg"])
    assert taken == [NOW] and av is None and notes == [songpick_ui.NO_MOOD]


# ---------- tag lookup (Last.fm, MusicBrainz) ----------

@pytest.fixture
def net(monkeypatch):
    """Canned HTTP: {(host-ish, method or query): body}. Records calls."""
    tags_mod.clear_cache()
    calls = []
    sleeps = []
    monkeypatch.setattr(tags_mod, "_sleep", sleeps.append)
    state = {"routes": {}}

    def get_json(url, params, headers=None, timeout=5.0):
        calls.append((url, dict(params), dict(headers or {})))
        key = params.get("method") or params.get("query")
        return state["routes"].get(key)
    monkeypatch.setattr(tags_mod, "_get_json", get_json)
    yield state, calls, sleeps
    tags_mod.clear_cache()


def lfm(*pairs):
    return {"toptags": {"tag": [{"name": n, "count": c} for n, c in pairs]}}


def test_clean_drops_junk_and_artist_and_rescales():
    out = tags_mod.clean([("Seen Live", 100), ("Indie-Pop", 50), ("Neon Kites", 40), ("summer", 4), ("dance", "x")],
                         "Neon Kites")
    assert out == [("indie pop", 50.0)]
    assert tags_mod.clean([("a", 0)], "x") == [] and tags_mod.clean([], "x") == []


def test_lastfm_track_tags_then_artist_fill(monkeypatch, net):
    state, calls, _ = net
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    state["routes"] = {"track.gettoptags": lfm(("summer", 100)),
                       "artist.gettoptags": lfm(("indie pop", 100), ("happy", 60))}
    out = tags_mod.lookup([("neon kites|beach party", "Neon Kites", "Beach Party")])
    assert out == {"neon kites|beach party": [("summer", 100.0), ("indie pop", 50.0), ("happy", 30.0)]}
    assert all(p["api_key"] == "k" and set(p) >= {"artist", "method"} for _, p, _ in calls)
    assert not any("query" in p for _, p, _ in calls)  # Last.fm had tags: no MusicBrainz
    n = len(calls)
    tags_mod.lookup([("neon kites|beach party", "Neon Kites", "Beach Party")])
    assert len(calls) == n  # cached in memory


def test_no_key_uses_musicbrainz(monkeypatch, net):
    state, calls, sleeps = net
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    state["routes"] = {'artist:"Glum Owls"': {"artists": [{"score": 100, "tags": [{"name": "Ambient", "count": 4},
                                                                                  {"name": "sad", "count": 2}]}]},
                       'artist:"Weak Match"': {"artists": [{"score": 40, "tags": [{"name": "rock", "count": 9}]}]}}
    out = tags_mod.lookup([("glum owls|rainy window", "Glum Owls, Guest", "Rainy Window"),
                           ("weak match|x", "Weak Match", "X")])
    assert out == {"glum owls|rainy window": [("ambient", 100.0), ("sad", 50.0)]}
    _, params, headers = calls[0]
    assert headers["User-Agent"].startswith("bac_prototype") and params["fmt"] == "json"
    assert len(sleeps) <= len(calls)  # rate limit goes through _sleep


def test_network_failure_gives_no_tags_and_is_not_cached(monkeypatch, net):
    state, calls, _ = net
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    out = tags_mod.lookup([("a|b", "A", "B")])  # every request returns None
    assert out == {}
    n = len(calls)
    tags_mod.lookup([("a|b", "A", "B")])
    assert len(calls) > n  # failures are retried next time


def test_musicbrainz_lookups_are_capped(monkeypatch, net):
    _, calls, _ = net
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    with config.override({"songpick.vibe.musicbrainz.max_lookups": 2}):
        tags_mod.lookup([(f"a{i}|t", f"A{i}", "T") for i in range(5)])
    assert len(calls) == 2


# ---------- Colab only ----------

@pytest.mark.gpu
def test_album_vibe_on_fixtures(fixtures):
    from conftest import gpu_available
    if not gpu_available():
        pytest.skip("no GPU")
    media = sorted(str(f) for f in fixtures.iterdir() if f.suffix in (".mp4", ".mov", ".jpg"))
    av = vibe.album_vibe(media)
    assert av.emb.shape == (768,) and abs(float(np.linalg.norm(av.emb)) - 1) < 1e-4
    assert -1 <= av.valence <= 1 and -1 <= av.arousal <= 1 and av.n_assets == len(av.taken_at)
    print(f"fixture album mood: valence {av.valence:.3f}, arousal {av.arousal:.3f}")


@pytest.mark.network
def test_tag_lookup_live():
    tags_mod.clear_cache()
    out = tags_mod.lookup([("radiohead|no surprises", "Radiohead", "No Surprises")])
    assert out and out["radiohead|no surprises"]
