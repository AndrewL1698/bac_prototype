"""Phase 12.9 acceptance (sentiment match from audio): the top songs' audio mood re-ranks the stage A list.
Pure Python: a fake YouTube fetch and fake audio moods (made-up artists); no network, no weights, except the
`gpu` test, which runs the real song analysis on a fixture song."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from montage import config
from montage.app import songpick_ui
from montage.songpick import sentiment
from montage.songpick.pick import NO_AUDIO, suggest
from test_songpick_vibe import NOW, TAGS, album, fake_encode, scorer, top_lists


@pytest.fixture(autouse=True)
def _no_why_llm():
    """The Suggest handler calls the director's LLM for why-this-song (12.10) when a key is set; keep it offline."""
    with config.override({"songpick.why.enabled": False}):
        yield


FIX, GLUM, NEON = "fixture bloc|steady song", "glum owls|rainy window", "neon kites|beach party"
# How the songs sound (valence, arousal), deliberately against their tags: Glum Owls sounds upbeat, Neon Kites sad.
AUDIO = {FIX: (0.0, 0.0), GLUM: (0.7, 0.6), NEON: (-0.5, -0.6)}
SAME = {k: [("summer", 100.0)] for k in AUDIO}   # tags that do not separate the songs
DATES = [NOW - timedelta(days=2)]


def key_of(track: str, artist: str) -> str:
    return f"{artist}|{track}".lower()


def listener(kind: str, audio=AUDIO, fail=(), seen=None):
    """sentiment.Scorer for a party/calm album with a fake fetch (path = the song key) and fake moods."""
    def fetch(track, artist, duration_ms):
        k = key_of(track, artist)
        if seen is not None:
            seen.append(k)
        if k in fail:
            raise RuntimeError("blocked")
        return Path(k), {}
    return sentiment.Scorer(album(kind), fetch, lambda p: audio.get(str(p)))


def run(kind: str, tags=SAME, **kw):
    return suggest(DATES, mode="vibe", top=top_lists(), now=NOW, vibe_scorer=scorer(kind, tags),
                   sentiment_scorer=listener(kind, **kw))


@pytest.mark.parametrize("kind,winner", [("party", GLUM), ("calm", NEON)])
def test_audio_mood_reranks(kind, winner):
    pick = run(kind)
    first = pick.candidates[0]
    assert first.key == winner and pick.vibe_source == "audio" and NO_AUDIO not in pick.notes
    assert first.mood_source == "audio" and (first.valence, first.arousal) == AUDIO[winner]
    assert first.vibe_sent == 1.0 and first.vibe_clip == 0.5   # tags tie, so the sound decides
    word = "upbeat" if kind == "party" else "melancholic"
    assert first.reason.startswith(f"Sounds {word}, like your photos (tagged summer).")
    assert all(c.reason.count("(tagged") <= 1 for c in pick.candidates)   # the stage A line was replaced
    alpha = config.get_config().songpick.vibe.alpha_by_mode["vibe"]
    for c in pick.candidates:
        assert c.score == pytest.approx(alpha * c.affinity + (1 - alpha) * c.vibe, abs=1e-6)
    assert run(kind).model_dump() == pick.model_dump()   # deterministic


@pytest.mark.parametrize("w_clip,w_sent", [(0.6, 0.4), (0.4, 0.6)])
def test_weights_decide_between_tags_and_sound(w_clip, w_sent):
    # party album: Neon Kites' tags fit (CLIP), Glum Owls' sound fits; Fixture Bloc has no tags and no audio.
    with config.override({"songpick.vibe.w_clip": w_clip, "songpick.sentiment.w_sent": w_sent}):
        pick = run("party", tags=TAGS, fail={FIX})
    by = {c.key: c for c in pick.candidates}
    assert (by[NEON].vibe_clip, by[NEON].vibe_sent, by[GLUM].vibe_clip, by[GLUM].vibe_sent) == (1.0, 0.0, 0.0, 1.0)
    order = [c.key for c in pick.candidates]
    assert (order.index(NEON) < order.index(GLUM)) == (w_clip > w_sent)
    assert by[FIX].vibe_sent is None and by[FIX].mood_source is None   # failed fetch keeps its stage A vibe


@pytest.mark.parametrize("how", ["blocked", "heuristic", "budget"])
def test_no_audio_keeps_the_tag_order(how):
    base = suggest(DATES, mode="vibe", top=top_lists(), now=NOW, vibe_scorer=scorer("party", TAGS))
    kw = {"blocked": {"fail": set(AUDIO)}, "heuristic": {"audio": {}}, "budget": {}}[how]
    with config.override({"songpick.sentiment.budget_s": -1.0 if how == "budget" else 90.0}):
        pick = run("party", tags=TAGS, **kw)
    assert [c.key for c in pick.candidates] == [c.key for c in base.candidates]
    assert pick.vibe_source == "tags" and NO_AUDIO in pick.notes
    assert all(c.mood_source != "audio" for c in pick.candidates)


def test_only_the_best_max_songs_are_fetched():
    seen: list[str] = []
    base = suggest(DATES, mode="vibe", top=top_lists(), now=NOW, vibe_scorer=scorer("party", TAGS))
    with config.override({"songpick.sentiment.max_songs": 2}):
        pick = run("party", tags=TAGS, seen=seen)
    assert sorted(seen) == sorted(c.key for c in base.candidates[:2]) and len(pick.candidates) == 3


def test_audio_without_any_tags():
    pick = run("party", tags={})
    assert pick.vibe_source == "audio" and pick.candidates[0].key == GLUM
    assert pick.candidates[0].reason.startswith("Sounds upbeat, like your photos.")


def test_release_eligibility_is_unchanged_by_audio():
    from test_songpick_connect import listening
    recent, top = listening()
    dates = [datetime(2021, 7, 3), datetime(2021, 7, 10)]
    with config.override({"songpick.shortlist_size": 50}):
        base = suggest(dates, mode="release", recent=recent, top=top, now=NOW)
        moods = {c.key: (0.5, 0.5) for c in base.candidates}
        heard = suggest(dates, mode="release", recent=recent, top=top, now=NOW,
                        vibe_scorer=scorer("party", {}), sentiment_scorer=listener("party", audio=moods))
    assert {c.key for c in heard.candidates} == {c.key for c in base.candidates}
    assert heard.period_start == base.period_start and heard.mode == "release" and heard.vibe_source == "audio"


def test_standardize_uses_center_and_scale():
    with config.override({"songpick.sentiment.song_mood_center": {"valence": 0.1, "arousal": -0.2},
                          "songpick.sentiment.song_mood_scale": {"valence": 0.5, "arousal": 0.25}}):
        v, a = sentiment.standardize((0.1, 0.05))
    assert v == pytest.approx(0.0) and a == pytest.approx(0.7616, abs=1e-4)


# ---------- app handler ----------

@pytest.fixture
def ui(monkeypatch):
    av = album("party")
    av.taken_at = DATES
    monkeypatch.setattr(songpick_ui, "_album", lambda media, progress=None: (av.taken_at, av, []))
    from montage.features import clip_encoder
    from montage.songpick import tags as tags_mod
    monkeypatch.setattr(tags_mod, "lookup", lambda triples: {k: SAME[k] for k, _, _ in triples if k in SAME})
    monkeypatch.setattr(clip_encoder, "encode_text", fake_encode)
    built = []
    real = listener("party")   # built before the patch below replaces sentiment.Scorer

    def fake_scorer(album_vibe):
        built.append(album_vibe)
        return real
    monkeypatch.setattr(sentiment, "Scorer", fake_scorer)
    return built


@pytest.mark.parametrize("ticked,mert,expect", [(False, True, "tags"), (True, False, "tags"), (True, True, "audio")])
def test_ui_listens_only_when_ticked_and_mert_is_there(monkeypatch, ui, ticked, mert, expect):
    from montage.music import emotion
    monkeypatch.setattr(emotion, "mert_available", lambda: mert)
    choices, pick, md = songpick_ui.suggest(["album"], None, {"top": top_lists()}, "vibe", youtube_ok=ticked)
    assert pick.vibe_source == expect and len(ui) == (1 if expect == "audio" else 0)
    if expect == "audio":
        assert md.startswith("Your songs that sound most like your photos.") and "Glum Owls" in choices[0][0]


# ---------- on Colab ----------

@pytest.mark.gpu
def test_song_mood_on_fixture_song(fixtures):
    from conftest import gpu_available
    from montage.music.emotion import mert_available
    if not gpu_available() or not mert_available():
        pytest.skip("no GPU or no MERT emotion head")
    va = sentiment.song_mood(fixtures / "song_sections.wav")
    assert va is not None and all(-1 <= x <= 1 for x in va)
    print(f"fixture song mood: valence {va[0]:.3f}, arousal {va[1]:.3f}")
