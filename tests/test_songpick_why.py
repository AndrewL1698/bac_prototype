"""Phase 12.10 acceptance ("why this song"): what is sent to the LLM, how its lines are used, and the fallbacks.
A fake Anthropic client (made-up artists); the `api` test calls the real director model on Colab."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from montage import config, director
from montage.app import songpick_ui
from montage.director import llm
from montage.songpick import why
from montage.types import SongPick, TrackCandidate
from test_songpick_vibe import NOW, album, fake_encode, top_lists

REASON = "You played it 12 times between Jul 03 and Jul 10, 2026, 3.1x more than usual for you."


def cand(artist: str, track: str, tags: list[str], va=None, source=None) -> TrackCandidate:
    return TrackCandidate(
        key=f"{artist}|{track}".lower(), track=track, artist=artist, album="Secret Album", uri="spotify:track:x",
        plays_window=12, plays_total=40, n_plays_window=12, n_skips_window=0, lift=3.1, affinity=1.0, score=0.9,
        reason=REASON, tags=tags, valence=va[0] if va else None, arousal=va[1] if va else None, mood_source=source,
        release_date=date(2019, 5, 1), release_precision="month", duration_ms=201_000)


def pick() -> SongPick:
    return SongPick(period_start=datetime(2026, 7, 3), period_end=datetime(2026, 7, 10), period_source="album",
                    n_events=500, n_events_window=40, album_valence=0.7, album_arousal=0.6, vibe_source="audio",
                    candidates=[cand("Neon Kites", "Beach Party", ["happy", "dance"], (0.7, 0.6), "audio"),
                                cand("Glum Owls", "Rainy Window", ["sad"], (-0.5, -0.6), "tags"),
                                cand("Fixture Bloc", "Steady Song", [])])


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply in ("refusal", "max_tokens"):
            return SimpleNamespace(stop_reason=self.reply, content=[SimpleNamespace(type="text", text="")])
        text = json.dumps({"lines": self.reply})
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


@pytest.fixture
def client(monkeypatch):
    def make(reply, enabled=True):
        c = FakeClient(reply)
        monkeypatch.setattr(llm, "get_client", lambda: c)
        monkeypatch.setattr(director, "llm_enabled", lambda: enabled)
        return c
    return make


def test_facts_carry_names_tags_and_moods_only():
    text = why.facts(pick(), ["beach", "sunset"])
    assert "Photos: mood upbeat; they show beach, sunset." in text
    assert "1. Neon Kites - Beach Party; tags: happy, dance; mood: upbeat (heard in its audio)" in text
    assert "2. Glum Owls - Rainy Window; tags: sad; mood: melancholic (from its tags)" in text
    assert "3. Fixture Bloc - Steady Song" in text
    # never listening data, dates, album names or ids
    for leak in ("played", "12", "2026", "2019", "Jul", "usual", "Secret Album", "spotify:", "201"):
        assert leak not in text, leak


def test_explain_fills_why_and_keeps_the_reason(client):
    lines = ["Bright and danceable, like your beach day.", "A quiet one for the slower moments.", "Steady and easy."]
    c = client(lines)
    p = pick()
    assert why.explain(p, ["beach"]) is True
    assert [x.why for x in p.candidates] == lines and all(x.reason == REASON for x in p.candidates)
    sent = c.calls[0]
    assert sent["model"] == config.get_config().director.model and "Never mention lyrics" in sent["system"]
    assert sent["messages"][0]["content"] == why.facts(pick(), ["beach"])


@pytest.mark.parametrize("reply", [["only one line"], ["a", ""], "refusal", "max_tokens", TimeoutError("slow")])
def test_failures_leave_why_unset(client, reply):
    p = pick()
    client(reply)
    assert why.explain(p) is False and all(x.why is None for x in p.candidates)


def test_off_without_llm_or_when_disabled(client):
    c = client(["x", "y", "z"], enabled=False)
    assert why.explain(pick()) is False and not c.calls
    c = client(["x", "y", "z"])
    with config.override({"songpick.why.enabled": False}):
        assert why.explain(pick()) is False and not c.calls


def test_ui_shows_why_and_lists_the_facts_below(monkeypatch, client):
    av = album("party")
    av.taken_at, av.tags = [NOW - timedelta(days=2)], ["beach"]
    monkeypatch.setattr(songpick_ui, "_album", lambda media, progress=None: (av.taken_at, av, []))
    from montage.features import clip_encoder
    from montage.songpick import tags as tags_mod
    monkeypatch.setattr(tags_mod, "lookup", lambda triples: {})
    monkeypatch.setattr(clip_encoder, "encode_text", fake_encode)
    client(["One.", "Two.", "Three."])
    choices, p, md = songpick_ui.suggest(["album"], None, {"top": top_lists()}, "vibe")
    assert [label.split(": ", 1)[1] for label, _ in choices] == ["One.", "Two.", "Three."]
    assert "Details:" in md and "#1 in your top songs" in md


def test_album_tags_count_each_asset_once(monkeypatch):
    from montage.features import tags as ftags
    from montage.songpick import vibe
    rows = {"0": ["beach", "sea"], "1": ["beach"], "2": ["beach"], "3": ["beach", "city"]}
    monkeypatch.setattr(ftags, "top_tags", lambda emb, ids: {i: rows[i] for i in ids})
    # asset a has three rows (a video), b one: "beach" counts twice (once per asset), not four times
    assert vibe.album_tags(np.zeros((4, 4)), ["a", "a", "a", "b"]) == ["beach", "city", "sea"]


@pytest.mark.api
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_why_live():
    p = pick()
    assert why.explain(p, ["beach", "friends"]) is True
    for c in p.candidates:
        print(f"{c.artist} - {c.track}: {c.why}")
        assert c.why and len(c.why.split()) <= 2 * config.get_config().songpick.why.max_words
