"""Phase 12.5-12.7: Spotify sign-in (PKCE, callback, API parsing), multi-source suggestions, opt-in
YouTube matching, and the app handlers. No network: HTTP and yt-dlp are replaced at their seams."""

from __future__ import annotations

import base64
import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from montage import config
from montage.app import songpick_ui
from montage.songpick import spotify, youtube
from montage.songpick.pick import suggest
from montage.types import UserError
from scripts.fixtures_history import (BACKGROUND, FAVOURITE, TOP_PICK, spotify_recent_json, spotify_top_json,
                                      write_history)

NOW = datetime(2026, 11, 5, 12, 0, 0)
APP = "https://abc123def.gradio.live/"
CALLBACK = Path(__file__).resolve().parents[1] / "docs" / "spotify" / "callback.html"


def state_app_url(state: str) -> str:
    """What docs/spotify/callback.html does with the state: base64url-decode the part after '~'."""
    s = state.split("~", 1)[1]
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode("utf-8")


@pytest.fixture
def client_id(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test-client-id")
    spotify._pending.clear()
    yield "test-client-id"
    spotify._pending.clear()


def fake_api(calls: list):
    """A stand-in for spotify._request that serves the synthetic Web API."""
    def _request(method, url, **kw):
        calls.append((method, url, kw))
        if url == spotify.TOKEN_URL:
            return 200, {"access_token": "tok", "expires_in": 3600, "token_type": "Bearer"}, {}
        if url.endswith("/me/player/recently-played"):
            return 200, spotify_recent_json(NOW), {}
        if url.endswith("/me/top/tracks"):
            return 200, spotify_top_json(kw["params"]["time_range"]), {}
        return 404, {}, {}
    return _request


def token():
    return {"access_token": "tok", "expires_at": 9e18}


def listening():
    recent = spotify.recent_from_json(spotify_recent_json(NOW))
    top = {r: spotify.top_from_json(spotify_top_json(r), r) for r in spotify.TIME_RANGES}
    return recent, top


# ---------- PKCE, state, callback rule ----------

def test_pkce_matches_rfc7636_vector():
    assert spotify.code_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == \
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    v = spotify.make_verifier()
    assert 43 <= len(v) <= 128 and all(c.isalnum() or c in "-._~" for c in v)


@pytest.mark.parametrize("url,ok", [
    ("https://abc123.gradio.live/", True),
    ("https://abc123.gradio.live/?x=1", True),
    ("http://127.0.0.1:7860/", True),
    ("http://localhost:7860", True),
    ("http://abc123.gradio.live/", False),          # share links are https
    ("https://gradio.live/", False),                # bare suffix
    ("https://evil.com/.gradio.live", False),
    ("https://abc.gradio.live.evil.com/", False),
    ("https://user:pw@abc.gradio.live/", False),
    ("javascript:alert(1)", False),
    ("", False),
])
def test_safe_return_url(url, ok):
    assert spotify.safe_return_url(url) is ok


def test_state_round_trip_and_callback_page_rule():
    s = spotify.make_state(APP)
    assert state_app_url(s) == APP and s.split("~")[0]
    page = CALLBACK.read_text()
    assert "innerHTML" not in page
    for host in config.get_config().songpick.spotify.return_hosts:  # the page hardcodes the config's hosts
        assert re.search(r'"' + re.escape(host) + r'"', page), f"{host} missing from callback.html allowed()"
    assert config.get_config().songpick.spotify.redirect_uri.endswith("/spotify/callback.html")


def test_begin_auth_requires_client_id_and_safe_url(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    with pytest.raises(UserError, match="not set up"):
        spotify.begin_auth(APP)
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "x")
    with pytest.raises(UserError, match="share link"):
        spotify.begin_auth("https://example.com/")


def test_auth_flow_with_fake_spotify(client_id, monkeypatch):
    calls = []
    monkeypatch.setattr(spotify, "_request", fake_api(calls))
    url = spotify.begin_auth(APP + "?spotify_code=old&state=old")
    assert url.startswith(spotify.AUTH_URL)
    assert "scope=user-read-recently-played+user-top-read" in url and "code_challenge_method=S256" in url
    assert "client_secret" not in url
    state = [p.split("=", 1)[1] for p in url.split("?", 1)[1].split("&") if p.startswith("state=")][0]
    from urllib.parse import unquote
    state = unquote(state)
    assert state_app_url(state) == APP  # old query stripped from the return URL
    tok = spotify.finish_auth("the-code", state)
    assert tok["access_token"] == "tok"
    sent = calls[-1][2]["data"]
    assert sent["code"] == "the-code" and sent["code_verifier"] and "client_secret" not in sent
    with pytest.raises(UserError, match="expired or was already used"):
        spotify.finish_auth("the-code", state)  # one use only


def test_auth_state_expires(client_id, monkeypatch):
    monkeypatch.setattr(spotify, "_request", fake_api([]))
    url = spotify.begin_auth(APP, now=1000.0)
    from urllib.parse import parse_qs, urlsplit
    state = parse_qs(urlsplit(url).query)["state"][0]
    ttl = config.get_config().songpick.spotify.auth_ttl_s
    with pytest.raises(UserError, match="expired"):
        spotify.finish_auth("c", state, now=1000.0 + ttl + 1)


@pytest.mark.parametrize("status,match", [(401, "expired"), (403, "Development Mode"), (429, "limiting"),
                                          (500, "HTTP 500")])
def test_api_errors_are_friendly(monkeypatch, status, match):
    monkeypatch.setattr(spotify, "_request", lambda *a, **k: (status, {"error": "x"}, {}))
    with pytest.raises(UserError, match=match):
        spotify.recently_played(token())


# ---------- parsing ----------

def test_recent_and_top_parsing():
    recent, top = listening()
    assert len(recent) == 50 and all(e.source == "recent" and e.duration_ms for e in recent)
    assert recent == sorted(recent, key=lambda e: (e.ts, e.artist, e.track)) and recent[-1].ts <= NOW
    assert top["short_term"][0].rank == 1 and (top["short_term"][0].artist, top["short_term"][0].track) == TOP_PICK
    t = top["short_term"][0]
    assert t.release_date.isoformat() == "2026-06-12" and t.release_precision == "day" and t.release_reliable
    fav = next(x for x in top["long_term"] if (x.artist, x.track) == FAVOURITE)
    assert fav.release_date.isoformat() == "2019-01-01" and fav.release_precision == "year"
    assert recent[-1].release_date is not None
    from scripts.fixtures_history import REMASTER
    assert not next(x for x in top["short_term"] if (x.artist, x.track) == REMASTER).release_reliable
    assert spotify.recent_from_json({"items": [{"track": {"type": "episode", "name": "Pod"},
                                                "played_at": "2026-11-01T00:00:00Z"}]}) == []


def test_fetch_listening(monkeypatch):
    calls = []
    monkeypatch.setattr(spotify, "_request", fake_api(calls))
    recent, top = spotify.fetch_listening(token())
    assert len(recent) == 50 and set(top) == set(spotify.TIME_RANGES)
    assert all(c[2]["headers"]["Authorization"] == "Bearer tok" for c in calls)
    with pytest.raises(UserError, match="Connect Spotify"):
        spotify.recently_played({})


# ---------- suggestions from sign-in data ----------

def test_suggest_recent_photos_uses_recent_plays_and_short_term():
    recent, top = listening()
    pick = suggest([NOW - timedelta(days=2), NOW - timedelta(days=1)], mode="release", recent=recent, top=top, now=NOW)
    first = pick.candidates[0]
    assert (first.artist, first.track) == TOP_PICK  # played during the album, though it came out in June
    assert "recent" in first.sources and "top:short_term" in first.sources
    assert "recently played" in first.reason and "top songs of the last 4 weeks" in first.reason
    assert first.duration_ms == 195_000 and first.score == 1.0
    assert pick.mode == "release" and pick.period_source in ("release", "release_widened")


def test_suggest_old_photos_release_mode_picks_songs_from_back_then():
    recent, top = listening()
    pick = suggest([datetime(2021, 7, 3), datetime(2021, 7, 10)], mode="release", recent=recent, top=top, now=NOW)
    names = [(c.artist, c.track) for c in pick.candidates]
    assert pick.mode == "release" and BACKGROUND[0] in names       # came out Jul 2021
    assert TOP_PICK not in names                                   # the #1 song of this month came out in 2026
    assert all(c.release_date and c.release_date.year <= 2022 for c in pick.candidates)
    a1 = pick.candidates[names.index(BACKGROUND[0])]
    assert a1.reason.startswith("Came out Jul 2021, near your photos (Jul 2021).")
    with config.override({"songpick.release.min_candidates": 1}):  # one song is enough: no widening
        narrow = suggest([datetime(2021, 7, 3), datetime(2021, 7, 10)], mode="release", recent=recent, top=top, now=NOW)
    assert narrow.period_source == "release" and [(c.artist, c.track) for c in narrow.candidates] == [BACKGROUND[0]]
    assert pick.period_source == "release_widened" and pick.period_start < narrow.period_start


def test_suggest_release_mode_skips_remaster_dates_and_falls_back():
    from scripts.fixtures_history import REMASTER
    recent, top = listening()
    # May 2025 photos: only the remastered album "came out" then, and its date is not the song's
    with config.override({"songpick.release.max_before_days": 100, "songpick.release.min_candidates": 1}):
        pick = suggest([datetime(2025, 5, 3), datetime(2025, 5, 9)], mode="release", recent=recent, top=top, now=NOW)
    assert pick.mode == "vibe" and pick.candidates
    assert REMASTER not in [(c.artist, c.track) for c in pick.candidates[:1]]
    assert any("remaster or compilation" in n for n in pick.notes)
    assert any("None of the songs you played this year came out around May 2025" in n for n in pick.notes)


def test_suggest_vibe_mode_ignores_dates():
    recent, top = listening()
    pick = suggest([datetime(2021, 7, 3)], mode="vibe", recent=recent, top=top, now=NOW)
    assert pick.mode == "vibe" and pick.period_start is None and pick.period_source == "history"
    assert any("Vibe matching" in n for n in pick.notes)
    assert set(pick.sources) >= {"top:short_term", "top:medium_term", "top:long_term"}


def test_suggest_export_mode_uses_the_export(tmp_path):
    from montage.songpick import load_history
    truth = write_history(tmp_path)
    recent, top = listening()
    dates = [datetime.fromisoformat(s) for s in truth["album_dates"]]  # July 2026: ~4 months before NOW
    pick = suggest(dates, mode="export", export=load_history([tmp_path]), recent=recent, top=top, now=NOW)
    assert (pick.candidates[0].artist, pick.candidates[0].track) == TOP_PICK
    assert pick.mode == "export" and pick.sources[0] == "export" and "top:long_term" not in pick.sources
    with pytest.raises(UserError, match="history files"):
        suggest(dates, mode="export", recent=recent, top=top, now=NOW)
    with pytest.raises(ValueError):
        suggest(dates, mode="nope", recent=recent, top=top, now=NOW)


def test_suggest_without_dates_and_without_data():
    recent, top = listening()
    pick = suggest([None, None], mode="release", recent=recent, top=top, now=NOW)
    assert pick.period_start is None and pick.mode == "vibe" and any("no capture dates" in n for n in pick.notes)
    assert set(pick.sources) >= {"top:short_term", "top:long_term"}
    with pytest.raises(UserError, match="Connect Spotify"):
        suggest([NOW], now=NOW)


def test_suggest_is_deterministic():
    recent, top = listening()
    for mode, dates in (("release", [NOW - timedelta(days=2)]), ("release", [datetime(2021, 7, 3)]),
                        ("vibe", [NOW])):
        assert suggest(dates, mode=mode, recent=recent, top=top, now=NOW).model_dump() == \
            suggest(dates, mode=mode, recent=list(recent), top=dict(top), now=NOW).model_dump()


# ---------- YouTube matching ----------

def r(id_, title, channel, duration):
    return {"id": id_, "title": title, "channel": channel, "duration": duration}


def test_pick_best_prefers_official_audio_and_rejects_versions():
    results = [
        r("aaaaaaaaaa1", "The Fixtures - Summer Anthem (Live at the Arena)", "The Fixtures", 260),
        r("aaaaaaaaaa2", "Summer Anthem (Cover by Someone)", "Someone Else", 197),
        r("aaaaaaaaaa3", "Summer Anthem", "The Fixtures - Topic", 196),
        r("aaaaaaaaaa4", "The Fixtures - Summer Anthem (Official Video)", "TheFixturesVEVO", 230),
    ]
    best, score = youtube.pick_best(results, "Summer Anthem", "The Fixtures", 195_000)
    assert best["id"] == "aaaaaaaaaa3" and score > 1.0
    # without the Topic upload, the official video wins despite its longer intro (live and cover are penalized)
    best2, _ = youtube.pick_best([results[0], results[1], results[3]], "Summer Anthem", "The Fixtures", None)
    assert best2["id"] == "aaaaaaaaaa4"


def test_pick_best_keeps_words_that_are_in_the_title():
    res = [r("bbbbbbbbbb1", "Live Forever", "Oasis - Topic", 276)]
    assert youtube.pick_best(res, "Live Forever", "Oasis", 276_000) is not None


def test_pick_best_rejects_bad_ids_lengths_and_weak_matches():
    assert youtube.pick_best([r("short", "Summer Anthem", "The Fixtures - Topic", 196)], "Summer Anthem", "The Fixtures") is None
    assert youtube.pick_best([r("cccccccccc1", "Summer Anthem", "The Fixtures - Topic", 30)], "Summer Anthem", "The Fixtures") is None
    assert youtube.pick_best([r("cccccccccc2", "Cooking pasta at home", "Chef", 300)], "Summer Anthem", "The Fixtures") is None


def test_find_and_fetch_caches_audio(monkeypatch):
    downloads = []

    def fake_download(video_id, out_dir):
        downloads.append(video_id)
        f = out_dir / f"{video_id}.m4a"
        f.write_bytes(b"fake audio")
        return f

    monkeypatch.setattr(youtube, "search", lambda q, n: [r("dddddddddd1", "Summer Anthem", "The Fixtures - Topic", 196)])
    monkeypatch.setattr(youtube, "_download", fake_download)
    p1, meta = youtube.find_and_fetch("Summer Anthem", "The Fixtures", 195_000)
    p2, _ = youtube.find_and_fetch("Summer Anthem", "The Fixtures", 195_000)
    assert p1 == p2 == youtube.audio_path("dddddddddd1") and p1.read_bytes() == b"fake audio"
    assert downloads == ["dddddddddd1"] and meta["id"] == "dddddddddd1"
    with pytest.raises(ValueError):
        youtube.fetch_audio("../../etc/passwd")


def test_youtube_errors_are_friendly(monkeypatch):
    monkeypatch.setattr(youtube, "search", lambda q, n: [])
    with pytest.raises(UserError, match="Upload the song file"):
        youtube.find_and_fetch("Summer Anthem", "The Fixtures")
    monkeypatch.delenv("MONTAGE_YT_COOKIES", raising=False)
    err = youtube._friendly(RuntimeError("ERROR: Sign in to confirm you're not a bot"))
    assert "blocking downloads" in str(err) and "YOUTUBE_COOKIES" in str(err)
    assert "cookiefile" not in youtube._ydl_opts()


def test_youtube_cookies_from_env(monkeypatch, tmp_path):
    cookies = tmp_path / "c.txt"
    monkeypatch.setenv("MONTAGE_YT_COOKIES", str(cookies))
    assert youtube.cookies_path() is None  # a path that does not exist is ignored
    cookies.write_text("# Netscape HTTP Cookie File\n")
    assert youtube.cookies_path() == str(cookies) and youtube._ydl_opts()["cookiefile"] == str(cookies)
    err = youtube._friendly(RuntimeError("Sign in to confirm you\u2019re not a bot. confirm you're not a bot"))
    assert "may have expired" in str(err)
    url = youtube.search_url("Summer Anthem", "The Fixtures, Guest")
    assert url == "https://www.youtube.com/results?search_query=The+Fixtures+-+Summer+Anthem"


# ---------- app handlers ----------

def test_ui_page_load_and_connect(client_id, monkeypatch):
    monkeypatch.setattr(spotify, "_request", fake_api([]))
    assert songpick_ui.on_page_load({}) == (None, "")
    url, md = songpick_ui.connect(APP)
    from urllib.parse import parse_qs, urlsplit
    state = parse_qs(urlsplit(url).query)["state"][0]
    sp, status = songpick_ui.on_page_load({"spotify_code": "c", "state": state})
    assert sp["token"]["access_token"] == "tok" and len(sp["recent"]) == 50 and "connected" in status
    sp2, msg = songpick_ui.on_page_load({"spotify_code": "c", "state": state})  # reload: link used up
    assert sp2 is None and "expired" in msg
    with pytest.raises(UserError, match="two parts"):
        songpick_ui.paste_code("only-one-part")


def test_ui_suggest_and_use_song(monkeypatch, tmp_path):
    recent, top = listening()
    sp = {"token": token(), "recent": recent, "top": top}
    monkeypatch.setattr(songpick_ui, "_album_dates", lambda media: [datetime.utcnow() - timedelta(days=1)])
    with pytest.raises(UserError, match="Connect Spotify"):
        songpick_ui.suggest(["album"], None, {})
    choices, pick, md = songpick_ui.suggest(["album"], None, sp, "release")
    assert choices[0][1] == "0" and TOP_PICK[1] in choices[0][0] and md.startswith("Songs that came out around")
    _, vpick, vmd = songpick_ui.suggest(["album"], None, sp, "vibe")
    assert vpick.mode == "vibe" and vmd.startswith("Your best picks overall")
    with pytest.raises(UserError, match="history files"):
        songpick_ui.suggest(["album"], None, sp, "export")
    path, md = songpick_ui.use_song("0", pick, youtube_ok=False)
    assert path is None and "tick" in md and "youtube.com/results?search_query=" in md
    cached = tmp_path / "cache" / "youtube_audio" / "abcdefghijk.m4a"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"fake m4a")
    monkeypatch.setattr(youtube, "find_and_fetch",
                        lambda t, a, d: (cached, {"title": t, "channel": "Topic", "duration": 196}))
    path, md = songpick_ui.use_song("0", pick, youtube_ok=True)
    assert "3:16" in md
    served = Path(path)  # a temp copy, never the private cache file itself
    assert served.read_bytes() == b"fake m4a" and served.name == cached.name and served.parent != cached.parent
    assert served.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())

    def blocked(t, a, d):
        raise UserError("YouTube is blocking downloads from this server right now.")
    monkeypatch.setattr(youtube, "find_and_fetch", blocked)
    path, md = songpick_ui.use_song("0", pick, youtube_ok=True)
    assert path is None and "blocking downloads" in md and "Song" in md and "search_query=" in md
    with pytest.raises(UserError, match="Suggest songs"):
        songpick_ui.use_song("0", None, True)



def test_app_never_serves_the_youtube_cache():
    pytest.importorskip("gradio")
    from montage import paths
    from montage.app.gradio_app import served_paths
    allowed, blocked = served_paths()
    cache = str(paths.cache_dir("youtube_audio"))
    assert not any(cache.startswith(a) or a.startswith(cache) for a in allowed) and cache in blocked

# ---------- network (Colab only) ----------

@pytest.mark.network
def test_youtube_search_live():
    pytest.importorskip("yt_dlp")
    res = youtube.search("Kevin MacLeod - Monkeys Spinning Monkeys", 5)
    assert youtube.pick_best(res, "Monkeys Spinning Monkeys", "Kevin MacLeod") is not None
