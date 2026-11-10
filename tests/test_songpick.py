"""Phase 12 acceptance (fallback stage): history parsing, album period, period affinity, shortlist.
Pure Python on a synthetic history (scripts/fixtures_history.py); no network, no weights, no media."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from montage import config
from montage.songpick import load_history, shortlist
from montage.songpick.affinity import play_weight, score_tracks, track_key
from montage.songpick.history import parse_row
from montage.songpick.period import album_span, padded
from montage.types import ListeningEvent, UserError
from scripts.fixtures_history import write_history


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    d = tmp_path_factory.mktemp("history")
    truth = write_history(d)
    return d, truth


@pytest.fixture(scope="module")
def events(history):
    return load_history([history[0]])


NOW = datetime(2026, 11, 5, 12, 0, 0)


def _dates(truth):
    return [datetime.fromisoformat(s) for s in truth["album_dates"]]


def ev(ts: str, artist="A", track="T", ms=200_000, skipped=False) -> ListeningEvent:
    return ListeningEvent(ts=datetime.fromisoformat(ts), ms_played=ms, track=track, artist=artist, skipped=skipped)


# ---------- history ----------

def test_parse_extended_row_keeps_only_contract_fields():
    row = {"ts": "2026-07-04T18:22:31Z", "ms_played": 201000, "ip_addr": "1.2.3.4", "conn_country": "US",
           "master_metadata_track_name": "Song", "master_metadata_album_artist_name": "Band",
           "master_metadata_album_album_name": "Album", "spotify_track_uri": "spotify:track:abc",
           "reason_end": "trackdone", "skipped": None}
    e = parse_row(row)
    assert e is not None and e.ts == datetime(2026, 7, 4, 18, 22, 31) and e.ts.tzinfo is None
    assert (e.track, e.artist, e.album, e.uri, e.skipped) == ("Song", "Band", "Album", "spotify:track:abc", False)
    assert "1.2.3.4" not in e.model_dump_json()


def test_parse_drops_podcasts_and_bad_rows():
    assert parse_row({"ts": "2026-01-01T00:00:00Z", "ms_played": 9, "master_metadata_track_name": None,
                      "episode_name": "Ep", "spotify_episode_uri": "spotify:episode:x"}) is None
    assert parse_row({"ts": "not a date", "ms_played": 9, "master_metadata_track_name": "S",
                      "master_metadata_album_artist_name": "B"}) is None
    assert parse_row("junk") is None


def test_parse_account_format_and_fwdbtn_skip():
    e = parse_row({"endTime": "2025-03-02 21:05", "artistName": "B", "trackName": "S", "msPlayed": 1000})
    assert e is not None and e.ts == datetime(2025, 3, 2, 21, 5) and e.uri is None
    s = parse_row({"ts": "2026-01-01T00:00:00Z", "ms_played": 5000, "master_metadata_track_name": "S",
                   "master_metadata_album_artist_name": "B", "reason_end": "fwdbtn"})
    assert s is not None and s.skipped


def test_load_history_folder_merges_formats(history, events):
    d, truth = history
    # every synthetic music row (podcasts dropped) plus the account-format file
    n_podcasts = 40
    assert len(events) == truth["n_rows"] - n_podcasts + 30
    assert events == sorted(events, key=lambda e: (e.ts, e.artist, e.track))
    # loading the same files twice does not double-count plays
    again = load_history([d, *sorted(d.glob("*.json"))])
    assert len(again) == len(events)


def test_load_history_errors(tmp_path):
    with pytest.raises(UserError):
        load_history([tmp_path])
    bad = tmp_path / "Streaming_History_Audio_x.json"
    bad.write_text("{not json")
    with pytest.raises(UserError):
        load_history([bad])
    bad.write_text(json.dumps({"a": 1}))
    with pytest.raises(UserError):
        load_history([bad])


# ---------- period ----------

def test_album_span_trims_and_padded_pads():
    dates = [datetime(2026, 7, d) for d in range(3, 11)] * 3 + [datetime(2019, 1, 1)]
    span = album_span(dates)
    assert span == (datetime(2026, 7, 3), datetime(2026, 7, 10))  # the stray 2019 photo is trimmed
    pad = config.get_config().songpick.pad_days
    start, end = padded(span, pad)
    assert (span[0] - start).days == pad and (end - span[1]).days == pad


def test_padded_minimum_length_and_no_dates():
    with config.override({"songpick.min_window_days": 3}):
        one = datetime(2026, 7, 4, 12)
        start, end = padded(album_span([one]), 0)
        assert (end - start).total_seconds() == pytest.approx(3 * 86400)
        assert start < one < end
    assert album_span([None, None]) is None
    assert album_span([]) is None


# ---------- affinity ----------

def test_play_weight():
    cfg = config.get_config().songpick
    assert play_weight(ev("2026-01-01", ms=cfg.full_play_ms * 2)) == 1.0
    assert play_weight(ev("2026-01-01", ms=cfg.full_play_ms // 2)) == pytest.approx(0.5)
    assert play_weight(ev("2026-01-01", ms=cfg.min_play_ms - 1)) == 0.0
    assert play_weight(ev("2026-01-01", ms=cfg.full_play_ms, skipped=True)) == pytest.approx(1 - cfg.skip_penalty)


def test_lift_prefers_song_of_the_period():
    period = (datetime(2026, 7, 1), datetime(2026, 7, 14))
    evs = []
    for m in range(1, 13):  # favourite: 4 plays every month, 4 of them in the period
        evs += [ev(f"2026-{m:02d}-0{d}", "Fav", "Always") for d in (2, 4, 6, 8)]
    evs += [ev(f"2026-07-0{d}", "Sum", "Mer") for d in (2, 3, 5)]  # 3 plays, all in the period
    evs.sort(key=lambda e: e.ts)
    c = {x.key: x for x in score_tracks(evs, period)}
    fav, summer = c[track_key("Fav", "Always")], c[track_key("Sum", "Mer")]
    assert summer.lift > 1 > fav.lift
    assert summer.affinity > fav.affinity
    assert max(x.affinity for x in c.values()) == 1.0


def test_track_key_merges_case_and_spacing():
    assert track_key("The  Band", "Song ") == track_key("the band", "song")


# ---------- shortlist (acceptance) ----------

def test_shortlist_finds_song_of_that_summer(events, history):
    truth = history[1]
    pick = shortlist(events, _dates(truth), now=NOW)
    names = [(c.artist, c.track) for c in pick.candidates]
    assert pick.period_source == "album" and pick.period_start < datetime(2026, 7, 3)
    assert names[0] == tuple(truth["top_pick"])
    assert tuple(truth["skipped"]) not in names
    assert len(pick.candidates) == config.get_config().songpick.shortlist_size
    top = pick.candidates[0]
    assert top.lift > 1.5 and "more than usual" in top.reason and top.score == top.affinity
    assert pick.vibe_source == "none" and all(c.vibe is None for c in pick.candidates)
    assert pick.n_events_window < pick.n_events


def test_shortlist_is_deterministic(events, history):
    a = shortlist(events, _dates(history[1]))
    b = shortlist(list(events), _dates(history[1]))
    assert a.model_dump() == b.model_dump()


def test_shortlist_without_dates_uses_whole_history(events, history):
    pick = shortlist(events, [None] * 10)
    assert pick.period_source == "history" and pick.period_start is None
    assert pick.n_events_window == pick.n_events and pick.notes
    names = [(c.artist, c.track) for c in pick.candidates]
    assert names[0] == tuple(history[1]["favourite"])  # overall favourite wins without a period
    assert all(c.lift == 1.0 for c in pick.candidates)


def test_shortlist_widens_period_then_falls_back(events):
    # album weeks before the history starts: the window widens until it reaches the first plays
    widened = shortlist(events, [datetime(2024, 11, 25)], now=NOW)
    assert widened.period_source == "album_widened" and widened.notes
    # album years before the history: falls back to the whole history
    old = shortlist(events, [datetime(2020, 6, 1)], now=NOW)
    assert old.period_source == "history" and old.candidates


def test_shortlist_single_plays_and_errors():
    evs = [ev(f"2026-07-0{d}T12:00:00", f"Artist {d}", f"Track {d}") for d in range(1, 10)]
    with config.override({"songpick.min_window_events": 1}):
        pick = shortlist(evs, [datetime(2026, 7, 5)])
    assert pick.period_source == "album" and pick.candidates and any("single plays" in n for n in pick.notes)
    with pytest.raises(UserError):
        shortlist([], [datetime(2026, 7, 5)])
    with pytest.raises(UserError):  # only sub-30 s plays
        shortlist([ev("2026-07-05T12:00:00", ms=1000)], [datetime(2026, 7, 5)])


def test_cli_prints_shortlist(history, capsys):
    from montage.songpick.__main__ import main
    d, truth = history
    assert main(["--history", str(d), "--start", "2026-07-03", "--end", "2026-07-10", "--k", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n1. ") + out.startswith("1. ") == 1 and truth["top_pick"][1] in out.split("\n2. ")[0]
    assert main(["--history", str(d / "missing")]) == 2
