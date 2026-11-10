"""Phase 12 song-pick modes: release dates, age-scaled windows, and the Colab helpers they rely on
(YouTube cookies secret, start_app branch switch). Pure Python: no network, no media."""

from __future__ import annotations

import base64
import importlib.util
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from montage import config
from montage.songpick import release as rel
from montage.songpick.period import (age_scaled_days, listening_window, median_date, release_before_days,
                                     release_window)

NOW = datetime(2026, 11, 5, 12, 0, 0)
REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- release dates ----------

@pytest.mark.parametrize("value,prec,want", [
    ("1999-07-21", "day", (date(1999, 7, 21), "day")),
    ("1999-07", "month", (date(1999, 7, 1), "month")),
    ("1999", "year", (date(1999, 1, 1), "year")),
    ("1999-07-21", None, (date(1999, 7, 21), "day")),   # precision inferred
    ("1999", None, (date(1999, 1, 1), "year")),
    ("2001-02-30", "day", (None, None)),                # impossible day
    ("0000", "year", (None, None)),                     # Spotify's "unknown"
    ("1999", "day", (None, None)),                      # claims day precision, has a year
    ("", None, (None, None)),
    (None, None, (None, None)),
])
def test_parse_release(value, prec, want):
    assert rel.parse_release(value, prec) == want


def test_release_interval_and_describe():
    assert rel.interval(date(1999, 1, 1), "year") == (datetime(1999, 1, 1), datetime(1999, 12, 31, 23, 59, 59, 999999))
    assert rel.interval(date(2024, 2, 1), "month")[1].date() == date(2024, 2, 29)
    assert rel.interval(date(2024, 2, 9), "day")[0] == datetime(2024, 2, 9)
    assert rel.describe(date(1999, 1, 1), "year") == "in 1999" and rel.describe(date(2021, 7, 1), "month") == "Jul 2021"


def test_reliable_flags_compilations_and_remasters():
    assert rel.reliable("Summer Album", "Summer Anthem", "album")
    assert not rel.reliable("Hits", "Song", "compilation")
    assert not rel.reliable("Abbey Road (Remastered 2019)", "Come Together", "album")
    assert not rel.reliable("Album", "Song - 2011 Remaster", "album")
    assert not rel.reliable("1989 (Deluxe Edition)", "Song", "album")
    assert not rel.reliable("The Best of Someone", "Song", "album")
    assert rel.reliable("Bestiary", "Editions of You", "album")        # whole words only: no false match
    assert rel.reliable("Best Offer", "Remastering Love", "album")


def test_from_album_handles_missing_fields():
    assert rel.from_album(None, "Song") == {"release_date": None, "release_precision": None, "release_reliable": True}
    got = rel.from_album({"name": "X", "release_date": "2020-05", "album_type": "single"}, "Song")
    assert got == {"release_date": date(2020, 5, 1), "release_precision": "month", "release_reliable": True}


# ---------- windows ----------

def test_age_scaled_days_interpolates_and_clamps():
    table = [[0, 21], [1, 30], [5, 90], [10, 180]]
    assert age_scaled_days(0, table) == 21 and age_scaled_days(-1, table) == 21
    assert age_scaled_days(0.5, table) == pytest.approx(25.5)
    assert age_scaled_days(3, table) == pytest.approx(60)
    assert age_scaled_days(10, table) == 180 and age_scaled_days(40, table) == 180


def test_median_date_ignores_strays_and_missing():
    dates = [datetime(2021, 7, d) for d in range(1, 11)] * 3 + [datetime(2010, 1, 1), None]
    assert median_date(dates).date() in (date(2021, 7, 5), date(2021, 7, 6))
    assert median_date([None]) is None and median_date([]) is None
    assert median_date([datetime(2021, 1, 1), datetime(2021, 1, 3)]) == datetime(2021, 1, 2)


def test_release_window_is_asymmetric_and_scales_with_age():
    cfg = config.get_config().songpick.release
    recent = [NOW - timedelta(days=3)]
    old = [datetime(2021, 7, 3), datetime(2021, 7, 10)]
    w_recent, w_old = release_window(recent, NOW), release_window(old, NOW)
    before_recent = release_before_days(recent, NOW)
    assert before_recent == pytest.approx(cfg.before_days_by_age[0][1], abs=1)
    assert (recent[0] - w_recent[0]).days == pytest.approx(before_recent, abs=1)
    assert (w_recent[1] - recent[0]).days == pytest.approx(cfg.after_frac * before_recent, abs=1)
    assert (w_old[1] - w_old[0]) > (w_recent[1] - w_recent[0])                 # older photos: wider window
    assert w_old[0] <= old[0] and w_old[1] >= old[1]                           # always covers the album
    wide = release_window(old, NOW, scale=2.0)
    assert wide[0] < w_old[0] and wide[1] > w_old[1]
    assert release_window([None], NOW) is None


def test_listening_window_widths_by_age():
    table = config.get_config().songpick.window_by_age_days
    for years in (0, 1, 5, 10):
        when = NOW - timedelta(days=365.25 * years)
        start, end = listening_window([when], NOW)
        assert (end - start).total_seconds() / 86400 == pytest.approx(age_scaled_days(years, table), abs=0.1)  # age in whole days
    span = [datetime(2026, 7, 1), datetime(2026, 7, 31), datetime(2026, 7, 2)]  # a month-long album, 4 months ago
    start, end = listening_window(span, NOW)
    assert start <= datetime(2026, 7, 1) and end >= datetime(2026, 7, 31)


# ---------- Colab helpers ----------

COOKIES = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tPREF\tf1=1\n"


def test_bootstrap_cookies_secret_formats(tmp_path, monkeypatch):
    boot = _load("colab_bootstrap")
    assert boot.cookies_text(COOKIES) == COOKIES
    assert boot.cookies_text(base64.b64encode(COOKIES.encode()).decode()) == COOKIES
    b64 = base64.b64encode(COOKIES.encode()).decode()
    assert boot.cookies_text(b64[:20] + "\n" + b64[20:]) == COOKIES     # wrapped base64 still works
    assert boot.cookies_text("not cookies") is None and boot.cookies_text("") is None
    monkeypatch.setattr(boot, "YT_COOKIES_FILE", tmp_path / "yt.txt")
    out = boot.write_cookies(COOKIES)
    assert out.read_text().startswith("# Netscape")
    if sys.platform != "win32":  # Windows chmod only toggles read-only
        assert (out.stat().st_mode & 0o777) == 0o600
    assert boot.write_cookies("junk") is None


def test_start_app_reexec_guard():
    start = _load("start_app")
    assert start.needs_reexec(True, {}) is True
    assert start.needs_reexec(True, {start.REEXEC_ENV: "1"}) is False   # the re-run itself never loops
    assert start.needs_reexec(False, {}) is False
