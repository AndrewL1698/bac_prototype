"""Shortlist songs from a Spotify history for an album or a date range.

    python -m montage.songpick --history DIR_OR_FILES... --media ALBUM_DIR [--k 5] [--json]
    python -m montage.songpick --history DIR_OR_FILES... --start 2026-07-01 --end 2026-07-14

Prints to your terminal only (the history is personal data: nothing is written or logged).
--media probes capture dates (needs ffprobe for videos).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from montage import config
from montage.songpick import load_history, shortlist
from montage.types import UserError


def _dates(args: argparse.Namespace) -> list[datetime | None]:
    if args.media:
        from montage.ingest.assets import ingest
        return [a.taken_at for a in ingest(args.media)]
    if args.start and args.end:
        return [datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m montage.songpick")
    ap.add_argument("--history", nargs="+", required=True, help="history JSON files or the export folder")
    ap.add_argument("--media", nargs="+", default=None, help="album folder(s): capture dates set the period")
    ap.add_argument("--start", default=None, help="period start YYYY-MM-DD (instead of --media)")
    ap.add_argument("--end", default=None, help="period end YYYY-MM-DD")
    ap.add_argument("--k", type=int, default=None, help="shortlist size")
    ap.add_argument("--json", action="store_true", help="print the SongPick as JSON")
    ap.add_argument("--set", action="append", default=[], help="config override a.b=value")
    args = ap.parse_args(argv)
    if bool(args.start) != bool(args.end):
        ap.error("--start and --end go together")
    if args.set:
        config.set_overrides(args.set)
    try:
        pick = shortlist(load_history(args.history), _dates(args), k=args.k)
    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(pick.model_dump_json(indent=1))
        return 0
    if pick.period_start and pick.period_end:
        print(f"Period: {pick.period_start:%Y-%m-%d} to {pick.period_end:%Y-%m-%d} ({pick.period_source}), "
              f"{pick.n_events_window} of {pick.n_events} plays inside")
    else:
        print(f"Period: whole history ({pick.n_events} plays)")
    for note in pick.notes:
        print(f"note: {note}")
    for i, c in enumerate(pick.candidates, 1):
        print(f"{i}. {c.artist} - {c.track}  (score {c.score:.2f})\n   {c.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
