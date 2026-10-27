"""Command line: generate / recut / export.

    python -m montage.cli generate --media DIR --song FILE --brief "..." [--seed 0] [--export] [--system full]
    python -m montage.cli recut --session ID --note "too fast in the middle"
    python -m montage.cli export --run RUN_ID
Any command accepts --set a.b=value (config overrides).
"""

from __future__ import annotations

import argparse
import json
import sys

from montage import config, paths
from montage.types import UserError


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser(prog="montage")
    ap.add_argument("--set", action="append", default=[], help="config override a.b=value")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--media", nargs="+", required=True)
    g.add_argument("--song", required=True)
    g.add_argument("--brief", default="")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--export", action="store_true")
    g.add_argument("--system", default="full")
    r = sub.add_parser("recut")
    r.add_argument("--session", required=True)
    r.add_argument("--note", default=None)
    r.add_argument("--seed", type=int, default=None)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    if args.set:
        config.set_overrides(args.set)
    from montage import pipeline
    try:
        if args.cmd == "generate":
            res = pipeline.generate(args.media, args.song, args.brief, seed=args.seed, system=args.system,
                                    progress=lambda f, m: print(f"[{f:4.0%}] {m}", flush=True))
            out = json.loads(res.model_dump_json())
            out["session_id"] = pipeline.session_for_run(res.run_id).session_id
            if args.export:
                out["export_path"] = pipeline.export(res.run_id)
        elif args.cmd == "recut":
            res = pipeline.recut(args.session, note=args.note, seed=args.seed,
                                 progress=lambda f, m: print(f"[{f:4.0%}] {m}", flush=True))
            out = json.loads(res.model_dump_json())
        else:
            out = {"export_path": pipeline.export(args.run)}
    except UserError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
