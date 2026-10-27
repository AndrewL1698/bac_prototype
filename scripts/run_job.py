"""Launch a long command as a detached background job (Colab).

    python scripts/run_job.py --name NAME [--force] -- CMD...

Writes DATA_ROOT/jobs/<user>/NAME.json and DATA_ROOT/logs/<user>/NAME.log, returns immediately.
Poll with: python scripts/job_status.py NAME
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import jobs, paths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: run_job.py --name NAME [--force] -- CMD...", file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    try:
        st = jobs.launch(args.name, cmd, force=args.force)
    except jobs.JobRunning as e:
        print(f"REFUSED: {e}")
        cur = jobs.read_status(args.name) or {}
        print(f"  state={cur.get('state')} progress={cur.get('progress')} message={cur.get('message')}")
        return 1
    print(f"started job {st['name']} (user {st['user']}, branch {st['branch']}@{st['commit']})")
    print(f"  log: {jobs.log_path(args.name)}")
    print(f"  status: python scripts/job_status.py {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
