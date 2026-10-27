"""Stop a background job and everything it started.

    python scripts/job_stop.py NAME
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import jobs, paths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: job_stop.py NAME", file=sys.stderr)
        return 2
    print(f"stopped {args[0]}" if jobs.stop(args[0]) else f"{args[0]} was not running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
