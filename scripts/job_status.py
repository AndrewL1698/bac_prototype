"""Show background job status.

    python scripts/job_status.py NAME          state, progress, message, elapsed, last 20 log lines
    python scripts/job_status.py               table of this user's jobs
    python scripts/job_status.py --all-users   table of everyone's jobs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import jobs, paths  # noqa: E402


def _elapsed(st: dict) -> str:
    try:
        t0 = datetime.fromisoformat(st["started"])
        t1 = datetime.now() if st.get("state") == "running" else datetime.fromisoformat(st["updated"])
        s = int((t1 - t0).total_seconds())
        return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"
    except Exception:
        return "?"


def _state(st: dict) -> str:
    s = st.get("state", "?")
    if s == "running" and not jobs.is_active(st):
        return "running?(process gone)"
    return s


def tail(path: Path, n: int = 20) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except FileNotFoundError:
        return ["(no log yet)"]
    # collapse carriage-return progress bars to their last state
    return [ln.split("\r")[-1] for ln in lines[-n:]]


def show_one(name: str, for_user: str | None) -> int:
    st = jobs.read_status(name, for_user)
    if not st:
        print(f"no job named {name!r} for user {for_user or paths.user()}")
        return 1
    print(f"{st['name']} [{st.get('user')}] state={_state(st)} progress={st.get('progress', 0):.0%} "
          f"elapsed={_elapsed(st)} branch={st.get('branch')}@{st.get('commit')}")
    print(f"  message: {st.get('message', '')}")
    if "exit_code" in st:
        print(f"  exit_code: {st['exit_code']}")
    url = paths.jobs_dir(st.get("user")) / f"{name}.url"
    if url.exists():
        print(f"  url: {url.read_text().strip()}")
    print("  --- log tail ---")
    for ln in tail(jobs.log_path(name, st.get("user"))):
        print("  " + ln)
    return 0


def show_table(all_users: bool) -> int:
    root = paths.data_root() / "jobs"
    files = sorted(root.glob("*/*.json") if all_users else paths.jobs_dir().glob("*.json"))
    if not files:
        print("no jobs")
        return 0
    print(f"{'user':10} {'name':20} {'state':24} {'prog':>5} {'elapsed':>10}  message")
    for p in files:
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        print(f"{st.get('user', '?'):10} {st.get('name', p.stem):20} {_state(st):24} "
              f"{st.get('progress', 0):>5.0%} {_elapsed(st):>10}  {str(st.get('message', ''))[:60]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--all-users", action="store_true")
    ap.add_argument("--user", default=None, help="look at another user's job")
    args = ap.parse_args(argv)
    if args.name:
        return show_one(args.name, args.user)
    return show_table(args.all_users)


if __name__ == "__main__":
    sys.exit(main())
