"""(Re)start the demo app on Colab in one cell and print its link. Run it from the Colab UI so Secrets load:

    !git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null
    %run /content/bac_prototype/scripts/start_app.py

Bootstraps (Drive, branch, requirements, Secrets incl. the optional APP_PASSWORD), stops a running app,
starts the `app` job, and waits for the share link. This is a human-run convenience cell, so it waits up
to --wait seconds (default 120); if the link is not ready by then it prints the command to fetch it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _job_state(data_root: Path, user: str, name: str) -> str | None:
    try:
        return json.loads((data_root / "jobs" / user / f"{name}.json").read_text()).get("state")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="main")
    ap.add_argument("--user", default=None)
    ap.add_argument("--wait", type=float, default=120.0)
    args = ap.parse_args(argv)
    import colab_bootstrap
    boot = ["--branch", args.branch] + (["--user", args.user] if args.user else [])
    if colab_bootstrap.main(boot):
        return 1
    data_root = Path(os.environ["MONTAGE_DATA_ROOT"])
    user = os.environ.get("MONTAGE_USER", "anon")
    deadline = time.time() + args.wait
    py = sys.executable
    # Fresh processes for everything else, so a git pull is always picked up (no stale imports here).
    while _job_state(data_root, user, "pip_install") == "running" and time.time() < deadline:
        time.sleep(2)
    subprocess.run([py, str(REPO / "scripts" / "job_stop.py"), "app"], cwd=REPO)
    url_file = data_root / "jobs" / user / "app.url"
    url_file.unlink(missing_ok=True)
    r = subprocess.run([py, str(REPO / "scripts" / "run_job.py"), "--name", "app", "--force", "--",
                        py, "-m", "montage.app.gradio_app", "--share"], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        return 1
    print("starting the app", end="", flush=True)
    while not url_file.exists() and time.time() < deadline:
        print(".", end="", flush=True)
        time.sleep(2)
    print()
    pw = "on" if os.environ.get("MONTAGE_APP_PASSWORD") else "off (add Colab Secret APP_PASSWORD to turn it on)"
    if url_file.exists():
        print(f"APP LINK: {url_file.read_text().strip()}\npassword: {pw}; username: "
              f"{os.environ.get('MONTAGE_APP_USER') or 'montage'}\nKeep this Colab tab open while you use the app.")
    else:
        print(f"Still starting. In a minute run: !python {REPO}/scripts/job_status.py app")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        raise SystemExit(rc)
