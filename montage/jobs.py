"""Background jobs on Colab: launch detached, track status JSON, report progress.

Status file DATA_ROOT/jobs/<user>/<name>.json:
    {name, user, cmd, branch, commit, pid, state: running|done|failed, started, updated,
     progress: 0-1, message, [child_pid, exit_code]}
Log file DATA_ROOT/logs/<user>/<name>.log.

`launch()` writes the initial status, then starts a detached watcher process
(`python -m montage.jobs --watch NAME -- CMD...`, new session) that runs CMD, waits,
and writes done/failed. Long-running code calls `report(progress, message)`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from montage import paths

STALE_START_S = 30.0


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def status_path(name: str, for_user: str | None = None) -> Path:
    return paths.jobs_dir(for_user) / f"{name}.json"


def log_path(name: str, for_user: str | None = None) -> Path:
    return paths.logs_dir(for_user) / f"{name}.log"


def read_status(name: str, for_user: str | None = None) -> dict | None:
    p = status_path(name, for_user)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_status(st: dict, for_user: str | None = None) -> None:
    target = status_path(st["name"], for_user or st.get("user"))
    for attempt in range(20):  # Windows refuses os.replace while a reader has the file open
        try:
            paths.atomic_write(target, json.dumps(st, indent=1))
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_active(st: dict | None) -> bool:
    """True if the status says running and the watcher is alive (or just started)."""
    if not st or st.get("state") != "running":
        return False
    if st.get("pid"):
        return pid_alive(st["pid"])
    try:
        started = datetime.fromisoformat(st.get("started", "")).timestamp()
    except ValueError:
        return False
    return time.time() - started < STALE_START_S


def git_info(repo: Path = paths.REPO_ROOT) -> tuple[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return run("rev-parse", "--abbrev-ref", "HEAD") or "?", run("rev-parse", "--short", "HEAD") or "?"


class JobRunning(RuntimeError):
    pass


def other_users_running(name: str) -> list[dict]:
    out = []
    root = paths.data_root() / "jobs"
    if not root.exists():
        return out
    for p in root.glob(f"*/{name}.json"):
        if p.parent.name == paths.user():
            continue
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if st.get("state") == "running":
            out.append(st)
    return out


def launch(name: str, cmd: list[str], force: bool = False, env_file: str | Path | None = paths.COLAB_ENV_FILE) -> dict:
    """Start CMD detached under the job runner. Raises JobRunning if already active (unless force)."""
    if not cmd:
        raise ValueError("empty command")
    if env_file:
        paths.load_colab_env(env_file)
    prev = read_status(name)
    if is_active(prev) and not force:
        raise JobRunning(f"job {name!r} is already running (pid {prev.get('pid')}); use --force to start anyway")
    for st in other_users_running(name):
        print(f"WARNING: user {st.get('user')} also has a running job named {name!r} "
              f"(started {st.get('started')}); this may duplicate work on the shared Drive.", file=sys.stderr)
    branch, commit = git_info()
    st = {
        "name": name, "user": paths.user(), "cmd": cmd, "branch": branch, "commit": commit,
        "pid": None, "state": "running", "started": now_iso(), "updated": now_iso(),
        "progress": 0.0, "message": "starting",
    }
    write_status(st)
    log = log_path(name)
    env = os.environ.copy()
    env["MONTAGE_JOB_NAME"] = name
    env["MONTAGE_USER"] = paths.user()
    env["MONTAGE_DATA_ROOT"] = str(paths.data_root())
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(paths.REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(log, "wb") as logf:
        subprocess.Popen(
            [sys.executable, "-m", "montage.jobs", "--watch", name, "--", *cmd],
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, cwd=str(paths.REPO_ROOT), close_fds=True, **kwargs,
        )
    return st


def stop(name: str, for_user: str | None = None, timeout: float = 15.0) -> bool:
    """Stop a running job and everything it started (its process group). Returns True if one was running."""
    st = read_status(name, for_user)
    if not is_active(st) or not st.get("pid"):
        return False
    pid = int(st["pid"])
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError):
                break
            t0 = time.time()
            while pid_alive(pid) and time.time() - t0 < timeout:
                time.sleep(0.2)
            if not pid_alive(pid):
                break
    st = read_status(name, for_user) or st
    st.update(state="failed", message="stopped", exit_code=-15, updated=now_iso())
    write_status(st, for_user)
    return True


def _watch(name: str, cmd: list[str]) -> int:
    st = read_status(name) or {"name": name, "user": paths.user(), "cmd": cmd, "started": now_iso()}
    st.update(pid=os.getpid(), state="running", updated=now_iso())
    write_status(st)
    try:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    except OSError as e:
        print(f"[job] failed to start: {e}", flush=True)
        st.update(state="failed", exit_code=127, message=str(e), updated=now_iso())
        write_status(st)
        return 127
    st = read_status(name) or st
    st.update(child_pid=child.pid, updated=now_iso())
    write_status(st)
    rc = child.wait()
    st = read_status(name) or st
    if rc == 0:
        st.update(state="done", progress=1.0, message=st.get("message") if st.get("message") != "starting" else "done")
    else:
        st.update(state="failed", message=f"exit code {rc}")
    st.update(exit_code=rc, updated=now_iso())
    write_status(st)
    print(f"[job] {name} finished with exit code {rc}", flush=True)
    return rc


# ---------- progress reporting from inside a job ----------

_last_report = 0.0
REPORT_EVERY_S = 5.0


def is_job() -> bool:
    return bool(os.environ.get("MONTAGE_JOB_NAME"))


def report(progress: float, message: str = "", force: bool = False) -> None:
    """Update this job's status (at most every 5 s). No-op outside a job."""
    global _last_report
    name = os.environ.get("MONTAGE_JOB_NAME")
    if not name:
        return
    now = time.time()
    if not force and now - _last_report < REPORT_EVERY_S:
        return
    _last_report = now
    st = read_status(name) or {"name": name, "user": paths.user(), "state": "running"}
    st.update(progress=round(max(0.0, min(1.0, float(progress))), 4), message=message, updated=now_iso())
    try:
        write_status(st)
    except OSError:
        pass
    print(f"[progress {progress:.0%}] {message}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "--watch" and "--" in args:
        i = args.index("--")
        sys.exit(_watch(args[1], args[i + 1:]))
    print("usage: python -m montage.jobs --watch NAME -- CMD...", file=sys.stderr)
    sys.exit(2)
