"""Phase 0 acceptance: background job runner."""

from __future__ import annotations

import sys
import time

import pytest

from montage import jobs


def _wait(name: str, user: str | None = None, timeout: float = 20.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = jobs.read_status(name, user)
        if st and st["state"] in ("done", "failed"):
            return st
        time.sleep(0.1)
    raise AssertionError(f"job {name} did not finish: {jobs.read_status(name, user)}")


def test_job_runner(fresh_root):
    cmd = [sys.executable, "-c", "import time; [time.sleep(0.2) for _ in range(10)]"]
    st = jobs.launch("sleepy", cmd, env_file=None)
    assert st["state"] == "running"
    # running: re-launch without force is refused
    time.sleep(0.5)
    assert jobs.read_status("sleepy")["state"] == "running"
    with pytest.raises(jobs.JobRunning):
        jobs.launch("sleepy", cmd, env_file=None)
    st = _wait("sleepy")
    assert st["state"] == "done" and st["exit_code"] == 0 and st["progress"] == 1.0
    assert st["branch"] and st["commit"]

    jobs.launch("bad", [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], env_file=None)
    st = _wait("bad")
    assert st["state"] == "failed" and st["exit_code"] == 3
    assert "boom" in jobs.log_path("bad").read_text()


def test_job_report(fresh_root):
    code = "from montage import jobs; jobs.report(0.5, 'halfway', force=True)"
    jobs.launch("rep", [sys.executable, "-c", code], env_file=None)
    st = _wait("rep")
    assert st["state"] == "done"
    assert "halfway" in jobs.log_path("rep").read_text()


def test_job_namespacing(fresh_root, monkeypatch):
    cmd_a = [sys.executable, "-c", "print('from a')"]
    cmd_b = [sys.executable, "-c", "print('from b')"]
    monkeypatch.setenv("MONTAGE_USER", "a")
    jobs.launch("same", cmd_a, env_file=None)
    monkeypatch.setenv("MONTAGE_USER", "b")
    jobs.launch("same", cmd_b, env_file=None)
    sa, sb = _wait("same", "a"), _wait("same", "b")
    assert sa["user"] == "a" and sb["user"] == "b"
    assert sa["cmd"] == cmd_a and sb["cmd"] == cmd_b
    assert "from a" in jobs.log_path("same", "a").read_text()
    assert "from b" in jobs.log_path("same", "b").read_text()


def test_report_noop_outside_job(monkeypatch):
    monkeypatch.delenv("MONTAGE_JOB_NAME", raising=False)
    jobs.report(0.3, "nothing")  # must not raise or write
    assert not jobs.is_job()
