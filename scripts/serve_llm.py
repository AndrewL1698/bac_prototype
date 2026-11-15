"""Serve the open-weight director model on Colab with Ollama (OpenAI-compatible API). Run it as a job:

    !python scripts/run_job.py --name llm -- python scripts/serve_llm.py
    !python scripts/job_status.py llm          # "ready" once the model answers

Then point the director at it: `--set director.provider=openai_compat` (CLI), or set `director.provider`
in a MONTAGE_CONFIG overlay for the app. Model, version, and address come from `director.openai_compat`.
Weights download to scratch (local VM disk), never to Drive or the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import jobs, paths  # noqa: E402
from montage.config import get_config, set_overrides  # noqa: E402


def ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="config override a.b=value")
    args = ap.parse_args(argv)
    if args.set:
        set_overrides(args.set)
    oc = get_config().director.openai_compat
    url = urlparse(oc.base_url)
    env = {**os.environ, "OLLAMA_HOST": url.netloc, "OLLAMA_MODELS": str(paths.scratch() / "ollama"),
           "OLLAMA_VERSION": oc.serve.ollama_version, "OLLAMA_KEEP_ALIVE": "-1"}  # never unload: a cold load beats the 30 s director timeout
    if not shutil.which("ollama"):
        jobs.report(0.05, f"installing ollama {oc.serve.ollama_version}", force=True)
        subprocess.run("apt-get -qq install -y zstd", shell=True, env=env)  # the installer unpacks a .tar.zst
        subprocess.check_call("curl -fsSL https://ollama.com/install.sh | sh", shell=True, env=env)
    proc = None if ready(oc.base_url) else subprocess.Popen(["ollama", "serve"], env=env)
    t0 = time.time()
    while not ready(oc.base_url):
        if proc.poll() is not None or time.time() - t0 > oc.serve.ready_timeout_s:
            print("ollama serve did not start", file=sys.stderr)
            return 1
        time.sleep(2)
    jobs.report(0.3, f"pulling {oc.model}", force=True)
    subprocess.check_call(["ollama", "pull", oc.model], env=env)
    req = urllib.request.Request(f"{url.scheme}://{url.netloc}/api/generate", data=json.dumps({"model": oc.model}).encode())
    urllib.request.urlopen(req, timeout=oc.serve.ready_timeout_s).read()  # an empty prompt loads the model into GPU memory
    jobs.report(1.0, f"ready: {oc.model} at {oc.base_url}", force=True)
    print(f"ready: {oc.model} at {oc.base_url}", flush=True)
    return proc.wait() if proc else 0


if __name__ == "__main__":
    raise SystemExit(main())
