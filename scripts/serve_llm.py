"""Serve the open-weight director model on Colab with vLLM (OpenAI-compatible API). Run it as a job:

    !python scripts/run_job.py --name llm -- python scripts/serve_llm.py
    !python scripts/job_status.py llm          # "ready" once the model answers

Then point the director at it: `--set director.provider=openai_compat` (CLI), or set `director.provider`
in a MONTAGE_CONFIG overlay for the app. Model, port, and memory share come from `director.openai_compat`.

vLLM is installed into its own venv (default /content/vllm_env) so Colab's CUDA torch is never replaced.
Weights download to scratch (Hugging Face cache), never to the repo.
"""

from __future__ import annotations

import argparse
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


def ensure_venv(venv: Path, version: str) -> Path:
    exe = venv / "bin" / "vllm"
    if exe.exists():
        return exe
    jobs.report(0.05, f"installing vllm=={version} into {venv}", force=True)
    uv = shutil.which("uv")
    if uv:
        subprocess.check_call([uv, "venv", "-q", "--python", sys.executable, str(venv)])
        subprocess.check_call([uv, "pip", "install", "-q", "--python", str(venv / "bin" / "python"),
                               f"vllm=={version}"])
    else:
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
        subprocess.check_call([str(venv / "bin" / "pip"), "install", "-q", f"vllm=={version}"])
    return exe


def ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Hugging Face repo id (default: director.openai_compat.model)")
    ap.add_argument("--venv", default="/content/vllm_env")
    ap.add_argument("--set", action="append", default=[],
                    help="config override a.b=value, e.g. director.openai_compat.serve.gpu_memory_utilization=0.8")
    args = ap.parse_args(argv)
    if args.set:
        set_overrides(args.set)
    oc = get_config().director.openai_compat
    args.model = args.model or oc.model
    if ready(oc.base_url):
        print(f"a server already answers at {oc.base_url}; nothing to do")
        return 0
    url = urlparse(oc.base_url)
    exe = ensure_venv(Path(args.venv), oc.serve.vllm_version)
    env = {**os.environ, "HF_HOME": str(paths.scratch() / "hf")}
    cmd = [str(exe), "serve", args.model, "--host", url.hostname or "127.0.0.1", "--port", str(url.port or 8000),
           "--served-model-name", oc.model, "--gpu-memory-utilization", str(oc.serve.gpu_memory_utilization),
           "--max-model-len", str(oc.serve.max_model_len)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env)
    t0 = time.time()
    while not ready(oc.base_url):
        if proc.poll() is not None:
            print(f"vllm exited with code {proc.returncode} before it was ready", file=sys.stderr)
            return proc.returncode or 1
        if time.time() - t0 > oc.serve.ready_timeout_s:
            proc.terminate()
            print("vllm did not become ready in time", file=sys.stderr)
            return 1
        jobs.report(0.5, f"loading {args.model} ({time.time() - t0:.0f} s)")
        time.sleep(5)
    jobs.report(1.0, f"ready: {oc.model} at {oc.base_url}", force=True)
    print(f"ready: {oc.model} at {oc.base_url}", flush=True)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
