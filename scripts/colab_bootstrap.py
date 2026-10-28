"""Bootstrap a Colab runtime (idempotent). Run from a notebook cell with %run so Drive and Secrets work:

    !git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null
    %run /content/bac_prototype/scripts/colab_bootstrap.py --branch main

Steps: mount Drive, check montage_data, sync the branch, install requirements (background job
when they changed), verify GPU/FFmpeg (static FFmpeg fallback if zscale is missing), write
/content/montage_env.sh (mode 600, never printed), print a one-line summary.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/AndrewL1698/bac_prototype"
REPO_DIR = Path("/content/bac_prototype")
DRIVE = Path("/content/drive/MyDrive")
DATA_ROOT = DRIVE / "montage_data"
ENV_FILE = Path("/content/montage_env.sh")
REQ_HASH_FILE = Path("/content/.montage_req_hash")
FFMPEG_DIR = Path("/content/bin")
STATIC_FFMPEG_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
SHARE_HELP = """montage_data not found in My Drive.
Keshav owns My Drive/montage_data and shares it (Editor). If you are the other collaborator:
open Drive -> 'Shared with me', right-click montage_data -> Organize -> Add shortcut -> My Drive.
Then re-run this cell. (See KICKOFF.md step 3.)"""


def sh(cmd: list[str] | str, check: bool = False, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, check=check, **kw)


def mount_drive() -> bool:
    mnt = Path("/content/drive")
    if os.path.ismount(mnt):
        return DRIVE.exists()
    if mnt.exists():  # a plain local folder (left by a run without Drive) blocks the mount: move it aside
        shutil.move(str(mnt), f"/content/drive_local_{int(time.time())}")
    try:
        from google.colab import drive  # only works in the notebook kernel (%run)
        drive.mount("/content/drive")
    except Exception as e:  # noqa: BLE001
        print(f"Drive not mounted ({e}). Run this script with %run from a notebook cell, "
              f"or run `from google.colab import drive; drive.mount('/content/drive')` first.")
    return DRIVE.exists()


_SECRETS_REACHABLE = True


def read_secret(name: str) -> str | None:
    global _SECRETS_REACHABLE
    if _SECRETS_REACHABLE:
        try:
            from google.colab import userdata
            v = userdata.get(name)
            if v:
                return v
        except Exception as e:  # noqa: BLE001  (missing secret, no notebook access, not in kernel)
            if "Timeout" in type(e).__name__:  # not run from the Colab UI: don't wait again
                _SECRETS_REACHABLE = False
    return os.environ.get(name) or None


def parse_env(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("export ") and "=" in line:
                k, v = line[len("export "):].split("=", 1)
                out[k] = v.strip().strip("'\"")
    return out


def git_sync(branch: str) -> str:
    token = read_secret("GITHUB_TOKEN")
    url = REPO_URL.replace("https://", f"https://x-access-token:{token}@") if token else REPO_URL
    if not (REPO_DIR / ".git").exists():
        r = sh(["git", "clone", "-q", url, str(REPO_DIR)])
        if r.returncode != 0:
            raise SystemExit("git clone failed (output hidden in case it contains a token)")
    g = ["git", "-C", str(REPO_DIR)]
    sh(g + ["fetch", "-q", url, "+refs/heads/*:refs/remotes/origin/*"]) if token else sh(g + ["fetch", "-q", "origin"])
    if sh(g + ["checkout", "-q", branch]).returncode != 0:
        r = sh(g + ["checkout", "-q", "-b", branch, f"origin/{branch}"])
        if r.returncode != 0:
            raise SystemExit(f"cannot check out branch {branch!r}: {r.stderr.strip()[:200]}")
    sh(g + ["merge", "-q", "--ff-only", f"origin/{branch}"])
    return sh(g + ["rev-parse", "--short", "HEAD"]).stdout.strip()


def req_hash() -> str:
    return hashlib.sha1((REPO_DIR / "requirements.txt").read_bytes()).hexdigest()


def install_cmd() -> str:
    return (f"{sys.executable} -m pip install -q -r {REPO_DIR}/requirements.txt && "
            f"{sys.executable} -m pip install -q -e {REPO_DIR} && echo {req_hash()} > {REQ_HASH_FILE}")


def ensure_requirements(foreground: bool) -> str:
    if REQ_HASH_FILE.exists() and REQ_HASH_FILE.read_text().strip() == req_hash():
        return "ok"
    if foreground:
        r = sh(install_cmd())
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return "FAILED"
        return "installed"
    # Background job so the cell returns quickly; poll with job_status.py pip_install.
    r = sh([sys.executable, str(REPO_DIR / "scripts" / "run_job.py"), "--name", "pip_install", "--force",
            "--", "bash", "-c", install_cmd()], cwd=str(REPO_DIR))
    print(r.stdout.strip() or r.stderr.strip())
    return "installing (job pip_install)"


def ffmpeg_filters(ffmpeg: str) -> tuple[str, bool, bool]:
    v = sh([ffmpeg, "-hide_banner", "-version"])
    if v.returncode != 0:
        return "missing", False, False
    filters = sh([ffmpeg, "-hide_banner", "-filters"]).stdout
    has = lambda f: any(line.split()[1:2] == [f] for line in filters.splitlines() if len(line.split()) > 1)  # noqa: E731
    return v.stdout.split("\n")[0].split(" ")[2], has("zscale"), has("tonemap")


def ensure_ffmpeg() -> tuple[str, bool]:
    if (FFMPEG_DIR / "ffmpeg").exists():
        os.environ["PATH"] = f"{FFMPEG_DIR}:{os.environ['PATH']}"
    ver, zs, tm = ffmpeg_filters(shutil.which("ffmpeg") or "ffmpeg")
    if zs and tm:
        return ver, True
    print("FFmpeg lacks zscale/tonemap; downloading a static build into /content/bin ...")
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    r = sh(f"cd /tmp && curl -sL {STATIC_FFMPEG_URL} -o ffs.tar.xz && tar xf ffs.tar.xz && "
           f"cp ffmpeg-*-static/ffmpeg ffmpeg-*-static/ffprobe {FFMPEG_DIR}/ && rm -rf ffs.tar.xz ffmpeg-*-static")
    if r.returncode != 0:
        print("static FFmpeg download failed; HDR tonemapping will be skipped")
        return ver, False
    os.environ["PATH"] = f"{FFMPEG_DIR}:{os.environ['PATH']}"
    ver, zs, tm = ffmpeg_filters(str(FFMPEG_DIR / "ffmpeg"))
    return ver, zs and tm


def write_env(user: str, api_key: str | None, app_password: str | None = None) -> None:
    lines = [
        f"export MONTAGE_DATA_ROOT={DATA_ROOT}",
        "export MONTAGE_SCRATCH=/content/scratch",
        f"export MONTAGE_USER={user}",
        f"export PATH={FFMPEG_DIR}:$PATH" if (FFMPEG_DIR / "ffmpeg").exists() else "",
    ]
    if api_key:
        lines.append(f"export ANTHROPIC_API_KEY='{api_key}'")
    if app_password:
        lines.append(f"export MONTAGE_APP_PASSWORD='{app_password}'")
    ENV_FILE.touch(mode=0o600, exist_ok=True)
    os.chmod(ENV_FILE, 0o600)
    ENV_FILE.write_text("\n".join(x for x in lines if x) + "\n")
    os.environ.update({"MONTAGE_DATA_ROOT": str(DATA_ROOT), "MONTAGE_SCRATCH": "/content/scratch", "MONTAGE_USER": user})
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    if app_password:
        os.environ["MONTAGE_APP_PASSWORD"] = app_password


def gpu_info() -> tuple[str, bool]:
    r = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    name = r.stdout.strip().replace("\n", "; ") if r.returncode == 0 else "none"
    cuda = sh([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"]).stdout.strip() == "True"
    return name, cuda


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="main")
    ap.add_argument("--foreground-install", action="store_true")
    ap.add_argument("--user", default=None, help="MONTAGE_USER if the Colab secret is unavailable")
    args = ap.parse_args(argv)

    drive_ok = mount_drive() and DATA_ROOT.exists()
    if DRIVE.exists() and not DATA_ROOT.exists():
        print(SHARE_HELP)
        return 1
    if not drive_ok:  # without Drive the app would run on an empty local folder (no demo albums, songs, or weights)
        print("Google Drive is not connected. Rerun this cell from the Colab page and approve the Drive pop-up "
              "(choose the Google account that has montage_data).")
        return 1
    commit = git_sync(args.branch)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    # Secrets are only released to cells run from the Colab UI; when run remotely (colab-mcp)
    # fall back to --user and to values a previous UI run saved in montage_env.sh.
    saved = parse_env(ENV_FILE)
    user = (read_secret("MONTAGE_USER") or args.user or saved.get("MONTAGE_USER") or "anon").strip().lower()
    api_key = read_secret("ANTHROPIC_API_KEY") or saved.get("ANTHROPIC_API_KEY")
    app_password = read_secret("APP_PASSWORD") or saved.get("MONTAGE_APP_PASSWORD")
    Path("/content/scratch").mkdir(parents=True, exist_ok=True)
    ffver, zs_ok = ensure_ffmpeg()
    write_env(user, api_key, app_password)
    req = ensure_requirements(args.foreground_install)
    gpu, cuda = gpu_info()
    print(f"bootstrap: branch {args.branch}@{commit} | user {user} | GPU {gpu} | CUDA {'ok' if cuda else 'NO'} | "
          f"FFmpeg {ffver} zscale {'ok' if zs_ok else 'missing'} | API key {'yes' if api_key else 'no'} | "
          f"app password {'yes' if app_password else 'no'} | "
          f"requirements {req} | {'shared Drive ok' if drive_ok else 'shared Drive MISSING'}")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        raise SystemExit(rc)
