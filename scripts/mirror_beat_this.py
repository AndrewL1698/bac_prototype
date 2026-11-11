"""Beat This! checkpoint mirror: check the sources, cache the checkpoint on Drive, or publish a mirror.

    python scripts/mirror_beat_this.py --check            # probe every configured source + the Drive copy
    python scripts/mirror_beat_this.py --cache            # fetch the verified checkpoint into DATA_ROOT/weights/
    python scripts/mirror_beat_this.py --upload <user>/beat-this-mirror   # needs `hf auth login` or HF_TOKEN (write)

--upload pushes every checkpoint listed under music.beat_this.sha256 (verified against the configured
hashes) plus a README with the authors' MIT license to a Hugging Face model repo, then prints the
pinned source line to add at the top of music.beat_this.sources in configs/default.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import paths, weights  # noqa: E402
from montage.config import get_config  # noqa: E402

README = """---
license: mit
tags: [beat-tracking, audio]
---
# Beat This! checkpoints (mirror)

Unmodified copies of the Beat This! model checkpoints by the Institute of Computational Perception,
JKU Linz, from https://github.com/CPJKU/beat_this (paper: "Beat This! Accurate Beat Tracking Without
DBN Postprocessing", ISMIR 2024). Mirrored so our app does not depend on a single download host.
The code and the published weights are released by the authors under the MIT license:

{license}

sha256:
{hashes}
"""


def probe(url: str) -> str:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req, timeout=30) as r:
        n = len(r.read(1024))
        return f"HTTP {r.status}, {n} bytes"


def check() -> int:
    cfg = get_config().music.beat_this
    bad = 0
    for s in cfg.sources:
        url = s.format(name=cfg.checkpoint)
        try:
            print(f"ok    {url}  ({probe(url)})")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {url}  ({e})")
    from montage.music.beats import checkpoint_on_drive
    p = checkpoint_on_drive()
    if p.exists():
        good = weights.sha256_of(p) == cfg.sha256[cfg.checkpoint]
        print(f"{'ok  ' if good else 'BAD '}  Drive copy {p} ({'sha256 ok' if good else 'sha256 mismatch'})")
    else:
        print(f"--    no Drive copy at {p}")
    return 1 if bad == len(cfg.sources) else 0


def cache() -> int:
    from montage.music.beats import ensure_checkpoint
    p, source = ensure_checkpoint()
    print(f"cached {p} (source: {source})")
    return 0


def upload(repo_id: str) -> int:
    from huggingface_hub import HfApi, get_token
    token = os.environ.get("HF_TOKEN") or get_token()   # env var, or a saved `hf auth login`
    if not token:
        print("no Hugging Face token: run `hf auth login` (write token) or set HF_TOKEN")
        return 2
    cfg = get_config().music.beat_this
    upstream = cfg.sources[-1]
    lic = urllib.request.urlopen("https://raw.githubusercontent.com/CPJKU/beat_this/main/LICENSE", timeout=30).read()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, sha in cfg.sha256.items():
            weights.fetch_verified(d / f"{name}.ckpt", [upstream.format(name=name)], sha,
                                   local=[paths.weights_dir() / f"beat_this-{name}.ckpt"])
        (d / "LICENSE").write_bytes(lic)
        (d / "README.md").write_text(README.format(
            license=lic.decode().strip(), hashes="\n".join(f"- {n}.ckpt: `{s}`" for n, s in cfg.sha256.items())))
        info = api.upload_folder(folder_path=str(d), repo_id=repo_id, repo_type="model",
                                 commit_message="Mirror Beat This! checkpoints")
    rev = info.oid
    print(f"uploaded to https://huggingface.co/{repo_id} at revision {rev}")
    print("add this as the first entry of music.beat_this.sources in configs/default.yaml:")
    print(f'      - "https://huggingface.co/{repo_id}/resolve/{rev}/{{name}}.ckpt"')
    return 0


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--cache", action="store_true")
    g.add_argument("--upload", metavar="REPO_ID")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.cache:
        return cache()
    return upload(a.upload)


if __name__ == "__main__":
    sys.exit(main())
