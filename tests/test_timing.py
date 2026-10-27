"""Timing acceptance (slow; run as a Colab job with MONTAGE_TIMING=1).

Phase 3: 60 photos + 20 videos x 30 s, cold cache, features under 90 s on the A100.
"""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import gpu_available

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("MONTAGE_TIMING") != "1",
                                                  reason="set MONTAGE_TIMING=1 to run timing tests")]


def make_album(d, n_photos=60, n_videos=20, video_s=30):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_photos):
        im = Image.new("RGB", (1600, 1200), tuple(int(x) for x in rng.integers(0, 255, 3)))
        dr = ImageDraw.Draw(im)
        for _ in range(20):
            x, y, r = rng.integers(0, 1600), rng.integers(0, 1200), rng.integers(20, 200)
            dr.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
        im.save(d / f"p{i:02d}.jpg", quality=90)
    srcs = ["testsrc2", "smptehdbars", "rgbtestsrc", "mandelbrot"]
    for i in range(n_videos):
        src = srcs[i % len(srcs)]
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"{src}=size=1280x720:rate=30:duration={video_s}" if src != "mandelbrot"
                        else f"mandelbrot=size=1280x720:rate=30,trim=duration={video_s}",
                        "-vf", f"hue=h={i * 17}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        str(d / f"v{i:02d}.mp4")], check=True)
    return d


def test_features_timing(tmp_path, fresh_root):
    if not gpu_available():
        pytest.skip("no GPU")
    from montage.candidates import build_media
    from montage.features.extract import extract
    album = make_album(tmp_path / "album")
    t0 = time.time()
    ms = build_media([album])
    t1 = time.time()
    f = extract(ms.assets, ms.candidates, ms.frames)
    t2 = time.time()
    print(f"ingest+candidates {t1 - t0:.1f}s, features {t2 - t1:.1f}s, {len(f.ids)} candidates")
    assert t2 - t1 < 90, f"features took {t2 - t1:.1f}s"
