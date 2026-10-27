"""Generate synthetic test media into tests/fixtures/ (gitignored). Needs FFmpeg for videos.

    python scripts/make_fixtures.py [--force]

Writes videos, photos, click tracks, a 3-section song, and truth.json (beats, downbeats,
section boundaries, scene cuts).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "fixtures"
SR = 44100
ENC = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def ffprobe_rotation(path: Path) -> int | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
                       capture_output=True, text=True)
    try:
        st = [s for s in json.loads(r.stdout)["streams"] if s["codec_type"] == "video"][0]
    except Exception:
        return None
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(sd["rotation"])
    if "rotate" in st.get("tags", {}):
        return -int(st["tags"]["rotate"])
    return None


# ---------------- videos ----------------

def make_videos(tmp: Path) -> dict:
    truth = {}
    ff("-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=12",
       "-f", "lavfi", "-i", "color=c=red:size=240x240:rate=30:duration=12",
       "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
       "-filter_complex", "[0][1]overlay=x='mod(t*600,1680)':y='420+250*sin(t*4)'[v]",
       "-map", "[v]", "-map", "2:a", *ENC, "-c:a", "aac", "-shortest", str(OUT / "vid_motion_landscape.mp4"))

    h, w = 1080, 1920
    yy, xx = np.mgrid[0:h, 0:w]
    grad = np.stack([xx / w * 200 + 30, yy / h * 180 + 40, np.full_like(xx, 120.0)], -1).astype(np.uint8)
    Image.fromarray(grad).save(tmp / "grad.png")
    ff("-loop", "1", "-framerate", "30", "-t", "8", "-i", str(tmp / "grad.png"),
       "-vf", "noise=alls=4:allf=t,format=yuv420p", *ENC, str(OUT / "vid_static_landscape.mp4"))

    ff("-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=6", *ENC, str(OUT / "vid_portrait.mp4"))

    # Landscape-encoded 1280x720 with an orientation marker (red box top-left), flagged rotate=90 (clockwise).
    ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=4",
       "-vf", "drawbox=x=0:y=0:w=320:h=180:color=red:t=fill", *ENC, str(tmp / "rot_src.mp4"))
    target = OUT / "vid_rotated.mov"
    target.unlink(missing_ok=True)
    try:  # FFmpeg >= 6.1: display matrix rotation is counter-clockwise, so -90 == legacy rotate=90
        ff("-display_rotation:v:0", "-90", "-i", str(tmp / "rot_src.mp4"), "-c", "copy", str(target))
    except subprocess.CalledProcessError:
        pass
    if ffprobe_rotation(target) is None:
        ff("-i", str(tmp / "rot_src.mp4"), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(target))
    truth["vid_rotated.mov"] = {"rotation": 90, "display_wh": [720, 1280], "probe_rotation": ffprobe_rotation(target)}
    # An upright reference (pre-rotated copy) for render orientation tests.
    ff("-i", str(tmp / "rot_src.mp4"), "-vf", "transpose=1", *ENC, str(OUT / "ref_rotated_upright.mp4"))

    ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=5",
       "-f", "lavfi", "-i", "smptehdbars=size=1280x720:rate=30:duration=5",
       "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v]", "-map", "[v]", *ENC, str(OUT / "vid_twoscenes.mp4"))
    truth["vid_twoscenes.mp4"] = {"scene_cuts": [5.0]}

    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2", *ENC,
       "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
       str(OUT / "vid_hdr.mp4"))
    truth["vid_hdr.mp4"] = {"is_hdr": True}
    return truth


# ---------------- photos ----------------

def _noise(img: Image.Image, seed: int, amp: float = 4.0) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    a += np.random.default_rng(seed).normal(0, amp, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _save(img: Image.Image, name: str, day: int) -> None:
    exif = Image.Exif()
    stamp = f"2026:07:{day:02d} 1{day % 10}:00:00"
    exif[0x0132] = stamp                      # DateTime (IFD0)
    exif.get_ifd(0x8769)[0x9003] = stamp      # DateTimeOriginal (Exif IFD)
    img.convert("RGB").save(OUT / name, quality=92, exif=exif.tobytes())


def make_photos() -> dict:
    truth = {}
    W, H = 1920, 1080
    # 1 sunset: gradient + sun
    yy = np.linspace(0, 1, H)[:, None, None]
    sky = (np.array([255, 140, 40]) * (1 - yy) + np.array([90, 30, 120]) * yy) * np.ones((1, W, 1))
    img = Image.fromarray(sky.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.ellipse([W // 2 - 150, H // 2 - 150, W // 2 + 150, H // 2 + 150], fill=(255, 230, 90))
    d.rectangle([0, int(H * 0.75), W, H], fill=(20, 20, 40))
    _save(_noise(img, 1), "photo_01_sunset.jpg", 1)
    # 2 checkerboard
    c = ((np.arange(H)[:, None] // 120 + np.arange(W)[None, :] // 120) % 2).astype(np.uint8)
    arr = np.where(c[..., None] == 1, np.array([30, 80, 220]), np.array([240, 240, 240])).astype(np.uint8)
    _save(_noise(Image.fromarray(arr), 2), "photo_02_checker.jpg", 2)
    # 3 circles, 4:3
    img = Image.new("RGB", (1600, 1200), (15, 15, 25))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y, r = rng.integers(0, 1600), rng.integers(0, 1200), rng.integers(30, 160)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(60, 255, 3)))
    _save(_noise(img, 3), "photo_03_circles_4x3.jpg", 3)
    # 4 stripes, portrait 3:4
    img = Image.new("RGB", (1200, 1600), (240, 250, 235))
    d = ImageDraw.Draw(img)
    for k in range(-1600, 1200, 90):
        d.line([(k, 0), (k + 1600, 1600)], fill=(30, 150, 60), width=35)
    _save(_noise(img, 4), "photo_04_stripes_portrait.jpg", 4)
    # 5 panorama 3:1 mountains
    PW, PH = 3000, 1000
    yy = np.linspace(0, 1, PH)[:, None, None]
    sky = (np.array([120, 180, 255]) * (1 - yy) + np.array([230, 240, 255]) * yy) * np.ones((1, PW, 1))
    img = Image.fromarray(sky.astype(np.uint8))
    d = ImageDraw.Draw(img)
    xs = np.linspace(0, PW, 25)
    ys = PH * (0.55 + 0.2 * np.sin(xs / 230.0) + 0.08 * np.cos(xs / 71.0))
    d.polygon([(0, PH)] + [(float(x), float(y)) for x, y in zip(xs, ys)] + [(PW, PH)], fill=(60, 70, 80))
    _save(_noise(img, 5), "photo_05_pano.jpg", 5)
    # 6 blurry: detailed pattern then heavy blur
    img = Image.new("RGB", (W, H), (200, 200, 190))
    d = ImageDraw.Draw(img)
    for i in range(0, W, 60):
        for j in range(0, H, 60):
            d.rectangle([i, j, i + 30, j + 30], fill=((i * 7) % 255, (j * 5) % 255, 90))
    img = img.filter(ImageFilter.GaussianBlur(25))
    _save(img, "photo_06_blurry.jpg", 6)
    # 7/8 near-duplicates: rings + text, then +2% brightness
    img = Image.new("RGB", (W, H), (10, 30, 40))
    d = ImageDraw.Draw(img)
    for r in range(600, 0, -50):
        col = (220, 60, 60) if (r // 50) % 2 else (40, 200, 190)
        d.ellipse([W // 2 - r, H // 2 - r, W // 2 + r, H // 2 + r], fill=col)
    d.text((80, 80), "HELLO MONTAGE", fill=(255, 255, 255))
    img = _noise(img, 7)
    _save(img, "photo_07_dupA.jpg", 7)
    brighter = np.clip(np.asarray(img).astype(np.float32) * 1.02, 0, 255).astype(np.uint8)
    _save(Image.fromarray(brighter), "photo_08_dupB.jpg", 8)
    truth["photos"] = {"blurry": "photo_06_blurry.jpg", "near_dups": ["photo_07_dupA.jpg", "photo_08_dupB.jpg"]}
    return truth


# ---------------- audio ----------------

def _burst(freq: float, dur: float, decay: float, amp: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)


def _add(y: np.ndarray, sig: np.ndarray, t0: float, gain: float = 1.0) -> None:
    i = int(round(t0 * SR))
    j = min(len(y), i + len(sig))
    if i < len(y):
        y[i:j] += gain * sig[: j - i]


def _write(name: str, y: np.ndarray) -> None:
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    wavfile.write(OUT / name, SR, (y * 32767).astype(np.int16))


def click_track(name: str, bpm: float, dur: float = 70.0, t0: float = 0.5, dynamic: bool = False) -> dict:
    period = 60.0 / bpm
    beats = np.arange(t0, dur - 0.2, period)
    y = np.zeros(int(dur * SR))
    t = np.arange(len(y)) / SR
    gain_t = np.ones_like(t)
    if dynamic:
        gain_t = np.where(t < 20, 0.15, np.where(t < 40, 0.4, 1.0))
    y += 0.03 * gain_t * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277.2 * t) + np.sin(2 * np.pi * 329.6 * t))
    hi, lo, kick = _burst(1000, 0.05, 80, 0.5), _burst(1500, 0.05, 80, 0.9), _burst(60, 0.15, 25, 0.9)
    for k, b in enumerate(beats):
        g = float(gain_t[min(len(t) - 1, int(b * SR))])
        if k % 4 == 0:
            _add(y, lo, b, g)
            _add(y, kick, b, g)
        else:
            _add(y, hi, b, g)
    if dynamic:  # denser hats in louder parts so onset density also steps up
        for b in beats:
            if b >= 20:
                _add(y, _burst(6000, 0.02, 200, 0.3), b + period / 2, float(gain_t[int(b * SR)]))
    _write(name, y)
    return {"bpm": bpm, "beats": [round(float(b), 4) for b in beats], "downbeats": [round(float(b), 4) for b in beats[::4]],
            "loudness_steps": [20.0, 40.0] if dynamic else []}


def song_sections(name: str = "song_sections.wav") -> dict:
    dur, bpm = 90.0, 120.0
    period = 60.0 / bpm
    beats = np.arange(0.0, dur - 0.2, period)
    y = np.zeros(int(dur * SR))
    t = np.arange(len(y)) / SR
    rng = np.random.default_rng(0)
    a = t < 30
    b = (t >= 30) & (t < 60)
    c = t >= 60
    # section A: quiet C-major pad, soft kick on downbeats
    y += a * 0.06 * sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    # section B: loud F-major saw-ish pad + bass, full drums
    y += b * 0.12 * sum(np.sign(np.sin(2 * np.pi * f * t)) * 0.5 for f in (349.2, 440.0, 523.3))
    y += b * 0.25 * np.sin(2 * np.pi * 87.3 * t)
    # section C: quiet A-minor triangle-ish pad
    y += c * 0.07 * sum((2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f * t)) for f in (220.0, 261.6, 329.6))
    kick = _burst(55, 0.2, 20, 1.0)
    snare = rng.normal(0, 1, int(0.12 * SR)) * np.exp(-np.arange(int(0.12 * SR)) / SR * 30) * 0.5
    hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 150) * 0.25
    click = _burst(2000, 0.03, 120, 0.2)
    for k, bt in enumerate(beats):
        if bt < 30:
            if k % 4 == 0:
                _add(y, kick, bt, 0.35)
        elif bt < 60:
            _add(y, kick, bt, 1.0)
            if k % 2 == 1:
                _add(y, snare, bt, 1.0)
            _add(y, hat, bt)
            _add(y, hat, bt + period / 2)
        else:
            if k % 2 == 0:
                _add(y, click, bt, 1.0)
    _write(name, y)
    return {"bpm": bpm, "beats": [round(float(x), 4) for x in beats], "downbeats": [round(float(x), 4) for x in beats[::4]],
            "boundaries": [30.0, 60.0]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="skip FFmpeg-made files")
    args = ap.parse_args(argv)
    if (OUT / "truth.json").exists() and not args.force:
        print(f"fixtures exist in {OUT} (use --force to regenerate)")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_tmp"
    tmp.mkdir(exist_ok=True)
    truth: dict = {}
    truth.update(make_photos())
    for name, bpm in (("click_090.wav", 90), ("click_120.wav", 120), ("click_150.wav", 150)):
        truth[name] = click_track(name, bpm)
    truth["click_120_dynamic.wav"] = click_track("click_120_dynamic.wav", 120, dynamic=True)
    truth["song_sections.wav"] = song_sections()
    if not args.no_video:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found; run with --no-video or install FFmpeg", file=sys.stderr)
            return 1
        truth.update(make_videos(tmp))
    shutil.rmtree(tmp, ignore_errors=True)
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
