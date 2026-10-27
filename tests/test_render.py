"""Phase 6 acceptance: framing, shot rendering, assembly (media tests run on Colab)."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from montage.render.framing import decide_framing
from montage.types import EDL, Asset, EditPlan, Excerpt, KenBurns, Shot, ShotScores

SC = ShotScores(importance=0, mood=0, director=0, unary_total=0, pairwise_in=0)


def _asset(w, h, kind="photo", **kw):
    return Asset(id="a000", kind=kind, src_path="x", hash="h", width=w, height=h, filename="x", **kw)


def test_framing_table():
    assert decide_framing(_asset(1920, 1080)) == "native"
    assert decide_framing(_asset(1600, 1200)) == "crop"        # 4:3 loses 25% <= max_crop_loss 0.25
    assert decide_framing(_asset(1200, 1600)) == "fill_blur"   # 3:4
    assert decide_framing(_asset(1500, 1200)) == "fill_blur"   # 5:4 loses 30% > 0.25
    assert decide_framing(_asset(1000, 1000)) == "fill_blur"   # 1:1
    assert decide_framing(_asset(3000, 1000)) == "pan"         # 3:1 photo
    assert decide_framing(_asset(2200, 1000, "video")) == "crop"  # 2.2:1 loses 19% <= 0.25
    assert decide_framing(_asset(3000, 1000, "video")) == "fill_blur"
    assert decide_framing(_asset(1700, 1080)) == "crop"        # 1.57:1 loses 11%


def test_kenburns_smooth():
    cv2 = pytest.importorskip("cv2")
    from montage.render import kenburns
    rng = np.random.default_rng(0)
    tex = (rng.uniform(0, 255, (900, 1600, 3))).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 2)
    p = Path(__file__).parent / "_kb_tex.png"
    Image.fromarray(tex).save(p)
    try:
        kb = KenBurns(zoom_from=1.0, zoom_to=1.12, pan_x=0.02, pan_y=0.0)
        frs = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in kenburns.frames(str(p), "native", kb, 45, 640, 360)]
    finally:
        p.unlink(missing_ok=True)
    orb = cv2.ORB_create(1500)
    k0, d0 = orb.detectAndCompute(frs[0], None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    scales = []
    for f in frs[1:]:
        k1, d1 = orb.detectAndCompute(f, None)
        m = bf.match(d0, d1)
        a = np.float32([k0[x.queryIdx].pt for x in m])
        b = np.float32([k1[x.trainIdx].pt for x in m])
        M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC)
        scales.append(float(np.hypot(M[0, 0], M[1, 0])))
    s = np.array(scales)
    assert s[-1] > 1.08
    assert np.all(np.diff(s) > -0.004), np.round(s, 4)  # monotonic zoom, no reversals


# ---------------- media ----------------

def _probe_frames(path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int(r.stdout.strip())


def _frame_at(path, idx, W=1280, H=720) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-vframes", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


@pytest.fixture(scope="module")
def assets(fixtures):
    from montage.ingest.assets import ingest
    return {a.filename: a for a in ingest([fixtures])}


def _shot(j, f0, f1, a: Asset, framing, src_in=None, kb=None, fps=30):
    return Shot(slot_idx=j, start=f0 / fps, end=f1 / fps, frame_start=f0, frame_end=f1, candidate_id=a.id,
                asset_id=a.id, kind=a.kind, src_in=src_in, src_out=None if src_in is None else src_in + (f1 - f0) / fps,
                framing=framing, kenburns=kb, scores=SC, reason="test")


def _edl(shots, song_start=0.5, run_id="test_render") -> EDL:
    plan = EditPlan(vibe="", vibe_queries=[], avoid_queries=[], pacing=0, energy_bias=0, chronological=False,
                    target_length_s=0, excerpt_hint="auto", excerpt_start_s=0, opening_query="", closing_query="",
                    segments=[], must_include=[], exclude=[], rationale="")
    L = shots[-1].frame_end / 30
    return EDL(run_id=run_id, created=datetime.now(), system="full", seed=0, fps=30, width=1280, height=720,
               song_hash="h", excerpt=Excerpt(song_start=song_start, song_end=song_start + L, fade_out=1.5),
               plan=plan, shots=shots, total_cost=0)


@pytest.mark.media
def test_shot_frame_count(assets, tmp_path):
    from montage.render.shots import render_shot
    kb = KenBurns(zoom_from=1.0, zoom_to=1.08, pan_x=0.01, pan_y=0.0)
    cases = [(assets["vid_motion_landscape.mp4"], "native", 1.0, None, 37),
             (assets["vid_portrait.mp4"], "fill_blur", 0.5, None, 23),
             (assets["vid_motion_landscape.mp4"], "native", 11.5, None, 40),   # runs past the source end: tpad
             (assets["photo_01_sunset.jpg"], "native", None, kb, 31),
             (assets["photo_03_circles_4x3.jpg"], "crop", None, kb, 29),
             (assets["photo_04_stripes_portrait.jpg"], "fill_blur", None, kb, 17),
             (assets["photo_05_pano.jpg"], "pan", None, KenBurns(zoom_from=1, zoom_to=1, pan_x=1, pan_y=0), 45)]
    for j, (a, fr, src_in, k, n) in enumerate(cases):
        out = render_shot(_shot(j, 0, n, a, fr, src_in, k), a, tmp_path / f"s{j}.mp4", 1280, 720, 23)
        assert _probe_frames(out) == n, (a.filename, fr)


@pytest.mark.media
def test_fill_blur_no_black(assets, tmp_path):
    from montage.render.shots import render_shot
    a = assets["vid_portrait.mp4"]
    out = render_shot(_shot(0, 0, 10, a, "fill_blur", 1.0), a, tmp_path / "p.mp4", 1280, 720, 23)
    fr = _frame_at(out, 5).astype(float).mean(-1)
    assert fr[:, :128].mean() > 10 and fr[:, -128:].mean() > 10


@pytest.mark.media
def test_rotated_upright(assets, fixtures, tmp_path):
    from montage.ingest.assets import ingest
    from montage.render.shots import render_shot
    a = assets["vid_rotated.mov"]
    ref = next(x for x in ingest([fixtures]) if x.filename == "ref_rotated_upright.mp4")
    out1 = render_shot(_shot(0, 0, 10, a, "fill_blur", 1.0), a, tmp_path / "r1.mp4", 1280, 720, 18)
    out2 = render_shot(_shot(0, 0, 10, ref, "fill_blur", 1.0), ref, tmp_path / "r2.mp4", 1280, 720, 18)
    f1, f2 = _frame_at(out1, 3).astype(float), _frame_at(out2, 3).astype(float)
    assert np.abs(f1 - f2).mean() < 12
    # orientation marker: red box is top-left in the upright reference -> top-right after a clockwise turn
    fg = f2[:, 438:842]  # 405 px wide portrait foreground centered in 1280
    assert fg[:60, -60:, 0].mean() > fg[:60, -60:, 2].mean() + 40


@pytest.mark.media
def test_total_frames_and_cuts(assets, fixtures, tmp_path):
    from scenedetect import ContentDetector, SceneManager, open_video
    from montage.render.assemble import probe_counts, render_edl
    colors = [(230, 30, 30), (30, 200, 40), (40, 60, 230), (240, 240, 40), (200, 40, 220), (20, 220, 220)]
    shots, f0 = [], 0
    lens = [41, 29, 50, 37, 61, 44]
    amap = {}
    for j, (c, n) in enumerate(zip(colors, lens)):
        p = tmp_path / f"solid{j}.png"
        Image.new("RGB", (1920, 1080), c).save(p)
        a = Asset(id=f"s{j}", kind="photo", src_path=str(p), hash=f"s{j}", width=1920, height=1080, filename=p.name)
        amap[a.id] = a
        shots.append(_shot(j, f0, f0 + n, a, "native", None, KenBurns(zoom_from=1, zoom_to=1.05, pan_x=0, pan_y=0)))
        f0 += n
    edl = _edl(shots)
    out = render_edl(edl, amap, str(fixtures / "click_120.wav"), preview=True, out_path=tmp_path / "out.mp4")
    n_video, a_dur = probe_counts(out)
    assert n_video == f0
    assert abs(a_dur - f0 / 30) <= 1 / 30 + 1e-3
    video = open_video(str(out))
    sm = SceneManager()
    sm.add_detector(ContentDetector(min_scene_len=5))
    sm.detect_scenes(video)
    detected = [s[0].get_frames() for s in sm.get_scene_list()][1:]
    expected = [s.frame_start for s in shots[1:]]
    assert len(detected) == len(expected)
    assert all(abs(d - e) <= 1 for d, e in zip(detected, expected)), (detected, expected)


def _forty(assets):
    names = ["vid_motion_landscape.mp4", "photo_01_sunset.jpg", "vid_portrait.mp4", "photo_03_circles_4x3.jpg",
             "vid_static_landscape.mp4", "photo_04_stripes_portrait.jpg", "vid_twoscenes.mp4", "photo_05_pano.jpg"]
    shots, f0 = [], 0
    rng = np.random.default_rng(0)
    for j in range(40):
        a = assets[names[j % len(names)]]
        n = int(rng.choice([15, 30, 45, 60]))
        fr = decide_framing(a)
        kb = KenBurns(zoom_from=1, zoom_to=1 if fr == "pan" else 1.08, pan_x=1 if fr == "pan" else 0.01, pan_y=0) \
            if a.kind == "photo" else None
        src = None if a.kind == "photo" else float(rng.uniform(0.3, max(0.4, (a.duration or 4) - n / 30 - 0.3)))
        shots.append(_shot(j, f0, f0 + n, a, fr, src, kb))
        f0 += n
    return shots, {a.id: a for a in assets.values()}


@pytest.mark.media
def test_render_timing(assets, fixtures, tmp_path):
    from montage.render.assemble import render_edl
    shots, amap = _forty(assets)
    edl = _edl(shots, run_id="timing")
    t0 = time.time()
    render_edl(edl, amap, str(fixtures / "click_120.wav"), preview=True, out_path=tmp_path / "p.mp4")
    t1 = time.time()
    render_edl(edl, amap, str(fixtures / "click_120.wav"), preview=False, out_path=tmp_path / "x.mp4")
    t2 = time.time()
    print(json.dumps({"preview_s": round(t1 - t0, 1), "export_s": round(t2 - t1, 1), "frames": shots[-1].frame_end}))
    assert t1 - t0 < 25 and t2 - t1 < 60
