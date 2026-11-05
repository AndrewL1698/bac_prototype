"""Render an EDL: per-shot encodes in parallel -> concat (stream copy) -> song excerpt -> mux -> verify."""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from montage import paths
from montage.config import get_config
from montage.log import get_logger, stage
from montage.render import shot_cache
from montage.render.shots import render_shot
from montage.types import EDL, Asset

log = get_logger("render")


def _ff(*args: str) -> None:
    r = subprocess.run([get_config().render.ffmpeg_bin, "-y", "-v", "error", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")


def probe_counts(path: Path) -> tuple[int, float]:
    """(video packet count, audio duration seconds)."""
    c = get_config().render
    r = subprocess.run([c.ffprobe_bin, "-v", "error", "-count_packets", "-show_entries",
                        "stream=codec_type,nb_read_packets,duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    streams = json.loads(r.stdout or "{}").get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return int(v.get("nb_read_packets", 0)), float(a.get("duration", 0) or 0)


def _render_shots(edl: EDL, assets: dict[str, Asset], faces: dict[str, tuple[float, float]],
                  W: int, H: int, crf: int, work: Path) -> list[Path]:
    """Encoded shot files in EDL order. With the shot cache on, unchanged shots are reused and a shot
    that appears twice in one EDL is encoded once."""
    cfg = get_config().render
    face = lambda s: faces.get(s.candidate_id, (-1.0, -1.0))  # noqa: E731
    if not cfg.shot_cache.enabled:
        outs = [work / f"shot_{s.slot_idx:03d}.mp4" for s in edl.shots]
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            list(ex.map(lambda so: render_shot(so[0], assets[so[0].asset_id], so[1], W, H, crf, face(so[0])),
                        zip(edl.shots, outs)))
        return outs
    keys = [shot_cache.shot_key(s, assets[s.asset_id], W, H, crf, face(s)) for s in edl.shots]
    unique = {k: s for k, s in zip(keys, edl.shots)}
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        done = dict(zip(unique, ex.map(
            lambda ks: shot_cache.cached_shot(ks[0], ks[1], assets[ks[1].asset_id], W, H, crf, face(ks[1])),
            unique.items())))
    reused = sum(hit for _, hit in done.values())
    log.info("shot cache: reused %d of %d shots, encoded %d", reused, len(unique), len(unique) - reused)
    return [done[k][0] for k in keys]


def _song_audio(edl: EDL, song_path: str, L: float, work: Path) -> Path:
    """The song excerpt as AAC (with the fade-out). Cached next to the shots when the shot cache is on:
    the AAC encode is the slowest step of assembly, and most recuts keep the same excerpt."""
    cfg = get_config().render
    fade = edl.excerpt.fade_out

    def encode(out: Path) -> None:
        _ff("-ss", f"{edl.excerpt.song_start:.4f}", "-i", song_path, "-t", f"{L:.4f}", "-vn",
            "-af", f"afade=t=out:st={max(0.0, L - fade):.4f}:d={fade:.4f},apad,atrim=0:{L:.4f}",
            "-ar", "48000", "-c:a", "aac", "-b:a", cfg.audio_bitrate, str(out))

    if not cfg.shot_cache.enabled:
        encode(work / "audio.m4a")
        return work / "audio.m4a"
    p, _ = shot_cache.cached_file(
        shot_cache.audio_key(shot_cache.song_file_hash(song_path), edl.excerpt.song_start, L, fade), ".m4a", encode)
    return p


def render_edl(edl: EDL, assets: dict[str, Asset], song_path: str, preview: bool = True,
               faces: dict[str, tuple[float, float]] | None = None, out_path: Path | None = None) -> Path:
    cfg = get_config().render
    enc = cfg.preview if preview else cfg.export
    W, H, crf = enc.width, enc.height, enc.crf
    faces = faces or {}
    work = paths.scratch_dir("render") / f"{edl.run_id}_{'p' if preview else 'x'}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    total = edl.shots[-1].frame_end
    L = total / edl.fps
    try:
        with stage("render_shots"):
            outs = _render_shots(edl, assets, faces, W, H, crf, work)
        with stage("render_assemble"):
            (work / "concat.txt").write_text("".join(f"file '{o.as_posix()}'\n" for o in outs))
            _ff("-f", "concat", "-safe", "0", "-i", str(work / "concat.txt"), "-c", "copy", str(work / "video.mp4"))
            audio = _song_audio(edl, song_path, L, work)
            _ff("-i", str(work / "video.mp4"), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                "-c", "copy", "-movflags", "+faststart", str(work / "out.mp4"))
            n_video, a_dur = probe_counts(work / "out.mp4")
            if n_video != total:
                raise RuntimeError(f"render produced {n_video} video frames, expected {total}")
            if abs(a_dur - L) > 1.0 / edl.fps + 1e-3:
                raise RuntimeError(f"audio is {a_dur:.3f}s, expected {L:.3f}s")
        dest = out_path or paths.run_dir(edl.run_id) / ("preview.mp4" if preview else "video.mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)
        paths.atomic_write(dest, lambda tmp: shutil.copyfile(work / "out.mp4", tmp))
        if cfg.shot_cache.enabled:
            shot_cache.prune(keep={*outs, audio})
        return dest
    finally:
        shutil.rmtree(work, ignore_errors=True)
