"""Decode a song: 22,050 Hz mono for DSP (24 kHz for MERT loaded on demand)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from montage import cache, paths
from montage.config import get_config
from montage.types import UserError


@dataclass
class Audio:
    path: str
    song_hash: str
    y: np.ndarray          # mono float32 at sr
    sr: int

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr


def _decode_wav(path: str | Path, h: str) -> Path:
    """FFmpeg fallback for formats soundfile/audioread cannot open (m4a, aac, some mp3)."""
    out = paths.scratch_dir("audio") / f"{h}.wav"
    if not out.exists():
        r = subprocess.run([get_config().render.ffmpeg_bin, "-y", "-v", "error", "-i", str(path), "-vn",
                            "-ac", "1", "-ar", "44100", str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            out.unlink(missing_ok=True)
            raise UserError(f"Could not read the song file {Path(path).name}.")
    return out


def load_audio(path: str | Path, sr: int, h: str | None = None) -> np.ndarray:
    import librosa
    h = h or cache.file_hash(path)
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception:  # noqa: BLE001
        y, _ = librosa.load(str(_decode_wav(path, h)), sr=sr, mono=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def load_song(path: str | Path) -> Audio:
    cfg = get_config().music
    if not Path(path).exists():
        raise UserError(f"Song file not found: {Path(path).name}")
    h = cache.file_hash(path)
    y = load_audio(path, cfg.analysis_sr, h)
    if len(y) / cfg.analysis_sr < cfg.min_song_s:
        raise UserError(f"The song must be at least {cfg.min_song_s:.0f} seconds long.")
    return Audio(path=str(path), song_hash=h, y=y, sr=cfg.analysis_sr)
