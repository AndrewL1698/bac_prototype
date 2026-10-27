"""Full-song analysis (cached by song hash) and the per-plan excerpt MusicProfile."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from montage import cache
from montage.config import get_config
from montage.log import get_logger, stage
from montage.music import beats as beats_mod
from montage.music import emotion
from montage.music.energy import energy_curves, gauss_smooth, grid_for, zscore
from montage.music.structure import analyze_structure
from montage.types import Excerpt, MusicProfile, Section, UserError

log = get_logger("music")
ANALYSIS_VERSION = 1
ARRAY_KEYS = ("grid", "intensity", "arousal", "valence", "novelty", "rms_db", "onset_density", "centroid")


@dataclass
class FullAnalysis:
    song_hash: str
    duration: float
    tempo: float
    beat_mode: str
    beats: list[float]
    downbeats: list[float]
    boundaries: list[float]              # song time, snapped to downbeats
    emotion_source: str
    beat_source: str
    version: int = ANALYSIS_VERSION
    arrays: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def a(self, key: str) -> np.ndarray:
        return self.arrays[key]

    @property
    def bar(self) -> float:
        d = np.diff(self.downbeats)
        return float(np.median(d)) if len(d) else 4 * 60.0 / max(self.tempo, 1.0)


def _paths(h: str) -> tuple[Path, Path]:
    return cache.cache_path("music", h, ".json"), cache.cache_path("music", f"{h}_grid", ".npz")


def load_cached(h: str) -> FullAnalysis | None:
    jp, npz = _paths(h)
    meta = cache.read_json(jp)
    arrs = cache.load_npz(npz) if meta else None
    if not meta or arrs is None or meta.get("version") != ANALYSIS_VERSION:
        return None
    return FullAnalysis(**{k: v for k, v in meta.items() if k != "arrays"}, arrays=arrs)


def save_cached(fa: FullAnalysis) -> None:
    jp, npz = _paths(fa.song_hash)
    cache.save_npz(npz, **fa.arrays)
    meta = asdict(fa)
    meta.pop("arrays")
    cache.write_json(jp, meta)


def compute_intensity(arousal: np.ndarray, curves: dict[str, np.ndarray]) -> np.ndarray:
    c = get_config().music.intensity
    raw = c.w_arousal * zscore(arousal) + c.w_rms * zscore(curves["rms_db"]) + c.w_onset * zscore(curves["onset_density"])
    return zscore(gauss_smooth(raw, c.smooth_s))


def analyze_song(path: str | Path) -> FullAnalysis:
    """Beats, curves, sections, emotion and intensity for the whole song. Cached by content hash."""
    cfg = get_config().music
    h = cache.file_hash(path)
    want_emotion = "mert" if emotion.mert_available() else "heuristic"
    fa = load_cached(h)
    if fa is not None and fa.emotion_source == want_emotion:
        if fa.duration < cfg.min_song_s:
            raise UserError(f"The song must be at least {cfg.min_song_s:.0f} seconds long.")
        return fa
    from montage.music.load import load_audio, load_song
    with stage("music_load"):
        audio = load_song(path)
    y, sr = audio.y, audio.sr
    grid = grid_for(audio.duration)
    with stage("music_beats"):
        b = beats_mod.analyze_beats(y, sr)
    with stage("music_energy"):
        curves = energy_curves(y, sr, grid)
    with stage("music_structure"):
        bounds, nov = analyze_structure(y, sr, b["beats"], b["downbeats"], grid, audio.duration)
    with stage("music_emotion"):
        y24 = load_audio(path, cfg.mert_sr, h) if want_emotion == "mert" else None
        arousal, valence, src = emotion.predict_emotion(y24, grid, curves)
    intensity = compute_intensity(arousal, curves)
    fa = FullAnalysis(
        song_hash=h, duration=round(audio.duration, 4), tempo=b["tempo"], beat_mode=b["beat_mode"],
        beats=b["beats"], downbeats=b["downbeats"], boundaries=bounds, emotion_source=src, beat_source=b["beat_source"],
        arrays={"grid": grid.astype(np.float32), "intensity": intensity,
                "arousal": gauss_smooth(arousal, 1.0), "valence": gauss_smooth(valence, 1.0),
                "novelty": nov, "rms_db": curves["rms_db"], "onset_density": curves["onset_density"],
                "centroid": curves["centroid"]},
    )
    save_cached(fa)
    return fa


# ---------------- excerpt profile ----------------

def label_sections(fa: FullAnalysis, s0: float, s1: float) -> list[Section]:
    """Sections of [s0, s1] (song time) on the excerpt timeline, labeled per spec 02 section 7."""
    thr = get_config().music.sections.slope_thr
    grid, inten = fa.a("grid"), fa.a("intensity")
    med = float(np.median(inten))
    edges = [s0] + [b for b in fa.boundaries if s0 + 1.0 < b < s1 - 1.0] + [s1]
    in_ex = (grid >= s0) & (grid < s1)
    peak_t = float(grid[in_ex][np.argmax(inten[in_ex])]) if in_ex.any() else s0
    out: list[Section] = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        m = (grid >= a) & (grid < b)
        vals, ts = inten[m], grid[m]
        mean = float(vals.mean()) if len(vals) else 0.0
        slope = float(np.polyfit(ts, vals, 1)[0]) if len(vals) >= 3 else 0.0
        if a <= peak_t < b:
            label = "peak"
        elif slope > thr:
            label = "building"
        elif slope < -thr:
            label = "falling"
        else:
            label = "high" if mean >= med else "low"
        out.append(Section(idx=i, start=round(a - s0, 4), end=round(b - s0, 4), label=label,
                           mean_intensity=round(mean, 4)))
    return out


def build_profile(fa: FullAnalysis, ex: Excerpt) -> MusicProfile:
    s0, s1 = ex.song_start, ex.song_end
    L = s1 - s0
    tgrid = np.round(np.arange(0.0, L, 0.5), 3)
    src = fa.a("grid")

    def resample(key: str) -> list[float]:
        return [round(float(v), 4) for v in np.interp(tgrid + s0, src, fa.a(key))]

    shift = lambda ts: [round(t - s0, 4) for t in ts if s0 - 1e-3 <= t < s1 - 1e-3]  # noqa: E731
    beats = shift(fa.beats)
    if not beats or beats[0] > 1e-3:
        beats = [0.0] + beats
    beats[0] = 0.0
    inten = resample("intensity")
    return MusicProfile(
        song_hash=fa.song_hash, song_duration=fa.duration, tempo=fa.tempo, beat_mode=fa.beat_mode, excerpt=ex,
        beats=beats, downbeats=shift(fa.downbeats), grid_times=[float(t) for t in tgrid], intensity=inten,
        arousal=resample("arousal"), valence=resample("valence"), novelty=resample("novelty"),
        sections=label_sections(fa, s0, s1), peak_time=float(tgrid[int(np.argmax(inten))]),
        emotion_source=fa.emotion_source, beat_source=fa.beat_source,
    )


def mmss(t: float) -> str:
    m, s = divmod(int(round(max(0.0, t))), 60)
    return f"{m}:{s:02d}"


def summarize(p: MusicProfile) -> str:
    L = p.excerpt.song_end - p.excerpt.song_start
    lines = [f"Song: {p.tempo:.0f} BPM, length {mmss(p.song_duration)}; excerpt {mmss(L)} "
             f"(starts at song {mmss(p.excerpt.song_start)}). Peak at {mmss(p.peak_time)}."]
    for s in p.sections:
        lines.append(f"S{s.idx} {s.start:.1f}-{s.end:.1f}s {s.label}")
    return "\n".join(lines)
