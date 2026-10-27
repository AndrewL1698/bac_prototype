"""Config: configs/default.yaml -> nested pydantic model. The single source of numbers.

Overrides, applied in order:
1. `MONTAGE_CONFIG=<extra.yaml>` merged on top of the defaults.
2. Dotted overrides, e.g. `--set planner.beam_width=64` (CLI) or `override({"planner.w_mood": 0})`.

Sub-models allow extra keys, so a new YAML key works before it is declared here
(declare it when code starts depending on it).
"""

from __future__ import annotations

import copy
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class _B(BaseModel):
    model_config = ConfigDict(extra="allow")


class IngestCfg(_B):
    min_assets: int
    max_assets: int
    max_total_video_s: float
    max_per_video_s: float
    analysis_fps: float
    analysis_short_side: int
    motion_pair_short_side: int
    motion_pair_dt_s: float
    workers: int


class CandidatesCfg(_B):
    window_s: float
    hop_s: float
    min_window_s: float
    edge_trim_s: float
    min_scene_s: float
    scene_detector: Literal["adaptive", "content"]
    thumb_width: int


class FeaturesCfg(_B):
    version: str
    clip_model: str
    clip_pretrained: str
    clip_batch: int
    aesthetic_weights: str
    yunet_model: str
    face_score: float
    tags_file: str
    tags_top_k: int
    tag_min_cos: float


class MoodPrompts(_B):
    arousal_high: list[str]
    arousal_low: list[str]
    valence_pos: list[str]
    valence_neg: list[str]


class MoodCfg(_B):
    templates: list[str]
    prompts: MoodPrompts
    arousal_weights: dict[str, float]
    valence_weights: dict[str, float]
    photo_motion_z: float


class DedupCfg(_B):
    cos_threshold: float


class HeuristicWeights(_B):
    aes: float
    sharp: float
    exp: float
    bright: float
    face_count: float
    face_area: float
    color: float
    motion: float
    shake: float
    junk: float


class ImportanceCfg(_B):
    heuristic: HeuristicWeights
    junk_tags: list[str]
    reject_sharpness_pct: float
    abs_min_sharpness: float
    reject_exposure_clip: float
    blend_alpha: float
    ranker_weights: str


class StructureCfg(_B):
    kernel_beats: int
    min_section_beats: int
    peak_k: float


class EmotionCfg(_B):
    use_mert: bool
    model: str
    window_s: float
    align: str
    batch: int
    head_weights: str


class IntensityCfg(_B):
    w_arousal: float
    w_rms: float
    w_onset: float
    smooth_s: float


class SectionsCfg(_B):
    slope_thr: float


class ExcerptWeights(_B):
    mean: float
    peak: float
    start: float
    end: float
    silence: float


class ExcerptCfg(_B):
    default_length_s: float
    min_length_s: float
    max_length_s: float
    fade_out_s: float
    silence_db: float
    peak_target_frac: float
    peak_width: float
    weights: ExcerptWeights


class MusicCfg(_B):
    min_song_s: float
    max_bpm: float
    min_bpm: float
    analysis_sr: int
    mert_sr: int
    structure: StructureCfg
    emotion: EmotionCfg
    intensity: IntensityCfg
    sections: SectionsCfg
    excerpt: ExcerptCfg


class DirectorWeights(_B):
    vibe: float
    seg: float
    avoid: float
    avoid_margin: float
    open: float
    close: float


class ContactSheetCfg(_B):
    cell_px: int
    max_cells: int


class DirectorCfg(_B):
    use_llm: bool
    model: str
    max_tokens: int
    timeout_s: float
    max_queries: int
    max_query_words: int
    send_thumbnails: bool
    editor_note: bool
    weights: DirectorWeights
    mood_targets: dict[str, float]
    energy_bias_scale: float
    contact_sheet: ContactSheetCfg


class InOutCfg(_B):
    w_aes: float
    w_motion: float
    w_rel: float
    drift_penalty: float
    step_s: float


class PlannerCfg(_B):
    min_shot_s: float
    max_shot_s: float
    max_fill_ratio: float
    boundary_novelty_scale: float
    w_imp: float
    w_mood: float
    w_val: float
    w_dir: float
    photo_min_s: float
    w_photo_short: float
    w_photo_hi: float
    w_must: float
    variety: float
    w_rep: float
    rep_tau: float
    w_src: float
    w_jump: float
    w_contrast: float
    w_chrono: float
    chrono_tol_s: float
    w_reuse: float
    max_per_cluster: int
    max_windows_per_video: int
    min_photo_frac: float
    max_photo_frac: float
    w_ratio: float
    min_pool: int
    expand_top: int
    beam_width: int
    naive_cut_s: float
    inout: InOutCfg
    use_forced_cuts: bool


class EncodeCfg(_B):
    width: int
    height: int
    crf: int


class RenderCfg(_B):
    fps: int
    preset: str
    preview: EncodeCfg
    export: EncodeCfg
    audio_bitrate: str
    max_crop_loss: float
    pano_aspect: float
    blur_sigma: float
    blur_darken: float
    kb_base: float
    kb_gain: float
    kb_max_pan: float
    fill_blur_oversize: float
    workers: int
    ffmpeg_bin: str
    ffprobe_bin: str


class AppCfg(_B):
    port: int
    share: bool
    default_seed: int


class EvalCfg(_B):
    beat_tol_s: float
    pace_window_s: float
    seeds: int
    systems: list[str]
    default_brief: str


class Config(_B):
    seed: int
    ingest: IngestCfg
    candidates: CandidatesCfg
    features: FeaturesCfg
    mood: MoodCfg
    dedup: DedupCfg
    importance: ImportanceCfg
    music: MusicCfg
    director: DirectorCfg
    planner: PlannerCfg
    render: RenderCfg
    app: AppCfg
    eval: EvalCfg


# ---------- loading ----------

def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_set(item: str) -> tuple[list[str], Any]:
    """'planner.beam_width=64' -> (['planner', 'beam_width'], 64). Values parse as YAML."""
    if "=" not in item:
        raise ValueError(f"override must look like a.b=value, got {item!r}")
    key, raw = item.split("=", 1)
    keys = [k for k in key.strip().split(".") if k]
    if not keys:
        raise ValueError(f"empty key in override {item!r}")
    return keys, yaml.safe_load(raw)


def _apply_dotted(data: dict, overrides: dict[str, Any] | list[str] | None) -> dict:
    if not overrides:
        return data
    items = (
        [parse_set(s) for s in overrides]
        if isinstance(overrides, list)
        else [([k for k in key.split(".") if k], v) for key, v in overrides.items()]
    )
    data = copy.deepcopy(data)
    for keys, value in items:
        node = data
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                raise KeyError(f"unknown config section {'.'.join(keys)}")
            node = node[k]
        node[keys[-1]] = value
    return data


def load_raw(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    extra = os.environ.get("MONTAGE_CONFIG")
    if extra:
        with open(extra, encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})
    return data


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | list[str] | None = None) -> Config:
    return Config.model_validate(_apply_dotted(load_raw(path), overrides))


_lock = threading.Lock()
_cached: Config | None = None
_cli_overrides: list[str] = []


def get_config() -> Config:
    global _cached
    with _lock:
        if _cached is None:
            _cached = load_config(overrides=_cli_overrides or None)
        return _cached


def set_overrides(items: list[str] | None) -> Config:
    """Apply CLI `--set a.b=v` overrides process-wide."""
    global _cached, _cli_overrides
    with _lock:
        _cli_overrides = list(items or [])
        _cached = None
    return get_config()


def reset() -> None:
    global _cached
    with _lock:
        _cached = None


@contextmanager
def override(values: dict[str, Any]) -> Iterator[Config]:
    """Temporarily replace the global config with dotted overrides (eval ablations, tests).
    Process-global: do not use concurrently from threads with different values."""
    global _cached
    base = get_config()
    patched = Config.model_validate(_apply_dotted(base.model_dump(), values))
    with _lock:
        prev, _cached = _cached, patched
    try:
        yield patched
    finally:
        with _lock:
            _cached = prev


def repo_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root (e.g. features.tags_file)."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
