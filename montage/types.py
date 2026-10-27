"""Data contracts. Defined exactly as in docs/spec/schemas.md; change both together."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class UserError(Exception):
    """Friendly, user-facing error. The app shows str(e)."""


# ---------- Ingest ----------
class Asset(BaseModel):
    id: str                       # "a000", "a001", ... stable order by (taken_at, filename)
    kind: Literal["photo", "video"]
    src_path: str                 # original upload path
    hash: str                     # sha1 of file bytes (first 16 hex chars)
    width: int                    # display width AFTER rotation is applied
    height: int
    rotation: int = 0             # degrees from metadata (0/90/180/270)
    duration: float | None = None # video only
    fps: float | None = None
    codec: str | None = None
    is_hdr: bool = False
    taken_at: datetime | None = None
    filename: str


class Candidate(BaseModel):
    id: str                       # "a003" for photos, "a007_w04" for video windows
    asset_id: str
    kind: Literal["photo", "video"]
    src_start: float | None = None   # video window start in source seconds
    src_end: float | None = None
    src_max_end: float | None = None # latest usable source time from src_start (scene end or video end minus edge trim);
                                     # lets a 3 s window fill a longer slot by extending forward
    scene_id: str | None = None      # "a007_s1"
    frame_times: list[float] = []    # sampled analysis frame times (video), [] for photo
    thumb_path: str                  # small JPEG for UI/contact sheet


# ---------- Features ----------
class FeatureTable(BaseModel):
    """Row-aligned with `candidate_ids`. Arrays stored in an .npz next to a JSON header."""
    candidate_ids: list[str]
    # stored in npz, exposed as numpy arrays after load (not validated by pydantic):
    # emb (N,768) float32 L2-normalized, aesthetic, sharpness, brightness, contrast,
    # exposure_clip, colorfulness, motion, shake, face_count, face_area,
    # mood_arousal_raw, mood_valence_raw (cached raw; z-scored within the batch at load),
    # dup_cluster (int, computed per batch, not cached)
    tags: dict[str, list[str]]       # candidate_id -> top-k tags
    feature_version: str             # bump when extraction code changes (invalidates cache)


class ImportanceScores(BaseModel):
    candidate_ids: list[str]
    heuristic: list[float]           # raw heuristic
    learned: list[float] | None      # None if ranker unavailable
    final: list[float]               # in [0,1], rank-normalized within modality, blended
    source: Literal["heuristic", "blend"]


# ---------- Music ----------
class Section(BaseModel):
    idx: int
    start: float                     # excerpt timeline
    end: float
    label: Literal["low", "building", "high", "peak", "falling"]
    mean_intensity: float


class Excerpt(BaseModel):
    song_start: float
    song_end: float
    fade_out: float                  # seconds of audio fade at the end


class MusicProfile(BaseModel):
    song_hash: str
    song_duration: float
    tempo: float
    beat_mode: Literal["normal", "half", "double"]
    excerpt: Excerpt
    beats: list[float]               # excerpt timeline, after half/double adjustment
    downbeats: list[float]
    grid_times: list[float]          # 2 Hz grid on excerpt timeline
    intensity: list[float]           # on grid, smoothed, roughly z-scored
    arousal: list[float]             # on grid, [-1,1]
    valence: list[float]             # on grid, [-1,1]
    novelty: list[float]             # on grid
    sections: list[Section]
    peak_time: float
    emotion_source: Literal["mert", "heuristic"]
    beat_source: Literal["beat_this", "librosa"]


# ---------- Director ----------
class SegmentDirective(BaseModel):
    anchor: Literal["intro", "build", "peak", "outro", "section"]
    section_index: int               # -1 unless anchor == "section"
    content_queries: list[str]       # concrete visual phrases for CLIP
    mood: Literal["calm", "neutral", "energetic", "match_music"]
    pacing: float                    # -1 (slower) .. +1 (faster); 0 = inherit global


class EditPlan(BaseModel):
    """Every field required (structured-output friendly). Empty string / list / 0 mean 'none'."""
    vibe: str
    vibe_queries: list[str]
    avoid_queries: list[str]
    pacing: float                    # -1..1
    energy_bias: float               # -1..1
    chronological: bool
    target_length_s: float           # 0 = auto (config default)
    excerpt_hint: Literal["auto", "start", "peak", "custom"]
    excerpt_start_s: float           # used only when excerpt_hint == "custom"
    opening_query: str
    closing_query: str
    segments: list[SegmentDirective]
    must_include: list[str]          # asset ids
    exclude: list[str]               # asset ids
    rationale: str                   # 1-2 sentences, shown to user
    source: Literal["llm", "rules"] = "rules"   # set by code, not by the LLM


# ---------- Planning ----------
class Slot(BaseModel):
    idx: int
    start: float
    end: float
    n_beats: int
    on_downbeat: bool
    intensity: float                 # mean over slot
    arousal: float
    valence: float
    section_idx: int
    boundary_strength: float         # strength of the cut at slot START (0..1); slot 0 = 0
    forced: bool                     # cut at start forced by a section boundary


class KenBurns(BaseModel):
    zoom_from: float
    zoom_to: float
    pan_x: float                     # fraction of frame width over the shot, signed
    pan_y: float


class ShotScores(BaseModel):
    importance: float
    mood: float
    director: float
    unary_total: float
    pairwise_in: float               # cost of transition from previous shot


class Shot(BaseModel):
    slot_idx: int
    start: float                     # excerpt timeline
    end: float
    frame_start: int                 # inclusive, output frame index
    frame_end: int                   # exclusive
    candidate_id: str
    asset_id: str
    kind: Literal["photo", "video"]
    src_in: float | None             # video source seconds
    src_out: float | None
    framing: Literal["native", "crop", "fill_blur", "pan"]
    kenburns: KenBurns | None
    scores: ShotScores
    reason: str                      # short human-readable "why this shot"


class EDL(BaseModel):
    run_id: str
    created: datetime
    system: str                      # "full", "naive", "beat_random", "greedy", or ablation name
    seed: int
    fps: int
    width: int
    height: int
    song_hash: str
    excerpt: Excerpt
    plan: EditPlan
    shots: list[Shot]
    total_cost: float
    notes: list[str] = []            # warnings (reuse allowed, few candidates, fallbacks used)


class RunResult(BaseModel):
    run_id: str
    edl_path: str
    video_path: str | None
    preview: bool
    editor_note: str
    timings: dict[str, float]
    fallbacks: list[str]             # e.g. ["director:rules", "emotion:heuristic"]
