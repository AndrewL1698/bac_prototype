"""Pipeline API used by the CLI, the app and evaluation.

analyze -> make_plan -> plan_edit -> render_edl; generate() runs them all; recut() reuses a session;
export() re-renders a run at 1080p. Heavy per-session objects stay in memory; everything is also
cached on disk, so a fresh process rebuilds a session quickly.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel

from montage import cache, config, paths
from montage.log import get_logger, stage, timings
from montage.types import EDL, EditPlan, RunResult, UserError

log = get_logger("pipeline")
Progress = Callable[[float, str], None] | None

SYSTEM_OVERRIDES: dict[str, dict] = {
    "no_mood": {"planner.w_mood": 0.0, "planner.w_val": 0.0},
    "no_pairwise": {"planner.w_rep": 0.0, "planner.w_src": 0.0, "planner.w_jump": 0.0, "planner.w_contrast": 0.0,
                    "planner.w_chrono": 0.0},
    "no_director": {"planner.w_dir": 0.0},
    "no_forced": {"planner.use_forced_cuts": False},
}
SYSTEMS = ["full", "naive", "beat_random", "greedy", "no_learned", *SYSTEM_OVERRIDES]


class Session(BaseModel):           # persisted to DATA_ROOT/sessions/<session_id>.json
    session_id: str
    media_paths: list[str]
    song_path: str
    asset_ids: list[str]
    song_hash: str
    plan_history: list[EditPlan] = []
    notes_history: list[str] = []
    pins: list[str] = []            # candidate ids
    excluded: list[str] = []        # candidate ids
    runs: list[str] = []            # run ids, newest last


@dataclass
class State:
    media: object                    # MediaSet
    features: object                 # Features
    importance: object               # ImportanceScores
    music: object                    # FullAnalysis
    notes: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    seeds: dict[str, int] = field(default_factory=dict)


_states: dict[str, State] = {}
_lock = threading.RLock()


def _report(progress: Progress, f: float, msg: str) -> None:
    if progress:
        progress(f, msg)
    from montage import jobs
    jobs.report(f, msg)


def _session_path(sid: str) -> Path:
    return paths.sessions_dir() / f"{sid}.json"


def save_session(s: Session) -> None:
    paths.atomic_write(_session_path(s.session_id), s.model_dump_json(indent=1))


def load_session(session_id: str) -> Session:
    p = _session_path(session_id)
    if not p.exists():
        raise UserError(f"Unknown session {session_id}")
    return Session.model_validate_json(p.read_text(encoding="utf-8"))


def component_fallbacks(st: State) -> list[str]:
    fb = []
    if st.music.emotion_source == "heuristic":
        fb.append("emotion:heuristic")
    if st.music.beat_source == "librosa":
        fb.append("beats:librosa")
    if st.importance.source == "heuristic":
        fb.append("importance:heuristic")
    return fb


def _build_state(media_paths: list[str], song_path: str, progress: Progress = None) -> State:
    from montage.candidates import build_media
    from montage.features.extract import extract
    from montage.importance.blend import importance
    from montage.music.profile import analyze_song
    if not song_path:
        raise UserError("Please add a song.")
    _report(progress, 0.02, "reading media")
    with stage("ingest"):
        ms = build_media(media_paths, lambda f, m: _report(progress, 0.02 + 0.28 * f, m))
    with stage("features"):
        feats = extract(ms.assets, ms.candidates, ms.frames, lambda f, m: _report(progress, 0.3 + 0.35 * f, m))
    with stage("importance"):
        imp = importance(feats)
    _report(progress, 0.68, "listening to the song")
    with stage("music"):
        fa = analyze_song(song_path)
    st = State(media=ms, features=feats, importance=imp, music=fa, notes=list(ms.notes))
    st.fallbacks = component_fallbacks(st)
    return st


def analyze(media_paths: list[str], song_path: str, progress: Progress = None) -> Session:
    st = _build_state([str(p) for p in media_paths], str(song_path), progress)
    sid = f"{paths.user()}_{datetime.now():%Y%m%d-%H%M%S}_{secrets.token_hex(2)}"
    s = Session(session_id=sid, media_paths=[str(p) for p in media_paths], song_path=str(song_path),
                asset_ids=[a.id for a in st.media.assets], song_hash=st.music.song_hash)
    with _lock:
        _states[sid] = st
    save_session(s)
    return s


def state(s: Session, progress: Progress = None) -> State:
    with _lock:
        st = _states.get(s.session_id)
    if st is None:  # new process: rebuild from caches
        st = _build_state(s.media_paths, s.song_path, progress)
        with _lock:
            _states[s.session_id] = st
    return st


def _profile(st: State, plan: EditPlan):
    from montage.music.excerpt import choose_excerpt
    from montage.music.profile import build_profile
    ex = choose_excerpt(st.music, plan.target_length_s, plan.excerpt_hint, plan.excerpt_start_s)
    return build_profile(st.music, ex)


def director_context(s: Session, st: State, plan: EditPlan | None = None, edl: EDL | None = None) -> dict:
    from montage.director import context
    from montage.director.rules import plan_from_rules
    prof = _profile(st, plan or plan_from_rules(""))
    ctx = {"song_summary": context.song_summary(prof),
           "inventory": context.media_inventory(st.media.assets, st.media.candidates, st.features.table.tags),
           "tag_vocab": context.allowed_queries_hint(), "asset_ids": [a.id for a in st.media.assets]}
    if edl is not None:
        ctx["edl_summary"] = context.edl_summary(edl, st.features.table.tags)
    if config.get_config().director.send_thumbnails:
        try:
            ctx["contact_sheet"] = context.contact_sheet(st.media.assets, st.media.candidates, st.importance.final)
        except Exception as e:  # noqa: BLE001  optional
            log.warning("contact sheet failed: %s", e)
    return ctx


def make_plan(s: Session, brief: str) -> EditPlan:
    from montage import director
    st = state(s)
    plan, fb = director.make_plan(brief, director_context(s, st))
    plan = _sanitize(plan, st)
    st.fallbacks = sorted(set(component_fallbacks(st) + fb))
    s.plan_history.append(plan)
    save_session(s)
    return plan


def _sanitize(plan: EditPlan, st: State) -> EditPlan:
    known = {a.id for a in st.media.assets}
    plan.must_include = [a for a in plan.must_include if a in known]
    plan.exclude = [a for a in plan.exclude if a in known]
    return plan


def plan_edit(s: Session, plan: EditPlan, seed: int = 0, system: str = "full", run_id: str | None = None,
              preview: bool = True, persist: bool = True) -> EDL:
    """persist=False skips writing the run folder (evaluation grids plan many EDLs that are never rendered)."""
    from montage.director.apply import compile_plan
    from montage.importance.blend import importance
    from montage.planning import PlanInputs
    from montage.planning import plan_edit as _plan
    if system not in SYSTEMS:
        raise UserError(f"Unknown system {system!r}")
    st = state(s)
    run_id = run_id or paths.new_run_id()
    with stage("plan"):
        prof = _profile(st, plan)
        imp = st.importance if system != "no_learned" else importance(st.features, ranker=False)
        d = compile_plan(plan, prof, st.features, [c.asset_id for c in st.media.candidates],
                         pins=s.pins, excluded=s.excluded)
        inp = PlanInputs(st.media.assets, st.media.candidates, st.features, np.asarray(imp.final),
                         np.asarray(imp.heuristic), prof, d)
        with config.override(SYSTEM_OVERRIDES.get(system, {})):
            edl = _plan(inp, run_id, seed, system if system in ("naive", "beat_random", "greedy") else "full", preview)
    edl.system = system
    edl.notes = list(dict.fromkeys(edl.notes + st.notes))
    if not persist:
        return edl
    rd = paths.run_dir(run_id)
    cache.write_json(rd / "plan.json", json.loads(plan.model_dump_json()))
    paths.atomic_write(rd / "edl.json", edl.model_dump_json(indent=1))
    idx = st.features.index()
    faces = {c.id: [float(st.features.arrays["face_cx"][idx[c.id]]), float(st.features.arrays["face_cy"][idx[c.id]])]
             for c in st.media.candidates if c.id in {sh.candidate_id for sh in edl.shots}}
    cache.write_json(rd / "meta.json", {"session_id": s.session_id, "song_path": s.song_path, "faces": faces})
    return edl


def load_edl(run_id: str) -> EDL:
    p = paths.run_dir(run_id) / "edl.json"
    if not p.exists():
        raise UserError(f"Unknown run {run_id}")
    return EDL.model_validate_json(p.read_text(encoding="utf-8"))


def render_edl(edl: EDL, preview: bool = True) -> str:
    from montage.render.assemble import render_edl as _render
    meta = cache.read_json(paths.run_dir(edl.run_id) / "meta.json") or {}
    s = load_session(meta["session_id"])
    st = state(s)
    assets = {a.id: a for a in st.media.assets}
    faces = {k: tuple(v) for k, v in meta.get("faces", {}).items()}
    with stage("render"):
        return str(_render(edl, assets, meta["song_path"], preview=preview, faces=faces))


def _artifacts(s: Session, st: State, edl: EDL, plan: EditPlan) -> None:
    from montage.app.plots import cost_bars_png, curves_png
    from montage.app.storyboard import storyboard_jpg
    rd = paths.run_dir(edl.run_id)
    prof = _profile(st, plan)
    try:
        curves_png(prof, edl, rd / "curves.png")
        cost_bars_png(edl, rd / "costs.png")
        storyboard_jpg(edl, {c.id: c for c in st.media.candidates}, rd / "storyboard.jpg")
    except Exception as e:  # noqa: BLE001  plots are nice-to-have
        log.warning("could not write plots: %s", e)


def _run(s: Session, plan: EditPlan, seed: int, preview: bool, system: str, progress: Progress,
         t: dict[str, float], extra_fallbacks: list[str] | None = None) -> RunResult:
    from montage.director.editor_note import editor_note
    st = state(s)
    _report(progress, 0.75, "planning the edit")
    edl = plan_edit(s, plan, seed, system, preview=preview)
    _report(progress, 0.8, f"rendering {len(edl.shots)} shots")
    video = render_edl(edl, preview)
    with stage("editor_note"):
        note, _ = editor_note(edl, _profile(st, plan), director_context(s, st, plan, edl))
    _artifacts(s, st, edl, plan)
    fallbacks = sorted(set(st.fallbacks + (extra_fallbacks or [])))
    st.seeds[edl.run_id] = seed
    res = RunResult(run_id=edl.run_id, edl_path=str(paths.run_dir(edl.run_id) / "edl.json"), video_path=video,
                    preview=preview, editor_note=note, timings=dict(t), fallbacks=fallbacks)
    s.runs.append(edl.run_id)
    save_session(s)
    paths.atomic_write(paths.run_dir(edl.run_id) / "result.json", res.model_dump_json(indent=1))
    _report(progress, 1.0, "done")
    return res


def generate(media_paths: list[str], song_path: str, brief: str, seed: int = 0, preview: bool = True,
             system: str = "full", progress: Progress = None, overrides: dict | None = None) -> RunResult:
    with timings() as t:
        s = analyze(media_paths, song_path, progress)
        _report(progress, 0.72, "directing")
        with stage("director"):
            plan = make_plan(s, brief)
        if overrides:
            plan = plan.model_copy(update=overrides)
        res = _run(s, plan, seed, preview, system, progress, t)
    res.timings = dict(t)
    paths.atomic_write(paths.run_dir(res.run_id) / "result.json", res.model_dump_json(indent=1))
    return res


def session_for_run(run_id: str) -> Session:
    meta = cache.read_json(paths.run_dir(run_id) / "meta.json") or {}
    if "session_id" not in meta:
        raise UserError(f"Unknown run {run_id}")
    return load_session(meta["session_id"])


def recut(session_id: str, note: str | None = None, pins: list[str] | None = None, excluded: list[str] | None = None,
          seed: int | None = None, preview: bool = True, progress: Progress = None, system: str = "full") -> RunResult:
    from montage.director.notes import revise_plan
    with timings() as t:
        s = load_session(session_id)
        st = state(s, progress)
        if pins is not None:
            s.pins = list(dict.fromkeys(pins))
        if excluded is not None:
            s.excluded = list(dict.fromkeys(excluded))
        if not s.plan_history:
            raise UserError("Generate a cut first.")
        plan = s.plan_history[-1]
        fb: list[str] = []
        if note and note.strip():
            prev_edl = load_edl(s.runs[-1]) if s.runs else None
            with stage("director"):
                plan, fb = revise_plan(plan, prev_edl, note, director_context(s, st, plan, prev_edl))
            plan = _sanitize(plan, st)
            s.notes_history.append(note.strip())
            s.plan_history.append(plan)
        if seed is None:
            seed = st.seeds.get(s.runs[-1], 0) if s.runs else 0
        res = _run(s, plan, seed, preview, system, progress, t, fb)
    res.timings = dict(t)
    paths.atomic_write(paths.run_dir(res.run_id) / "result.json", res.model_dump_json(indent=1))
    return res


def export(run_id: str, progress: Progress = None) -> str:
    edl = load_edl(run_id)
    from montage.config import get_config
    enc = get_config().render.export
    edl = edl.model_copy(update={"width": enc.width, "height": enc.height})
    _report(progress, 0.1, "rendering 1080p")
    return render_edl(edl, preview=False)
