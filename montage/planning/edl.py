"""EDL assembly: frame quantization, framing, scores, and the human-readable reason per shot."""

from __future__ import annotations

from datetime import datetime

from montage.config import get_config
from montage.types import EDL, Candidate, EditPlan, Excerpt, Shot, Slot


def frame_bounds(slots: list[Slot], fps: int) -> list[int]:
    """f_k = round(t_k * fps) for all cut times at once (never per-shot durations)."""
    times = [s.start for s in slots] + [slots[-1].end]
    return [int(round(t * fps)) for t in times]


def make_reason(imp_rank: float, mood_gap: float, slot_label: str, query: str | None, opening: bool,
                closing: bool, vr: float) -> str:
    phrases = []
    if opening:
        phrases.append("opening shot you asked for")
    elif closing:
        phrases.append("closing shot you asked for")
    if query:
        phrases.append(f"matches '{query}'")
    if imp_rank >= 0.8:
        phrases.append("top-ranked moment")
    elif imp_rank >= 0.6:
        phrases.append("strong shot")
    if mood_gap < 0.2:
        feel = "energetic" if vr >= 0.6 else "calm" if vr <= 0.4 else "steady"
        phrases.append(f"{feel} footage for a {slot_label} section")
    if not phrases:
        phrases.append("keeps the cut varied")
    return "; ".join(phrases[:3])


def assemble(run_id: str, system: str, seed: int, song_hash: str, excerpt: Excerpt, plan: EditPlan,
             slots: list[Slot], picks: list[dict], total_cost: float, notes: list[str], preview: bool = True) -> EDL:
    """picks[j]: {cand, asset, src_in, src_out, framing, kenburns, scores, reason}."""
    cfg = get_config().render
    enc = cfg.preview if preview else cfg.export
    f = frame_bounds(slots, cfg.fps)
    shots = []
    for j, (s, p) in enumerate(zip(slots, picks)):
        c: Candidate = p["cand"]
        shots.append(Shot(slot_idx=j, start=s.start, end=s.end, frame_start=f[j], frame_end=f[j + 1],
                          candidate_id=c.id, asset_id=c.asset_id, kind=c.kind, src_in=p.get("src_in"),
                          src_out=p.get("src_out"), framing=p["framing"], kenburns=p.get("kenburns"),
                          scores=p["scores"], reason=p["reason"]))
    return EDL(run_id=run_id, created=datetime.now(), system=system, seed=seed, fps=cfg.fps, width=enc.width,
               height=enc.height, song_hash=song_hash, excerpt=excerpt, plan=plan, shots=shots,
               total_cost=round(float(total_cost), 4), notes=list(dict.fromkeys(notes)))

