"""Editor's note: two plain sentences explaining the cut (LLM in phase 9A; template fallback)."""

from __future__ import annotations

from montage.log import get_logger
from montage.music.profile import mmss
from montage.types import EDL, MusicProfile

log = get_logger("editor_note")


def template_note(edl: EDL, profile: MusicProfile) -> str:
    peak = next((s for s in profile.sections if s.start <= profile.peak_time < s.end), None)
    label = peak.label if peak else "strongest"
    n_photo = sum(s.kind == "photo" for s in edl.shots)
    return (f"Cut {len(edl.shots)} shots ({n_photo} photos, {len(edl.shots) - n_photo} video clips) to "
            f"{profile.tempo:.0f} BPM; strongest moments placed in the {label} section at {mmss(profile.peak_time)}.")


def editor_note(edl: EDL, profile: MusicProfile, ctx: dict | None = None) -> tuple[str, list[str]]:
    from montage.config import get_config
    from montage.director import llm_enabled
    if get_config().director.editor_note and llm_enabled():
        try:
            from montage.director.llm import note_from_llm
            return note_from_llm(edl, profile, ctx or {}), []
        except Exception as e:  # noqa: BLE001  optional component
            log.warning("editor's note falls back to the template: %s", e)
    return template_note(edl, profile), []
