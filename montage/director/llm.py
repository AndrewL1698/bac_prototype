"""Claude as the director: brief -> EditPlan, note -> revised EditPlan, and the editor's note.

Structured outputs (json_schema) on messages.create; enums are lowercased before validation; the plan is
repaired (unknown ids dropped, numbers clamped, queries capped). Any failure raises, and callers fall back
to the rule-based director.
"""

from __future__ import annotations

import base64
import json

from montage.config import get_config
from montage.director.prompts import EDITOR_NOTE_PROMPT, REVISE_INSTRUCTIONS, SYSTEM_PROMPT
from montage.director.schema import ENUM_FIELDS, SEGMENT_ENUM_FIELDS, json_schema
from montage.log import get_logger
from montage.types import EDL, EditPlan, MusicProfile

log = get_logger("llm")
_client = None


class LLMFailure(RuntimeError):
    pass


def get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(max_retries=1)
    return _client


def _text(resp) -> str:
    return next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")


def _blocks(ctx: dict, parts: list[tuple[str, str]]) -> list[dict]:
    blocks: list[dict] = []
    sheet = ctx.get("contact_sheet")
    if sheet:
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                   "data": base64.b64encode(sheet).decode("ascii")}})
        parts = [("Contact sheet", "The image above is a contact sheet; each tile shows its asset id.")] + parts
    text = "\n\n".join(f"## {title}\n{body}" for title, body in parts if body)
    blocks.append({"type": "text", "text": text})
    return blocks


def _structured_call(blocks: list[dict]) -> dict:
    cfg = get_config().director
    resp = get_client().messages.create(
        model=cfg.model, max_tokens=cfg.max_tokens, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": blocks}],
        output_config={"format": {"type": "json_schema", "schema": json_schema()}, "effort": "low"},
        timeout=cfg.timeout_s,
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise LLMFailure(f"stop_reason={resp.stop_reason}")
    try:
        return json.loads(_text(resp))
    except json.JSONDecodeError as e:
        raise LLMFailure(f"bad JSON: {e}") from e


def _lower_enums(data: dict) -> dict:
    for k in ENUM_FIELDS:
        if isinstance(data.get(k), str):
            data[k] = data[k].strip().lower()
    for seg in data.get("segments") or []:
        for k in SEGMENT_ENUM_FIELDS:
            if isinstance(seg.get(k), str):
                seg[k] = seg[k].strip().lower()
    return data


def repair(data: dict, asset_ids: list[str] | None) -> EditPlan:
    cfg = get_config()
    d = cfg.director
    clamp = lambda x: max(-1.0, min(1.0, float(x or 0)))  # noqa: E731

    def qs(lst) -> list[str]:
        out = []
        for q in lst or []:
            q = " ".join(str(q).split()[: d.max_query_words]).strip()
            if q and q not in out:
                out.append(q)
        return out[: d.max_queries]

    data = _lower_enums(dict(data))
    known = set(asset_ids or [])
    data["vibe_queries"] = qs(data.get("vibe_queries"))
    data["avoid_queries"] = qs(data.get("avoid_queries"))
    data["opening_query"] = " ".join(str(data.get("opening_query") or "").split()[: d.max_query_words])
    data["closing_query"] = " ".join(str(data.get("closing_query") or "").split()[: d.max_query_words])
    data["pacing"] = clamp(data.get("pacing"))
    data["energy_bias"] = clamp(data.get("energy_bias"))
    t = float(data.get("target_length_s") or 0)
    ex = cfg.music.excerpt
    data["target_length_s"] = 0.0 if t <= 0 else max(ex.min_length_s, min(ex.max_length_s, t))
    data["excerpt_start_s"] = max(0.0, float(data.get("excerpt_start_s") or 0))
    segs = []
    for s in data.get("segments") or []:
        s = dict(s)
        s["content_queries"] = qs(s.get("content_queries"))
        s["pacing"] = clamp(s.get("pacing"))
        if s.get("anchor") != "section":
            s["section_index"] = -1
        segs.append(s)
    data["segments"] = segs
    if asset_ids is not None:
        data["must_include"] = [a for a in data.get("must_include") or [] if a in known]
        data["exclude"] = [a for a in data.get("exclude") or [] if a in known]
    data["source"] = "llm"
    return EditPlan.model_validate(data)


def _plan_call(blocks: list[dict], asset_ids: list[str] | None) -> EditPlan:
    last: Exception | None = None
    for attempt in range(2):  # one retry, then the caller falls back to rules
        try:
            return repair(_structured_call(blocks), asset_ids)
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("LLM plan attempt %d failed: %s", attempt + 1, e)
    raise LLMFailure(str(last))


def plan_from_llm(brief: str, ctx: dict) -> EditPlan:
    blocks = _blocks(ctx, [("Director's brief", brief.strip() or "(no brief: make a fun recap matched to the music)"),
                           ("Song summary", ctx.get("song_summary", "")),
                           ("Media inventory (asset id, kind, duration, date | tags)", ctx.get("inventory", "")),
                           ("Tag vocabulary (phrases CLIP knows well)", ctx.get("tag_vocab", ""))])
    return _plan_call(blocks, ctx.get("asset_ids"))


def revise_from_llm(prev: EditPlan, edl: EDL | None, note: str, ctx: dict) -> EditPlan:
    prev_json = prev.model_dump(exclude={"source"})
    blocks = _blocks(ctx, [("Task", REVISE_INSTRUCTIONS), ("Director's note", note.strip()),
                           ("Previous plan (JSON)", json.dumps(prev_json, indent=1)),
                           ("Current cut (time-coded shots: time asset kind | tags)", ctx.get("edl_summary", "")),
                           ("Song summary", ctx.get("song_summary", "")),
                           ("Media inventory", ctx.get("inventory", ""))])
    return _plan_call(blocks, ctx.get("asset_ids"))


def note_from_llm(edl: EDL, profile: MusicProfile, ctx: dict) -> str:
    cfg = get_config().director
    plan = edl.plan
    body = "\n\n".join([f"## Plan\nvibe: {plan.vibe}\nrationale: {plan.rationale}\nqueries: {plan.vibe_queries}",
                        f"## Song summary\n{ctx.get('song_summary', '')}",
                        f"## Cut (time-coded shots)\n{ctx.get('edl_summary', '')}"])
    resp = get_client().messages.create(
        model=cfg.model, max_tokens=400, system=EDITOR_NOTE_PROMPT,
        messages=[{"role": "user", "content": body}], output_config={"effort": "low"}, timeout=cfg.timeout_s)
    text = _text(resp).strip()
    if resp.stop_reason == "refusal" or not text:
        raise LLMFailure("no editor note")
    return text
