"""The LLM director: brief -> EditPlan, note -> revised EditPlan, and the editor's note.

Two providers (`director.provider`): Claude (structured outputs on messages.create) or an open-weight model
behind an OpenAI-compatible server such as vLLM or Ollama (`response_format` json_schema, so decoding is
constrained to the plan schema). Either way enums are lowercased before validation and the plan is repaired
(unknown ids dropped, numbers clamped, queries capped). Any failure raises, and callers fall back to the
rule-based director.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request

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


# ---------- OpenAI-compatible servers (vLLM, Ollama, LM Studio) ----------

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _post(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMFailure(f"HTTP {e.code}: {e.read()[:300]!r}") from e
    except (urllib.error.URLError, OSError) as e:
        raise LLMFailure(f"server unreachable at {url}: {e}") from e


def _openai_messages(system: str, blocks: list[dict]) -> list[dict]:
    """Anthropic-style content blocks -> one OpenAI chat user message (images as data URLs)."""
    if all(b["type"] == "text" for b in blocks):
        content: str | list[dict] = "\n\n".join(b["text"] for b in blocks)
    else:
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:{b['source']['media_type']};base64,{b['source']['data']}"}}
                   if b["type"] == "image" else {"type": "text", "text": b["text"]} for b in blocks]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _openai_complete(system: str, blocks: list[dict], max_tokens: int, schema: dict | None) -> str:
    cfg = get_config().director
    oc = cfg.openai_compat
    payload: dict = {"model": oc.model, "messages": _openai_messages(system, blocks),
                     "max_tokens": max_tokens, "temperature": oc.temperature, **(oc.extra_body or {})}
    if schema is not None:
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "edit_plan", "schema": schema, "strict": True}}
    key = os.environ.get(oc.api_key_env, "") if oc.api_key_env else ""
    resp = _post(oc.base_url.rstrip("/") + "/chat/completions", payload,
                 {"Authorization": f"Bearer {key}"} if key else {}, cfg.timeout_s)
    try:
        choice = resp["choices"][0]
        text = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMFailure(f"unexpected response: {str(resp)[:300]}") from e
    if choice.get("finish_reason") == "length":
        raise LLMFailure("stop_reason=max_tokens")
    return _THINK.sub("", text).strip()


# ---------- provider switch ----------

def _complete(system: str, blocks: list[dict], max_tokens: int, schema: dict | None = None) -> str:
    """One model call -> response text. Raises LLMFailure on refusal, truncation, or transport errors."""
    cfg = get_config().director
    if cfg.provider == "openai_compat":
        return _openai_complete(system, blocks, max_tokens, schema)
    output_config: dict = {"effort": "low"}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    resp = get_client().messages.create(
        model=cfg.model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": blocks}], output_config=output_config, timeout=cfg.timeout_s,
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise LLMFailure(f"stop_reason={resp.stop_reason}")
    return _text(resp)


def _structured_call(blocks: list[dict]) -> dict:
    text = _complete(SYSTEM_PROMPT, blocks, get_config().director.max_tokens, json_schema())
    try:
        return json.loads(text)
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
    plan = edl.plan
    body = "\n\n".join([f"## Plan\nvibe: {plan.vibe}\nrationale: {plan.rationale}\nqueries: {plan.vibe_queries}",
                        f"## Song summary\n{ctx.get('song_summary', '')}",
                        f"## Cut (time-coded shots)\n{ctx.get('edl_summary', '')}"])
    text = _complete(EDITOR_NOTE_PROMPT, [{"type": "text", "text": body}], 400).strip()
    if not text:
        raise LLMFailure("no editor note")
    return text
