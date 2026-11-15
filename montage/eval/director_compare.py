"""Compare LLM directors (Claude vs an open-weight model vs rules) on fixed briefs and notes.

    python -m montage.eval.director_compare --systems haiku=anthropic,qwen=openai_compat,rules

Text only: a fixed synthetic song summary and media inventory, so no media, GPU, or features are needed (the
open-weight model must already be served, see scripts/serve_llm.py). 21 briefs and 5 revision notes, each with
objective checks taken from the director prompt's rules (e.g. "45 seconds" -> target_length_s 45). Reports, per
system: valid plans (no fallback), checks passed, mean latency; plus each plan's queries and rationale for a
side-by-side taste read. Writes DATA_ROOT/eval/director_compare/<user>_<stamp>/{results.json, report.md}.

System spec: `name=provider[:model]`, provider anthropic | openai_compat, or the bare word `rules`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from montage import jobs, paths
from montage.config import override, set_overrides
from montage.log import get_logger
from montage.types import EditPlan

log = get_logger("director_compare")

Check = Callable[[EditPlan], bool]

INVENTORY = "\n".join([
    "a000 photo 2026-07-14 | beach, ocean, sunrise", "a001 video 0:20 2026-07-14 | road trip, car, highway",
    "a002 photo 2026-07-14 | group of friends, party, dancing", "a003 photo 2026-07-15 | sunset, beach",
    "a004 video 0:15 2026-07-15 | mountain, hiking, forest", "a005 photo 2026-07-15 | cake, birthday party",
    "a006 photo 2026-07-16 | screenshot, text document", "a007 photo 2026-07-16 | dog, park, grass",
    "a008 photo 2026-06-02 | graduation, cap and gown, stage", "a009 photo 2026-06-02 | family portrait, smiling",
    "a010 photo 2026-07-16 | selfie, close-up face", "a011 video 0:12 2026-07-15 | mountain summit, view, clouds",
])
SONG = ("Song: 118 BPM, length 3:30; excerpt 0:50 (starts at song 1:02). Peak at 0:36.\n"
        "S0 0.0-12.0s low (intro)\nS1 12.0-28.0s building\nS2 28.0-44.0s peak\nS3 44.0-50.0s falling (outro)")
TAGS = ("beach, ocean, sunrise, sunset, road trip, car, party, dancing, group of friends, mountain, hiking, "
        "summit, cake, birthday, dog, park, graduation, family, selfie, screenshot, blurry photo")
ASSET_IDS = [f"a{i:03d}" for i in range(12)]
EDL_SUMMARY = "\n".join(["0:00-0:03 a000 photo | beach, sunrise", "0:03-0:06 a001 video | road trip, car",
                         "0:06-0:10 a007 photo | dog, park", "0:10-0:13 a003 photo | sunset, beach",
                         "0:13-0:16 a002 photo | party, dancing", "0:16-0:20 a005 photo | cake, birthday"])
CTX = {"song_summary": SONG, "inventory": INVENTORY, "tag_vocab": TAGS, "asset_ids": ASSET_IDS,
       "edl_summary": EDL_SUMMARY}


def _q(p: EditPlan) -> list[str]:
    """All positive queries, lowercased."""
    qs = p.vibe_queries + [p.opening_query, p.closing_query] + [q for s in p.segments for q in s.content_queries]
    return [q.lower() for q in qs if q]


def has(words: list[str], *keys: str) -> bool:
    return any(k in w for w in words for k in keys)


def seg(p: EditPlan, anchor: str) -> list:
    return [s for s in p.segments if s.anchor == anchor]


@dataclass
class Case:
    id: str
    text: str
    checks: dict[str, Check]
    note: bool = False                       # revision note against PREV (else a fresh brief)


BRIEFS = [
    Case("summer", "Warm, nostalgic summer recap. Open slow on the beach at sunrise, build through the road trip, save "
         "the party for the drop, end on the sunset. Skip anything blurry.",
         {"opening sunrise/beach": lambda p: has([p.opening_query.lower()], "sunrise", "beach"),
          "closing sunset": lambda p: "sunset" in p.closing_query.lower(),
          "party at peak": lambda p: any(has([q.lower() for q in s.content_queries], "party", "danc")
                                         for s in seg(p, "peak")),
          "avoid blurry": lambda p: has([q.lower() for q in p.avoid_queries], "blur")}),
    Case("birthday", "Hype birthday montage, fast cuts, lots of friends and cake, 45 seconds.",
         {"faster": lambda p: p.pacing > 0, "45 s": lambda p: p.target_length_s == 45,
          "cake query": lambda p: has(_q(p), "cake")}),
    Case("hike", "Calm, chronological travel diary of our mountain hike; no screenshots; finish on the summit view.",
         {"chronological": lambda p: p.chronological, "not faster": lambda p: p.pacing <= 0,
          "avoid screenshots": lambda p: has([q.lower() for q in p.avoid_queries], "screenshot"),
          "closing summit": lambda p: has([p.closing_query.lower()], "summit", "mountain", "view")}),
    Case("defaults", "Just make something fun.",
         {"pacing 0": lambda p: p.pacing == 0, "length auto": lambda p: p.target_length_s == 0,
          "excerpt auto": lambda p: p.excerpt_hint == "auto", "chronological": lambda p: p.chronological,
          "no pins": lambda p: not p.must_include and not p.exclude}),
    Case("length_start", "Make it exactly 50 seconds and start the song from the very beginning.",
         {"50 s": lambda p: p.target_length_s == 50, "excerpt start": lambda p: p.excerpt_hint == "start"}),
    Case("climax", "Center the whole thing on the climax of the song.",
         {"excerpt peak": lambda p: p.excerpt_hint == "peak"}),
    Case("custom_start", "Start the song at 1:15.",
         {"excerpt custom": lambda p: p.excerpt_hint == "custom",
          "starts 75 s": lambda p: abs(p.excerpt_start_s - 75) <= 1}),
    Case("slower", "Slower and dreamier overall, nothing frantic.",
         {"slower": lambda p: p.pacing < 0}),
    Case("pins_ids", "Make sure a007 is in it, and leave out a006.",
         {"include a007": lambda p: "a007" in p.must_include, "exclude a006": lambda p: "a006" in p.exclude}),
    Case("pin_by_desc", "Please include the picture of the dog at the park.",
         {"include a007": lambda p: "a007" in p.must_include}),
    Case("hike_on_drop", "Put the mountain hike footage on the drop.",
         {"hike at peak": lambda p: any(has([q.lower() for q in s.content_queries], "mountain", "hik")
                                        for s in seg(p, "peak"))}),
    Case("intro_peak_moods", "Intro should be calm ocean shots, then go energetic at the peak.",
         {"calm intro": lambda p: any(s.mood == "calm" for s in seg(p, "intro")),
          "ocean in intro": lambda p: any(has([q.lower() for q in s.content_queries], "ocean", "beach", "sea")
                                          for s in seg(p, "intro")),
          "energetic peak": lambda p: any(s.mood == "energetic" for s in seg(p, "peak"))}),
    Case("section_ref", "Put the road trip clips in section S1.",
         {"section 1": lambda p: any(s.anchor == "section" and s.section_index == 1 for s in p.segments)
          or any(has([q.lower() for q in s.content_queries], "road", "car") for s in seg(p, "build"))}),
    Case("chrono", "In order of when things happened, please.",
         {"chronological": lambda p: p.chronological}),
    Case("no_people", "Open road landscapes, make sure it has no people.",
         {"people avoided": lambda p: has([q.lower() for q in p.avoid_queries], "people", "person", "friend", "couple", "selfie", "child", "famil", "crowd"),
          "no people in vibe": lambda p: not has([q.lower() for q in p.vibe_queries], "people", "person")}),
    Case("avoids", "No selfies, no screenshots, nothing blurry.",
         {"avoid selfie": lambda p: has([q.lower() for q in p.avoid_queries], "selfie"),
          "avoid screenshot": lambda p: has([q.lower() for q in p.avoid_queries], "screenshot"),
          "avoid blurry": lambda p: has([q.lower() for q in p.avoid_queries], "blur")}),
    Case("party_energy", "A high-energy party edit.",
         {"more energy or pace": lambda p: p.energy_bias > 0 or p.pacing > 0,
          "party query": lambda p: has(_q(p), "party", "danc")}),
    Case("grandma", "Something for my grandma: gentle, family-focused, 60 seconds.",
         {"60 s": lambda p: p.target_length_s == 60, "not faster": lambda p: p.pacing <= 0,
          "family query": lambda p: has(_q(p), "family")}),
    Case("open_close", "Open on the cake and close on the group of friends.",
         {"opening cake": lambda p: "cake" in p.opening_query.lower(),
          "closing friends": lambda p: has([p.closing_query.lower()], "friend", "group")}),
    Case("outro_energy", "Around 55 seconds, and make the ending energetic.",
         {"55 s": lambda p: p.target_length_s == 55,
          "energetic outro": lambda p: any(s.mood == "energetic" for s in seg(p, "outro"))}),
    Case("messy", "ok so its my sisters graduation, mostly her + family, skip the dog stuff, fast at the end",
         {"avoid dog": lambda p: has([q.lower() for q in p.avoid_queries], "dog"),
          "faster outro": lambda p: any(s.pacing > 0 for s in seg(p, "outro")),
          "graduation query": lambda p: has(_q(p), "graduat", "cap and gown")}),
]

NOTES = [
    Case("n_faster", "faster", {"faster": lambda p: p.pacing > 0,
                                "keeps closing": lambda p: "sunset" in p.closing_query.lower()}, note=True),
    Case("n_lose_012", "lose the shot at 0:12", {"exclude a003": lambda p: "a003" in p.exclude,
                                                 "keeps vibe": lambda p: has(p.vibe_queries, "beach")}, note=True),
    Case("n_more_dog", "more of the dog", {"dog query": lambda p: has(_q(p), "dog"),
                                           "keeps pacing": lambda p: p.pacing == 0}, note=True),
    Case("n_no_screens", "no screenshots", {"avoid screenshot": lambda p: has([q.lower() for q in p.avoid_queries],
                                                                              "screenshot")}, note=True),
    Case("n_slow_middle", "too slow in the middle",
         {"middle faster": lambda p: any(s.pacing > 0 for s in p.segments if s.anchor in ("build", "peak"))
          or p.pacing > 0}, note=True),
]
CASES = BRIEFS + NOTES

PREV = {"vibe": "sunny summer recap", "vibe_queries": ["golden hour beach", "friends laughing"], "avoid_queries": [],
        "pacing": 0.0, "energy_bias": 0.0, "chronological": False, "target_length_s": 0, "excerpt_hint": "auto",
        "excerpt_start_s": 0, "opening_query": "sunrise over the ocean", "closing_query": "sunset on the beach",
        "segments": [{"anchor": "peak", "section_index": -1, "content_queries": ["friends dancing at a party"],
                      "mood": "match_music", "pacing": 0.0}],
        "must_include": [], "exclude": [], "rationale": "Beach to open and close, party on the drop.",
        "source": "llm"}


@dataclass
class System:
    name: str
    provider: str                  # anthropic | openai_compat | rules
    model: str = ""
    overrides: dict = field(default_factory=dict)


def parse_systems(spec: str) -> list[System]:
    out = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        if item == "rules":
            out.append(System("rules", "rules"))
            continue
        name, _, rest = item.partition("=")
        provider, _, model = rest.partition(":")
        if provider not in ("anthropic", "openai_compat"):
            raise ValueError(f"bad system {item!r}: use name=anthropic[:model], name=openai_compat[:model], or rules")
        ov: dict = {"director.provider": provider, "director.use_llm": True}
        if model:
            ov["director.model" if provider == "anthropic" else "director.openai_compat.model"] = model
        out.append(System(name, provider, model, ov))
    return out


def run_case(system: System, case: Case) -> dict:
    from montage.director import llm
    from montage.director.notes import revise_rules
    from montage.director.rules import plan_from_rules
    prev = EditPlan.model_validate(PREV)
    t0 = time.perf_counter()
    plan, error = None, ""
    try:
        if system.provider == "rules":
            plan = revise_rules(prev, None, case.text) if case.note else plan_from_rules(case.text)
        else:
            with override(system.overrides):
                plan = (llm.revise_from_llm(prev, None, case.text, CTX) if case.note
                        else llm.plan_from_llm(case.text, CTX))
    except Exception as e:  # noqa: BLE001  a failure here is what the app would hand to the rules fallback
        error = f"{type(e).__name__}: {e}"[:300]
    secs = time.perf_counter() - t0
    checks = {}
    for name, fn in case.checks.items():
        try:
            checks[name] = bool(plan is not None and fn(plan))
        except Exception:  # noqa: BLE001
            checks[name] = False
    return {"system": system.name, "case": case.id, "note": case.note, "text": case.text, "seconds": round(secs, 2),
            "valid": plan is not None, "error": error, "checks": checks,
            "plan": plan.model_dump(mode="json") if plan is not None else None}


def summarize(rows: list[dict], systems: list[System]) -> list[dict]:
    out = []
    for s in systems:
        r = [x for x in rows if x["system"] == s.name]
        n_checks = sum(len(x["checks"]) for x in r)
        llm_r = [x for x in r if x["valid"]]
        out.append({"system": s.name, "provider": s.provider, "model": s.model, "cases": len(r),
                    "valid": sum(x["valid"] for x in r),
                    "checks_passed": sum(sum(x["checks"].values()) for x in r), "checks_total": n_checks,
                    "brief_checks": _rate([x for x in r if not x["note"]]),
                    "note_checks": _rate([x for x in r if x["note"]]),
                    "mean_s": round(sum(x["seconds"] for x in llm_r) / max(1, len(llm_r)), 2)})
    return out


def _rate(rows: list[dict]) -> str:
    tot = sum(len(x["checks"]) for x in rows)
    ok = sum(sum(x["checks"].values()) for x in rows)
    return f"{ok}/{tot}"


def report_md(summary: list[dict], rows: list[dict]) -> str:
    lines = ["# Director comparison", "", f"{len(BRIEFS)} briefs + {len(NOTES)} notes, fixed synthetic context.", "",
             "| system | model | valid plans | checks (briefs) | checks (notes) | total | mean latency |",
             "|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['system']} | {s['model'] or s['provider']} | {s['valid']}/{s['cases']} | "
                     f"{s['brief_checks']} | {s['note_checks']} | {s['checks_passed']}/{s['checks_total']} | "
                     f"{s['mean_s']} s |")
    lines += ["", "## Failed checks", ""]
    for x in rows:
        bad = [k for k, v in x["checks"].items() if not v]
        if bad or x["error"]:
            lines.append(f"- **{x['case']}** / {x['system']}: {', '.join(bad)}{' | ' + x['error'] if x['error'] else ''}")
    lines += ["", "## Plans side by side (queries and rationale)", ""]
    for case in CASES:
        lines.append(f"### {case.id}: {case.text}")
        for x in [r for r in rows if r["case"] == case.id]:
            p = x["plan"]
            if p is None:
                lines.append(f"- {x['system']}: (failed) {x['error']}")
                continue
            segs = "; ".join(f"{s['anchor']}:{s['mood']}:{s['content_queries']}" for s in p["segments"])
            lines.append(f"- {x['system']}: vibe {p['vibe_queries']} | avoid {p['avoid_queries']} | "
                         f"open '{p['opening_query']}' close '{p['closing_query']}' | segs {segs or '-'} | "
                         f"\"{p['rationale']}\"")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="haiku=anthropic,open=openai_compat,rules")
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    ap.add_argument("--out", default="")
    ap.add_argument("--set", action="append", default=[], help="config override a.b=value (e.g. the server URL)")
    args = ap.parse_args(argv)
    paths.load_colab_env()
    if args.set:
        set_overrides(args.set)
    systems = parse_systems(args.systems)
    cases = [c for c in CASES if not args.cases or c.id in args.cases.split(",")]
    rows = []
    total = len(systems) * len(cases)
    for i, (s, c) in enumerate([(s, c) for s in systems for c in cases]):
        jobs.report(i / max(1, total), f"{s.name} / {c.id}")
        rows.append(run_case(s, c))
        log.info("%s %s valid=%s %.1fs %s", s.name, c.id, rows[-1]["valid"], rows[-1]["seconds"],
                 rows[-1]["error"] or "")
    summary = summarize(rows, systems)
    out = Path(args.out) if args.out else (paths.data_root() / "eval" / "director_compare"
                                           / f"{paths.user()}_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    paths.atomic_write(out / "results.json", json.dumps({"summary": summary, "rows": rows}, indent=1))
    md = report_md(summary, rows)
    paths.atomic_write(out / "report.md", md)
    print(md.split("## Failed checks")[0])
    print(f"full report: {out / 'report.md'}")
    jobs.report(1.0, f"done: {out}", force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
