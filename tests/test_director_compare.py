"""Director comparison harness (mocked model server; no API, GPU, or media)."""

from __future__ import annotations

import json

from montage.director import llm
from montage.eval import director_compare as dc
from montage.types import EditPlan


def test_checks_never_raise():
    plan = EditPlan.model_validate(dc.PREV)
    for case in dc.CASES:
        for fn in case.checks.values():
            assert isinstance(fn(plan), bool)
    assert len(dc.BRIEFS) == 20 and len(dc.NOTES) == 5


def test_parse_systems():
    s = dc.parse_systems("haiku=anthropic,qwen=openai_compat:Qwen/Qwen3-30B-A3B-Instruct-2507,rules")
    assert [x.name for x in s] == ["haiku", "qwen", "rules"]
    assert s[1].overrides["director.openai_compat.model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert s[0].overrides == {"director.provider": "anthropic", "director.use_llm": True}


def test_compare_end_to_end(monkeypatch, tmp_path):
    good = {k: v for k, v in dc.PREV.items() if k != "source"}
    good.update(pacing=0.5, target_length_s=45, avoid_queries=["blurry photo", "screenshot"])
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append(payload["model"])
        return {"choices": [{"message": {"content": json.dumps(good)}, "finish_reason": "stop"}]}
    monkeypatch.setattr(llm, "_post", fake_post)
    assert dc.main(["--systems", "open=openai_compat:tiny-model,rules", "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    by = {s["system"]: s for s in res["summary"]}
    assert by["open"]["valid"] == 25 and by["rules"]["valid"] == 25 and set(calls) == {"tiny-model"}
    assert 0 < by["open"]["checks_passed"] < by["open"]["checks_total"]
    md = (tmp_path / "report.md").read_text()
    assert "| open | tiny-model | 25/25 |" in md and "### messy" in md
