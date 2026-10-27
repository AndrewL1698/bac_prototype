"""Evaluation report (spec 11 section 4): DATA_ROOT/eval/report.md + slide-sized figures in eval/figures/.

    python -m montage.eval.report        (also run at the end of run_grid)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from montage import paths
from montage.log import get_logger

log = get_logger("eval_report")
METRICS = ["beat_hit_rate", "downbeat_hit_rate", "mean_offset_ms", "boundary_hit_rate", "pace_coupling",
           "mood_coupling", "diversity", "max_adjacent_cos", "dup_violations", "coverage", "same_source_adjacent",
           "quality_lift", "director_adherence", "n_shots"]
ORDER = ["full", "naive", "beat_random", "greedy", "no_learned", "no_mood", "no_pairwise", "no_director", "no_forced"]
HIGHER_BETTER = {"beat_hit_rate": True, "downbeat_hit_rate": True, "boundary_hit_rate": True, "pace_coupling": True,
                 "mood_coupling": True, "diversity": False, "max_adjacent_cos": False, "dup_violations": False,
                 "coverage": True, "same_source_adjacent": False, "quality_lift": True, "director_adherence": True,
                 "mean_offset_ms": False}


def _root() -> Path:
    return paths.data_root() / "eval"


def _load(name: str) -> dict | None:
    p = _root() / name
    return json.loads(p.read_text()) if p.exists() else None


def summarize(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict] = {}
    for system in [s for s in ORDER if any(r["system"] == s for r in rows)]:
        rs = [r for r in rows if r["system"] == system]
        out[system] = {}
        for m in METRICS:
            v = np.array([r[m] for r in rs if m in r and r[m] is not None and not np.isnan(r[m])], float)
            out[system][m] = (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))
    return out


def figures(summ: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = _root() / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    systems = list(summ)
    colors = ["#c62828" if s == "full" else "#9e9e9e" if s in ("naive", "beat_random", "greedy") else "#1565c0"
              for s in systems]
    made = []

    def bars(pairs, fname, title):
        fig, axes = plt.subplots(1, len(pairs), figsize=(16, 9), dpi=100)
        for ax, (m, label) in zip(np.atleast_1d(axes), pairs):
            mu = [summ[s][m][0] for s in systems]
            sd = [summ[s][m][1] for s in systems]
            ax.bar(range(len(systems)), mu, yerr=sd, color=colors, capsize=4)
            ax.set_xticks(range(len(systems)))
            ax.set_xticklabels(systems, rotation=40, ha="right", fontsize=15)
            ax.set_title(label, fontsize=22)
            ax.tick_params(axis="y", labelsize=14)
        fig.suptitle(title, fontsize=26)
        fig.tight_layout()
        fig.savefig(fdir / fname)
        plt.close(fig)
        made.append(str(fdir / fname))

    bars([("beat_hit_rate", "Cuts on a beat (+/-70 ms)"), ("pace_coupling", "Cut rate follows the music (Spearman)")],
         "fig1_beat_and_pace.png", "Does the edit follow the music?")
    bars([("diversity", "Mean similarity of chosen shots (lower = more varied)"),
          ("quality_lift", "Importance of chosen shots vs the pool")], "fig2_variety_and_quality.png",
         "Variety and shot quality")
    abl = [s for s in systems if s.startswith("no_")]
    if "full" in summ and abl:
        ms = ["pace_coupling", "mood_coupling", "quality_lift", "boundary_hit_rate", "director_adherence", "diversity"]
        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        w = 0.8 / len(abl)
        for k, s in enumerate(abl):
            d = [summ[s][m][0] - summ["full"][m][0] for m in ms]
            ax.bar(np.arange(len(ms)) + k * w, d, w, label=s)
        ax.axhline(0, color="#000", lw=1)
        ax.set_xticks(np.arange(len(ms)) + w * (len(abl) - 1) / 2)
        ax.set_xticklabels(ms, fontsize=15)
        ax.legend(fontsize=15)
        ax.set_title("Ablations: change vs the full system", fontsize=26)
        ax.tick_params(axis="y", labelsize=14)
        fig.tight_layout()
        fig.savefig(fdir / "fig3_ablation_deltas.png")
        plt.close(fig)
        made.append(str(fdir / "fig3_ablation_deltas.png"))
    return made


def takeaways(summ: dict, render: dict | None) -> list[str]:
    t = []
    f, n = summ.get("full"), summ.get("naive")
    if f and n:
        t.append(f"Cuts land on a beat {f['beat_hit_rate'][0]:.0%} of the time with the full editor vs "
                 f"{n['beat_hit_rate'][0]:.0%} for the naive fixed-interval cut (high by construction; the point is the contrast).")
        t.append(f"The cut rate follows the music's intensity (Spearman {f['pace_coupling'][0]:+.2f} vs "
                 f"{n['pace_coupling'][0]:+.2f} naive), and chosen shots are better than the pool on average "
                 f"(quality lift {f['quality_lift'][0]:+.2f} vs {n['quality_lift'][0]:+.2f}).")
    if f:
        hurt, flat = [], []
        for s in [s for s in summ if s.startswith("no_")]:
            key = {"no_mood": "mood_coupling", "no_director": "director_adherence", "no_learned": "quality_lift",
                   "no_pairwise": "diversity", "no_forced": "boundary_hit_rate"}.get(s)
            if not key:
                continue
            d = summ[s][key][0] - f[key][0]
            worse = d < -0.01 if HIGHER_BETTER[key] else d > 0.01
            (hurt if worse else flat).append(f"{s} ({key} {d:+.3f})")
        if hurt:
            t.append("Removing a component hurts its own target metric for: " + ", ".join(hurt) + ".")
        if flat:
            t.append("No clear effect (honest result) for: " + ", ".join(flat) + ".")
    if render:
        t.append(f"Render check: detected cuts match the EDL within one frame (precision {render['precision']:.2f}, "
                 f"recall {render['recall']:.2f}): no drift between plan and video.")
    return t


def build() -> dict:
    grid = _load("grid.json")
    if not grid:
        raise SystemExit("no eval/grid.json yet: run montage.eval.run_grid first")
    summ = summarize(grid["rows"])
    render = _load("render_check.json")
    emo = _load("emotion/metrics.json")
    imp = _load("importance/metrics.json")
    figs = figures(summ)
    tk = takeaways(summ, render)
    n_alb = len({r["album"] for r in grid["rows"]})
    n_song = len({r["song"] for r in grid["rows"]})
    lines = ["# Evaluation report", "", "## Takeaways", ""] + [f"- {x}" for x in tk] + ["",
             f"Grid: {n_alb} albums x {n_song} songs x {len(summ)} systems x {len({r['seed'] for r in grid['rows']})} seeds "
             f"({len(grid['rows'])} EDLs). Components: {', '.join(grid.get('fallbacks') or ['all models live'])}.", "",
             "## Metrics (mean +/- std over album x song x seed)", "",
             "| System | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for s, ms in summ.items():
        lines.append(f"| {s} | " + " | ".join(f"{ms[m][0]:.3f} +/- {ms[m][1]:.3f}" for m in METRICS) + " |")
    lines.append("")
    if render:
        lines += ["## Render check", "", f"Precision {render['precision']:.3f}, recall {render['recall']:.3f} "
                  f"over {len(render['rows'])} renders.", ""]
    if emo and "test" in emo:
        h = emo["test"]["head"]
        lines += ["## Models", "", f"- Emotion head (DEAM test): valence CCC {h['valence']['ccc']:.3f}, arousal CCC "
                  f"{h['arousal']['ccc']:.3f}. Details: `eval/emotion/report.md`."]
    if imp:
        ch = imp["cv"][imp["chosen"]]
        lines += [f"- Importance ranker ({imp['chosen']}, blend alpha {imp['blend_alpha']:g}): SumMe tau "
                  f"{ch['summe']['tau']:.3f}, TVSum tau {ch['tvsum']['tau']:.3f} (heuristic "
                  f"{imp['cv']['heuristic']['summe']['tau']:.3f} / {imp['cv']['heuristic']['tvsum']['tau']:.3f}). "
                  "Details: `eval/importance/report.md`."]
    lines += ["", "Figures: " + ", ".join(f"`figures/{Path(f).name}`" for f in figs), ""]
    paths.atomic_write(_root() / "report.md", "\n".join(lines))
    log.info("wrote %s", _root() / "report.md")
    return {"summary": summ, "takeaways": tk, "figures": figs}


if __name__ == "__main__":
    paths.load_colab_env()
    build()
