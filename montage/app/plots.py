"""Under-the-hood plots: music curves with beats, sections and cuts; per-shot cost bars."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from montage.types import EDL, MusicProfile

SECTION_COLORS = {"low": "#cfd8dc", "building": "#ffe0b2", "high": "#c8e6c9", "peak": "#ffcdd2", "falling": "#d1c4e9"}


def curves_png(profile: MusicProfile, edl: EDL | None, out: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(profile.grid_times)
    fig, ax = plt.subplots(figsize=(16, 5.2), dpi=100)
    for s in profile.sections:
        ax.axvspan(s.start, s.end, color=SECTION_COLORS.get(s.label, "#eeeeee"), alpha=0.6, lw=0)
        ax.text((s.start + s.end) / 2, 2.55, s.label, ha="center", va="top", fontsize=12)
    ax.plot(t, profile.intensity, color="#c62828", lw=2.2, label="intensity")
    ax.plot(t, np.asarray(profile.arousal) * 2, color="#1565c0", lw=1.6, label="arousal x2")
    if any(abs(v) > 1e-6 for v in profile.valence):
        ax.plot(t, np.asarray(profile.valence) * 2, color="#2e7d32", lw=1.6, label="valence x2")
    for b in profile.beats:
        ax.plot([b, b], [-2.9, -2.65], color="#555", lw=0.6)
    for d in profile.downbeats:
        ax.plot([d, d], [-2.9, -2.4], color="#000", lw=1.6)
    if edl is not None:
        bounds = [s.start for s in profile.sections[1:]]
        for sh in edl.shots[1:]:
            forced = any(abs(sh.start - b) < 1e-3 for b in bounds)
            ax.axvline(sh.start, color="#6a1b9a" if forced else "#424242", lw=2.0 if forced else 0.8,
                       alpha=0.8, ls="-" if forced else "--")
    ax.set_xlim(0, max(t[-1] + 0.5, edl.shots[-1].end if edl else 0))
    ax.set_ylim(-3, 2.7)
    ax.set_xlabel("seconds (excerpt)", fontsize=12)
    ax.set_title(f"{profile.tempo:.0f} BPM | emotion: {profile.emotion_source} | beats: {profile.beat_source}"
                 f"{'' if edl is None else f' | {len(edl.shots)} cuts'}", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out


def cost_bars_png(edl: EDL, out: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(edl.shots))
    comps = {"importance": [s.scores.importance for s in edl.shots],
             "mood (cost)": [-s.scores.mood for s in edl.shots],
             "director": [s.scores.director for s in edl.shots],
             "pairwise (cost)": [-s.scores.pairwise_in for s in edl.shots]}
    fig, ax = plt.subplots(figsize=(16, 4), dpi=100)
    pos = np.zeros(len(x))
    neg = np.zeros(len(x))
    for name, v in comps.items():
        v = np.asarray(v, float)
        bottom = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=bottom, label=name)
        pos += np.maximum(v, 0)
        neg += np.minimum(v, 0)
    ax.set_xticks(x[::max(1, len(x) // 20)])
    ax.set_xlabel("shot", fontsize=12)
    ax.set_title("Why each shot: score components (higher is better)", fontsize=13)
    ax.legend(fontsize=10, ncol=4)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out
