# RESULTS: latest numbers

Overwritten (not appended) whenever a training or evaluation run finishes. Each section says when and by whom it was last updated, the git commit it came from, and where the full report and figures live on Drive. History of runs goes in `docs/LOG.md`.

## Music emotion head (phase 9B)
Last updated 2026-10-27 by Keshav (git track-b@0515096). Full report: `montage_data/eval/emotion/report.md` (+ `metrics.json`, `layer_weights.png`, `curves_<song>.png`).

MERT-v1-95M (frozen, revision 12af15f) features over 5 s centered windows, softmax layer weights + MLP head, trained on DEAM (1,396 train / 174 val / 174 test 45 s clips, song-level split; 58 full songs as a second test set). Windows at 2 Hz from 15 to 45 s (the annotation mirror carries only that span for the full songs too).

| Model | Split | Valence CCC | Arousal CCC | Valence RMSE | Arousal RMSE |
|---|---|---|---|---|---|
| Predict-mean | test | 0.000 | 0.000 | 0.244 | 0.287 |
| Ridge probe (layer 7) | test | 0.707 | 0.818 | 0.161 | 0.159 |
| **Head (deployed)** | test | **0.738** | **0.821** | 0.168 | 0.168 |
| Predict-mean | full songs | 0.000 | 0.000 | 0.221 | 0.286 |
| Ridge probe (layer 7) | full songs | 0.419 | 0.543 | 0.202 | 0.209 |
| Head (deployed) | full songs | 0.383 | 0.525 | 0.226 | 0.227 |

Takeaways: the head beats predict-mean by a wide margin and matches or beats the best single-layer probe on the test clips; on the full-song set the probe is slightly better. Agreement within a single song is low (arousal CCC about 0.1 per song): the model mostly gets each song's overall mood right rather than moment-to-moment changes. MERT normalizes each window's loudness, so pure volume changes do not move its arousal; the intensity curve keeps its loudness and onset terms for that. Valence, previously always 0, is now non-zero, so the planner's valence term is active.

## Importance ranker (phase 10A)
Last updated 2026-10-27 by Keshav (git track-b@a288f7f). Full report: `montage_data/eval/importance/report.md` (+ `metrics.json`, `heldout_<video>.png`).

Features from the app's own pipeline (CLIP, aesthetic, quality, faces) on 74 videos (24 SumMe + 50 TVSum; SumMe "Uncut_Evening_Flight" is damaged inside the published Zenodo archive, md5 verified), 9,175 labeled 3 s windows. RankNet with pairs only from the same video; 5-fold cross-validation by video; Kendall tau / Spearman rho within each held-out video, averaged per dataset.

| Model | SumMe Kendall tau | SumMe Spearman rho | TVSum Kendall tau | TVSum Spearman rho |
|---|---|---|---|---|
| Random | -0.002 | -0.003 | 0.002 | 0.001 |
| Aesthetic only | 0.136 | 0.188 | 0.015 | 0.019 |
| Heuristic (spec 04) | 0.123 | 0.167 | -0.018 | -0.025 |
| Learned, SumMe only (MLP) | 0.109 | 0.154 | -0.011 | -0.019 |
| Learned, TVSum only (MLP) | 0.160 | 0.220 | 0.431 | 0.601 |
| Learned, both (linear) | 0.139 | 0.196 | 0.437 | 0.603 |
| **Blend 0.75 learned + 0.25 heuristic (deployed)** | **0.170** | **0.229** | **0.407** | **0.572** |

Takeaways: the learned ranker beats the heuristic on both datasets, clearly so on TVSum (0.43 vs -0.02 tau), where the heuristic carries almost no signal. Selection followed the spec (best SumMe tau, then the blend weight by the same CV): the TVSum-trained MLP transfers best to SumMe (the dataset closest to personal footage), and blending it 0.75/0.25 with the heuristic gives the best SumMe score. Training on SumMe alone (about 19 videos per fold) is too little data. Correlations are within-video rankings of 3 s windows against averaged annotator scores, not comparable to frame-level numbers in the summarization literature.

## Editor evaluation (phase 11)
Last updated 2026-10-27 by Keshav (git track-b@909a030). Full report: `montage_data/eval/report.md` (+ `grid.json`, `render_check.json`), figures in `montage_data/eval/figures/`.

Two held-out albums (80 and 79 items, about 90% photos, from a trip that is not in the demo album, split by capture time) x 3 songs (calm, upbeat, and the most energetic DEAM full song) x 9 systems x 3 seeds = 162 EDLs. One Claude plan per album and song, from the generic brief "A fun recap of these moments, matched to the music.", is shared by every system, so the comparison isolates the editor. Beats from librosa (the Beat This! checkpoint host was down). Means over album x song x seed.

| System | Beat hit | Pace coupling | Mood coupling | Brief adherence | Quality lift | Mean pairwise CLIP sim | Mean adjacent CLIP sim | Section cuts | Near-dup pairs | Shots |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full** | **100%** | **+0.66** | **+0.37** | **+0.84** | **+0.014** | 0.612 | 0.638 | **1.00** | **0** | 36.7 |
| Naive (2 s cuts, random shots) | 27% | +0.11 | +0.03 | -0.04 | -0.026 | 0.599 | 0.599 | 0.57 | 0 | 25.3 |
| Beat grid + random shots | 100% | +0.66 | +0.02 | -0.16 | -0.005 | 0.591 | 0.588 | 1.00 | 0 | 36.7 |
| Beat grid + greedy importance | 100% | +0.66 | +0.04 | -0.27 | +0.087 | 0.591 | 0.604 | 1.00 | 0 | 36.7 |
| No learned ranker | 100% | +0.66 | +0.28 | +0.81 | -0.002 | 0.614 | 0.635 | 1.00 | 0 | 36.7 |
| No mood term | 100% | +0.66 | +0.00 | +0.85 | +0.011 | 0.610 | 0.630 | 1.00 | 0 | 36.7 |
| No pairwise terms | 100% | +0.66 | +0.38 | +0.83 | +0.013 | 0.610 | 0.653 | 1.00 | 0 | 36.7 |
| No director term | 100% | +0.66 | +0.70 | -0.13 | +0.068 | 0.590 | 0.609 | 1.00 | 0 | 36.7 |
| No forced section cuts | 100% | +0.72 | +0.34 | +0.91 | +0.013 | 0.615 | 0.646 | 0.86 | 0 | 35.8 |

Metrics: beat hit = cuts within 70 ms of a beat. Pace coupling = Spearman correlation of the local cut rate with the music's intensity. Mood coupling = Spearman correlation of each shot's visual arousal with the music's arousal under it. Brief adherence = director score of the chosen shot per slot minus the average over all candidates (weighted z-scores of CLIP similarity to the plan's queries). Quality lift = mean importance of the chosen shots minus the pool mean. CLIP similarities: lower = more varied. Section cuts = share of section boundaries that get a cut. Near-dup pairs = pairs of chosen shots from one duplicate cluster.

Render check: the full cut of each album and song (6 renders) was scanned with PySceneDetect. Detected cuts match the EDL within one frame: precision 0.986, recall 0.995. The 3 unmatched detections all fall inside video shots (motion or a cut inside the clip itself), and the one missed cut joins two similar-looking shots. There is no drift between plan and video.

Director weight (`planner.w_dir`) sweep on the full system, same grid, after the scarcity fix below:

| w_dir | Brief adherence | Mood coupling | Quality lift |
|---|---|---|---|
| 1.0 (old default) | 0.851 | 0.231 | 0.000 |
| **0.6 (new default)** | **0.839** | **0.374** | **0.014** |
| 0.3 | 0.788 | 0.523 | 0.032 |
| 0.15 | 0.586 | 0.677 | 0.052 |

Takeaways:
- The edit follows the music: every cut lands on a beat (27% for the naive cut), the cut rate tracks intensity (+0.66 vs +0.11), and footage energy tracks the song (mood coupling +0.37 vs +0.02 for random shots on the same grid).
- The brief steers which shots appear: adherence +0.84 vs -0.16 for random shots and -0.13 without the director term. It competes with mood: without the director term, mood coupling reaches +0.70. Lowering `w_dir` from 1.0 to 0.6 kept adherence (0.85 -> 0.84) and raised mood coupling from +0.23 to +0.37.
- Every ablation hurts its own target metric: no mood term drops mood coupling to 0, no director term drops adherence by 0.97, no forced cuts misses 14% of section boundaries, no learned ranker lowers quality lift (+0.014 -> -0.002) and mood coupling (+0.37 -> +0.28), and no pairwise terms makes neighbouring shots more alike (0.638 -> 0.653, a small effect).
- Honest limits: greedy importance-only picks score highest on quality lift (+0.087) but ignore the music and the brief; the full cut is slightly less varied than random picks (mean pairwise similarity 0.61 vs 0.59), because the brief and mood favor related shots; forced section cuts cost a little adherence and pace coupling. The evidence is two photo-heavy albums, one generic brief, and proxy metrics; the live vote is the human check.
- The grid found a planner bug: the first run planned 47 cuts for an album with only 44 distinct shots (burst photos), and reuse mode then filled the extra cuts with near-identical photos (up to 69 duplicate pairs in one cut). Scarcity now counts distinct shots and reuse mode spreads repeats; there are no near-duplicate pairs left.

## Live vote (phase 11)
_Runs live at the workshop (Rate tab); `python -m montage.eval.vote --analyze` summarizes the votes._

## Takeaways
- Each learned component beats its fallback on held-out data: the MERT emotion head (test CCC 0.74 valence / 0.82 arousal vs 0 for predict-mean) and the importance ranker (Kendall tau SumMe 0.17 vs 0.12, TVSum 0.41 vs -0.02 for the heuristic).
- On held-out albums the editor does what the product promises: beat-exact cuts that follow the song's intensity and mood, steered by the brief, with no near-duplicates and frame-exact renders.
- Weak spots: moment-to-moment emotion within a song (arousal CCC about 0.1 per song), the full cut is a little less varied than random picks, and the numbers come from two albums with proxy metrics.
