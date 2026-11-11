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

## Temporal music emotion head (phase 14)
Last updated 2026-11-11 by Andy (git phase-14-temporal-emotion@7a07514, T4). Full report: `montage_data/eval/emotion_temporal/report.md` (+ `metrics.json`, `curves_<song>.png`).

BiGRU over each song's MERT window sequence (spec 14), same MERT features, DEAM split and windows as 9B. Mean +- std over seeds 0/1/2. "Per song" = CCC inside each song, averaged over songs (follows the curve's shape); plain CCC is pooled over all songs (mostly the song's overall level). Both heads are shown with the app's 1 s smoothing.

| Model | Split | Arousal CCC per song | Valence CCC per song | Arousal CCC | Valence CCC | Arousal RMSE | Valence RMSE |
|---|---|---|---|---|---|---|---|
| Window head (9B, smoothed) | test | 0.096 +- 0.004 | 0.046 +- 0.004 | 0.822 +- 0.003 | 0.723 +- 0.019 | 0.168 | 0.173 |
| **Temporal head (deployed, smoothed)** | test | **0.177 +- 0.006** | **0.116 +- 0.008** | **0.838 +- 0.007** | **0.735 +- 0.011** | 0.163 | 0.174 |
| Window head (9B, smoothed) | full songs | 0.154 +- 0.010 | 0.013 +- 0.009 | 0.541 +- 0.013 | 0.393 +- 0.008 | 0.223 | 0.224 |
| Temporal head (smoothed) | full songs | 0.129 +- 0.025 | 0.084 +- 0.010 | 0.630 +- 0.014 | 0.404 +- 0.068 | 0.192 | 0.223 |

Deploy gate: passed (val score: temporal 0.435, seed 2, vs window 0.389). `weights/emotion_temporal.pt` is written; `emotion_head.pt` stays as the fallback. Only code on the phase 14 branch loads the new file until that branch is merged.

Takeaways: on the test clips, context almost doubles within-song tracking (arousal 0.10 -> 0.18, valence 0.05 -> 0.12), and the pooled numbers improve too, so it is not trading level for shape. The absolute within-song numbers are still low, so the curves are better, not good. On the 58 full songs, within-song valence and both pooled CCCs improve, but within-song arousal is slightly worse than the window head (0.13 vs 0.15, with a larger spread across seeds). The full songs only carry annotations from 15 to 45 s, so this is not yet a real test of length generalization. The annotator-lag study below separates lag from context.

### Annotator-lag study (phase 14.3)
2026-11-11 by Andy (git phase-14-temporal-emotion@c9f0051, T4). Report: `montage_data/eval/emotion_lag/report.md` (+ `metrics.json`). Each head is retrained with the annotation shifted `lag` seconds later than the audio (window at t paired with the annotation at t + lag), then scored on the shifted test targets. Seeds 0/1/2, smoothed as the app does.

| Lag (s) | Window head: arousal per song | Temporal head: arousal per song | Window head: valence per song | Temporal head: valence per song |
|---|---|---|---|---|
| 0 | 0.096 | **0.177** | 0.046 | 0.116 |
| 1 | 0.105 | 0.169 | 0.045 | 0.120 |
| 2 | 0.123 | 0.168 | 0.043 | 0.115 |
| 3 | **0.129** | 0.158 | 0.044 | 0.106 |
| 4 | 0.124 | 0.140 | 0.037 | 0.104 |

(Half-second steps and pooled CCC are in the report; pooled CCC stays 0.82-0.85 for both heads at every lag.)

Takeaways: lag explains part of the arousal gain, not all of it. Giving the window head its best lag (3 s) recovers about 40% of the arousal gap (0.096 -> 0.129, vs 0.177 for the temporal head). Lag explains none of the valence gain: the window head stays at 0.03-0.05 at every lag, while the temporal head stays at 0.10-0.13. The temporal head is best at lag 0 and declines as the lag grows, which suggests it already learned the annotators' delay. That means its curves likely sit a couple of seconds behind the music. Training it at about a 2 s lag costs little (0.168 vs 0.177) and would line its curves up with the audio; see the next section.

### Deployed: temporal head trained at a 2 s lag (phase 14.3 follow-up)
2026-11-11 by Andy (git phase-14-temporal-emotion@80eab64, T4; `python -m training.emotion.temporal --lag 2`). Report: `montage_data/eval/emotion_temporal/report.md`. The no-lag report and checkpoint are kept in `montage_data/eval/emotion_temporal_lag0/` (restore by copying `emotion_temporal_lag0.pt` back to `weights/emotion_temporal.pt`).

| Model (smoothed, 2 s lag) | Split | Arousal CCC per song | Valence CCC per song | Arousal CCC | Valence CCC |
|---|---|---|---|---|---|
| Window head | test | 0.123 +- 0.004 | 0.043 +- 0.007 | 0.834 | 0.732 |
| **Temporal head (deployed)** | test | **0.168 +- 0.010** | **0.115 +- 0.015** | 0.837 | 0.741 |
| Window head | full songs | 0.155 +- 0.009 | 0.012 +- 0.006 | 0.554 | 0.387 |
| Temporal head | full songs | 0.144 +- 0.010 | 0.068 +- 0.023 | 0.625 | 0.386 |

Timing check (test songs, arousal; best offset of the per-song cross-correlation, median; checked on a curve moved by a known 2 s first): the no-lag head's curves sit 0.5 s ahead of the annotations, the 2 s head's 2.0 s ahead, and in a direct comparison the 2 s head leads the no-lag head by 0.75 s.

Takeaways: the curves now lead the annotations by about 2 s, which, given that annotators trail the music by about 2-3 s (the window head's best lag), puts them close to the music's own timing. That is what the planner needs when it places shots on musical moments. The scores against the shifted targets are close to the no-lag head's (they are measured against different targets, so not strictly comparable). On the full songs, within-song arousal is better than the no-lag head (0.144 vs 0.129) and valence is worse (0.068 vs 0.084). Whether the earlier curves actually improve the edits is for the eval grid (14.4).

### Effect on the edits (phase 14.4)
2026-11-11 by Andy (git phase-14-temporal-emotion@4610e85, T4; `python -m training.emotion.edit_compare`). Report: `montage_data/eval/emotion_edit/report.md` (+ rows and EDLs per head). Full editor on the phase 11 eval set (2 albums x 3 songs x 3 seeds = 18 EDLs per head), same cached plans, same code; only the emotion head differs (9B window head vs the deployed 2 s temporal head). The phase 11 grid files were not touched.

| Metric | Window head (9B) | Temporal head |
|---|---|---|
| Mood coupling | +0.39 | +0.50 |
| Pace coupling | +0.65 | +0.53 |
| Brief adherence | +0.81 | +1.03 |
| Quality lift | +0.010 | +0.011 |
| Beat hit / section cuts | 100% / 100% | 100% / 100% |
| Shots | 35.8 | 34.5 |
| Arousal std inside the excerpt | 0.046 | 0.017 |

Screen time showing a different shot: 94% (89-100% per EDL pair). Per album x song, mood coupling goes up on 3 of 6 pairs (by +0.33 to +0.39), down slightly on 2 (-0.07, -0.08) and down on 1 (+0.24 -> -0.07).

Why the curves look flatter (whole eval songs, 1 s smoothed arousal): over a whole song both heads vary about as much (std 0.058-0.098 temporal vs 0.061-0.103 window), and running the temporal head in 30 s chunks changes little, so long songs are not the problem. Inside 30 s blocks the temporal head varies less (0.036-0.037 vs 0.049-0.066) and changes about half as much from step to step (0.003-0.005 vs 0.007-0.009): it follows slower trends and drops the window head's fast wiggles.

Takeaways: the new head leaves the hard guarantees intact (every cut on a beat, every section boundary cut) and changes which shots appear almost everywhere. The proxy metrics are mixed (mood coupling and brief adherence up, pace coupling down), and both coupling metrics are measured against each run's own curves, so they cannot say which edit is better. The temporal head is smoother within an excerpt. That matches the DEAM annotations, which are smooth, but it gives the arousal term less to steer cut speed with inside one excerpt. The 9B baseline here matches the phase 11 numbers (mood +0.39 vs +0.37, pace +0.65 vs +0.66), so the setup is consistent. Verdict: no measurable gain in the edits yet. A human side-by-side (render both cuts of a few album x song pairs and watch) is the next check before making the temporal head the default on `main`.

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
