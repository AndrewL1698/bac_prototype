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

## Importance ranker for photos (phase 15)
Last updated 2026-11-15 by Shiven (git phase-15-cufed-eval@bdb473e). Full report: `montage_data/eval/importance_cufed/report.md` (+ `metrics.json`, `per_album_tau.png`).

The 10A ranker learned from videos, but uploads are about 90% photos. CUFED (Wang et al., CVPR 2016) event albums with per-photo human importance: the 133 albums available (5,657 photos after ingest), run through the app's own feature pipeline. Same RankNet and features as 10A; 5-fold cross-validation by album (videos by video, folds as in 10A); Kendall tau / Spearman rho within each held-out album, averaged. Photos and videos are ranked separately in the app, so photos now get their own ranker (`weights/importance_ranker_photo.pt`) and the video ranker above is unchanged.

| Model (scoring photos) | CUFED Kendall tau | CUFED Spearman rho | SumMe tau | TVSum tau |
|---|---|---|---|---|
| Random | 0.003 | 0.004 | -0.002 | 0.002 |
| Aesthetic only | 0.154 | 0.212 | 0.136 | 0.015 |
| Heuristic (spec 04) | 0.166 | 0.222 | 0.123 | -0.018 |
| Video ranker as is (what photos used before) | 0.079 | 0.109 | 0.140 | (seen) |
| Learned, TVSum only (MLP) | 0.050 | 0.070 | 0.160 | 0.431 |
| Learned, SumMe + TVSum (linear) | 0.071 | 0.096 | 0.139 | 0.437 |
| Learned, all three (linear) | 0.283 | 0.376 | 0.105 | 0.404 |
| Learned, CUFED only (MLP) | 0.304 | 0.403 | 0.105 | 0.008 |
| **Learned, CUFED only (linear), alpha 1 (deployed photo ranker)** | **0.305** | **0.405** | 0.103 | 0.014 |

Gain over what photos used before: +0.226 tau per album (95% bootstrap CI +0.189 to +0.262), better on 86% of albums.

Takeaways: the video ranker did not transfer to photos; it ranked them worse than the plain heuristic (0.079 vs 0.166). Training on photo albums nearly quadruples photo ranking quality (0.305). The linear model is as good as the MLP, and using it alone beats blending it with the heuristic. The CUFED-trained models are poor on videos (SumMe 0.10, TVSum 0.01), which is why photos get a separate ranker instead of replacing the shared one: one shared ranker would have to trade photos against videos, and the video columns of this table never apply to the photo ranker. Honest limits: 133 albums (the only part of CUFED still downloadable), Flickr event albums rather than personal phone albums, and rankings against averaged rater scores.

CUFED photos are from YFCC100M (Yahoo Webscope, http://webscope.sandbox.yahoo.com; Thomee et al., arXiv:1503.01817); used under its terms (no redistribution, no identifying people).

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

## Story order (phase 17)
Last updated 2026-11-08 by Keshav (git phase-17-story-order@29f57f4). Plans only (no renders), seed 0, means over 7 album x song pairs: the demo album "Summer Trip (picked)" with its song, and the held-out eval albums sedona_1 and sedona_2 x the 3 eval songs. No brief unless noted. Story order = Kendall tau between shot position and capture time (1 = perfect order). Back jumps = shots taken more than 60 s before the previous shot. Parts covered = share of the album's parts (split at gaps over 1 hour) with at least one shot. Mood coupling and quality lift as in phase 11.

| Setup | Story order | Back jumps | Parts covered | Mood coupling | Quality lift |
|---|---|---|---|---|---|
| main today: Claude plan, order off | -0.07 | 18.6 | 73% | 0.49 | +0.012 |
| Order on, pairwise term only (w_chrono 1) | -0.03 | 6.0 | 72% | 0.34 | +0.001 |
| Order on, pairwise term only (w_chrono 30) | -0.12 | 3.4 | 70% | 0.24 | -0.001 |
| **Timeline term 4, Claude plan (new prompt)** | **0.66** | 4.0 | 69% | 0.39 | +0.019 |
| **Timeline term 4, Claude plan, generic brief** | **0.72** | 4.7 | 67% | 0.39 | +0.013 |
| **Timeline term 4, rules plan** | **0.90** | 2.3 | 70% | 0.38 | +0.050 |

Takeaways: the pairwise term alone never gave story order: the beam raced ahead to strong late shots and then had to jump back, and a heavier penalty only made fewer, bigger jumps. The timeline term (slot j draws from the same fraction of the album) fixes it. The price is mood coupling (about 0.39, the level of the phase 11 full system). Claude used to invent a per-section storyboard with no brief; with it, story order reached only 0.45, so the prompt now adds segments only when the brief names a part of the song. Coverage needs no extra term: about 70% of an album's parts get a shot and under one part of 3+ photos per album is missed.

## Render cache (phase 13)
Last updated 2026-11-08 by Keshav (git phase-17-story-order@d1813e9; render code as merged in PR #1). Colab A100, fresh runtime (empty shot cache), demo album "Summer Trip (picked)" (60 items), one-click generate (no brief, Claude director), 47 shots, 720p preview. Timings from the pipeline's stage timer.

| Run | Shots encoded | Render | Plan |
|---|---|---|---|
| Generate (empty shot cache) | 47 of 47 | 22.7 s | 21.6 s (about 20 s of it loads CLIP once per process) |
| Recut, one shot excluded | 26 of 47 (21 reused) | 11.2 s | 0.6 s |
| Fixtures (LOG 2026-11-05): 40-shot cut, recut changing 3 shots | 3 of 40 | 11.9 s cold, 1.5 s recut | |

Takeaways: on a real album a recut renders in about half the time of a cold render. The gain is smaller than on the fixtures because excluding one shot re-solves the whole cut: 36 of the 47 shots stay in it, but 15 of those move to a slot with a different length, so their cache key (in-point and length) changes and they are encoded again.

## Song vibe match, stage A (phase 12.8)
Album mood calibration (2026-11-17, T4; `scripts/calibrate_album_mood.py`): mean raw CLIP mood-pole score per album, `raw = 100 * (cos(high) - cos(low))`, pooled per asset. Albums: the 133 CUFED event albums (phase 15, `cufed_photos.npz`) plus our 3 (demo Summer Trip, eval sedona_1 and sedona_2). Mood = `tanh((raw - center) / scale)` with center = the mean over the 136 albums and scale = their std.

Config: `album_mood_center {valence 0.03, arousal 0.55}`, `album_mood_scale {valence 1.53, arousal 2.18}` (was `{0.28, -1.53}` / `{1.0, 0.6}`, fit on our 3 travel albums only on 2026-11-11).

| Albums (CUFED event type, or ours) | n | Raw valence | Raw arousal | Valence | Arousal | Mood word |
|---|---|---|---|---|---|---|
| Architecture | 3 | -2.19 | -1.37 | -0.90 | -0.71 | melancholic |
| BeachTrip | 20 | -1.43 | -1.93 | -0.74 | -0.81 | melancholic |
| Birthday | 12 | +2.22 | +1.92 | +0.89 | +0.56 | upbeat |
| BusinessActivity | 4 | -0.28 | +1.31 | -0.20 | +0.33 | energetic |
| CasualFamilyGather | 11 | +0.71 | +0.28 | +0.42 | -0.12 | bright |
| Christmas | 5 | +0.55 | +0.56 | +0.33 | +0.00 | bright |
| Cruise | 3 | -1.48 | -1.86 | -0.76 | -0.80 | melancholic |
| Graduation | 7 | +1.45 | +1.72 | +0.73 | +0.49 | upbeat |
| GroupActivity | 2 | +2.12 | +4.88 | +0.88 | +0.96 | upbeat |
| Halloween | 7 | +1.16 | +2.89 | +0.63 | +0.79 | upbeat |
| Museum | 3 | -0.91 | +0.03 | -0.55 | -0.24 | melancholic |
| NatureTrip | 3 | -1.33 | -2.79 | -0.71 | -0.91 | melancholic |
| PersonalArtActivity | 2 | -0.11 | -1.30 | -0.09 | -0.69 | calm |
| PersonalSports | 4 | -0.36 | +3.18 | -0.25 | +0.84 | intense |
| Protest | 4 | -1.14 | +1.59 | -0.64 | +0.44 | intense |
| ReligiousActivity | 3 | +0.36 | +0.43 | +0.21 | -0.05 | bright |
| Show | 9 | -0.66 | +2.07 | -0.43 | +0.60 | intense |
| Sports | 4 | +0.04 | +3.77 | +0.00 | +0.90 | energetic |
| ThemePark | 7 | -0.29 | +1.41 | -0.20 | +0.38 | intense |
| UrbanTrip | 6 | -1.84 | -0.85 | -0.84 | -0.57 | melancholic |
| Wedding | 8 | +1.98 | +1.45 | +0.85 | +0.39 | upbeat |
| Zoo | 6 | -0.14 | -1.43 | -0.11 | -0.72 | calm |
| Summer Trip (demo) | 1 | +0.44 | -1.15 | +0.26 | -0.65 | warm and calm (was energetic) |
| sedona_1 (eval) | 1 | +0.82 | -1.67 | +0.48 | -0.77 | warm and calm |
| sedona_2 (eval) | 1 | -0.44 | -1.77 | -0.30 | -0.79 | melancholic |

- Event types now spread out on arousal: parties, sports and Halloween high, zoo, nature and beach trips low. With the travel-only values (Shiven, PR #9 review), museum, wedding and sports albums all read as arousal ~1.
- Cost: measured against weddings and parties, every travel album reads calm, so Summer Trip's suggestions lean calmer than before.
- Limit: the valence poles favour smiling people, so scenery-heavy albums (beach, nature, city, architecture) read "melancholic". This was already true with the old values; candidate fix: valence prompts that do not hinge on people. 12.9 (MERT on the song's audio) replaces the song side of this comparison, not the album side.

Do different albums get different songs? (2026-11-18, T4, throwaway script, current code: rank-normalized terms, new calibration.) Songs: the 119 tags in `tag_mood.tsv` plus common genre tags, each as a one-tag song; vibe = 0.5 CLIP rank + 0.5 mood rank. Compared with the demo Summer Trip album (warm and calm; top 5: calm, mellow, peaceful, chill, chillout), one CUFED album per event type, a few shown:

| Album | CLIP cos to Summer Trip | Mood word | Spearman vs Summer Trip | Same top 10 | Top 5 |
|---|---|---|---|---|---|
| sedona_1 (eval) | 0.98 | warm and calm | +0.98 | 8/10 | (same trip as Summer Trip) |
| sedona_2 (eval) | 0.98 | melancholic | +0.93 | 6/10 | |
| Wedding | 0.75 | warm and calm | +0.86 | 4/10 | lovesong, chillout, love songs, mellow, peaceful |
| Birthday | 0.63 | upbeat | -0.20 | 1/10 | danceable, party, upbeat, triumphant, c pop |
| Sports | 0.51 | energetic | -0.23 | 2/10 | 10s, intense, epic, energetic, exciting |
| Protest | 0.66 | intense | -0.28 | 1/10 | aggressive, angry, alternative, intense, trap |
| Museum | 0.70 | melancholic | +0.67 | 1/10 | somber, melancholy, bittersweet, cpop, nostalgic |

- Different events get different songs. Our three eval/demo albums are photos from the same trip (CLIP cos 0.98; CUFED albums are 0.51-0.76 from Summer Trip), so the same shortlist for Summer Trip and sedona_1 is correct. The travel-only calibration told them apart only because its center sat between them.
- The CLIP term alone barely moves between albums (Spearman 0.96-0.98 across our three). Centering the album embedding on the mean CUFED album lowers that only to 0.86-0.96, so it is not worth the extra moving part; the mood term carries most of the album-to-album difference.
- Gap: "cpop", "c pop", "chinese" and "mandopop" are not in `tag_mood.tsv` (k-pop is), so songs tagged only with them are ranked by CLIP alone. Added in 12.9 (PR #15).

## Song audio mood, 12.9 (phase 12)
Song mood calibration (2026-11-26, T4; `scripts/calibrate_song_mood.py`): raw mood = the MERT emotion head's mean (valence, arousal) over a song. Fit on the 232 DEAM songs the head never trained on (174 test clips, 58 full songs); song mood = `tanh((raw - center) / scale)` with center = their mean and scale = their std.

Config: `song_mood_center {valence 0.11, arousal 0.16}`, `song_mood_scale {valence 0.2, arousal 0.25}` (were the neutral placeholders `{0, 0}` / `{1, 1}`).

| Song | Raw valence | Raw arousal | Valence | Arousal | Mood word |
|---|---|---|---|---|---|
| DEAM test (mean of 174) | +0.12 | +0.19 | | | |
| DEAM full (mean of 58) | +0.09 | +0.07 | | | |
| Calm - Spring Wilderness (demo) | -0.45 | -0.52 | -0.99 | -0.99 | melancholic |
| Life is a Highway (demo) | +0.72 | +0.71 | +1.00 | +0.98 | upbeat |
| deam_2019_energetic (eval) | +0.58 | +0.41 | +0.98 | +0.76 | upbeat |

- The head's outputs sit in a narrow band (std 0.2-0.25), so without the calibration every song read as neutral.
- Limit: DEAM is not chart pop. Our clear-cut songs saturate near +-1, so two upbeat pop songs may land close together; the ranking uses ranks of the match (`vibe._ranks`), which keeps their order. The offline study (12.10 PR) shows how well this separates 40 chart songs.

## Live vote (phase 11)
_Runs live at the workshop (Rate tab); `python -m montage.eval.vote --analyze` summarizes the votes._

## Takeaways
- Each learned component beats its fallback on held-out data: the MERT emotion head (test CCC 0.74 valence / 0.82 arousal vs 0 for predict-mean) and the importance ranker (Kendall tau SumMe 0.17 vs 0.12, TVSum 0.41 vs -0.02 for the heuristic).
- On held-out albums the editor does what the product promises: beat-exact cuts that follow the song's intensity and mood, steered by the brief, with no near-duplicates and frame-exact renders.
- Weak spots: moment-to-moment emotion within a song (arousal CCC about 0.1 per song), the full cut is a little less varied than random picks, and the numbers come from two albums with proxy metrics.
