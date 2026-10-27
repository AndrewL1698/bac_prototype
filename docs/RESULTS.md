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
_Not run yet._ Full report: `montage_data/eval/importance/report.md`

| Model | SumMe Kendall tau | SumMe Spearman rho | TVSum Kendall tau | TVSum Spearman rho |
|---|---|---|---|---|

## Editor evaluation (phase 11)
_Not run yet._ Full report: `montage_data/eval/report.md`, figures in `montage_data/eval/figures/`

| System | Beat hit rate | Pace coupling | Mood coupling | Mean pairwise CLIP sim | Quality lift |
|---|---|---|---|---|---|

## Live vote (phase 11)
_Not run yet._

## Takeaways
_Pending._
