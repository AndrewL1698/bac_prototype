# Music + emotion track: guide

Owner: Andy. Branch: `phase-14-temporal-emotion`. Spec: [`spec/14_temporal_emotion.md`](spec/14_temporal_emotion.md) (binding for the details; this file is the overview and run book).

> **Status (2026-11-10).** This is Track 2 of the 8-week plan (2026-11-04). The 2-week re-scope of 2026-11-07 (`ROADMAP.md`, `PLAN.md`) dropped phase 14 and reused "Track 2" for CUFED photo importance. Andy is continuing this track and is raising the re-scope with the team; until that is settled, `PLAN.md` on `main` does not list this work. Phase 14 code and tests are done; the DEAM training run (14.1) finished on 2026-11-11 and the temporal head is deployed (`RESULTS.md`).

## 1. Why this track exists

The montage is cut to the music: the song's **emotion curves** (valence = how positive, arousal = how energetic, both on a 2 Hz grid in [-1, 1]) decide a lot of the edit:

| Where | How the curves are used | Code |
|---|---|---|
| Cut speed | `intensity` = 0.5 x arousal + 0.3 x loudness + 0.2 x onset density (z-scored, smoothed). High intensity -> shorter shots (1-2 beats), low -> longer (4-8). | `music/profile.py: compute_intensity`, `planning/slots.py: walk` |
| Shot choice (mood term) | Each slot's music arousal (rank) is matched against each photo's visual arousal (`planner.w_mood` 0.8), and valence against visual valence (`planner.w_val` 0.3, only when the curves come from MERT). | `planning/costs.py: build_costs` |
| Song structure | Section labels (low / building / high / peak / falling) and the peak time come from the intensity curve. The director maps "the drop", "the intro" etc. onto them. | `music/profile.py: label_sections`, `director/apply.py: map_anchor` |
| Excerpt | The 45-60 s excerpt is chosen to place the intensity peak about two thirds in. | `music/excerpt.py` |

**The problem.** The current model (phase 9B: frozen MERT features per 5 s window + a small MLP head) gets each song's *overall* mood right (DEAM test CCC 0.74 valence, 0.82 arousal) but barely follows changes *inside* a song: arousal CCC within a song is about **0.1** (`RESULTS.md`, 9B). The planner reads the curves shot by shot, so within-song movement is exactly what the edit needs: a calm verse and a big chorus should get different shots and different cut speeds.

**The fix (phase 14).** Let each prediction see its neighbours: a small bidirectional GRU over the song's sequence of MERT windows, trained with a loss that rewards following the curve's shape within a song, not just its level.

## 2. How music analysis works today

```
song file
  -> load (22,050 Hz mono for DSP; 24 kHz for MERT)                    music/load.py
  -> beats + downbeats: Beat This! (GPU) or librosa fallback           music/beats.py
  -> energy curves on a 2 Hz grid: loudness, onsets, brightness        music/energy.py
  -> section boundaries: beat-synced MFCC+chroma, Foote novelty        music/structure.py
  -> emotion: MERT-v1-95M, 5 s windows centered on each grid time,     music/emotion.py
     13 hidden layers mean-pooled -> head -> (valence, arousal)
  -> intensity = mix of arousal + loudness + onsets                    music/profile.py
  => FullAnalysis, cached on Drive by the song's content hash
  -> choose excerpt -> MusicProfile (the contract the planner reads)   music/excerpt.py, profile.py
```

Fallbacks (golden rule 4): no MERT head -> heuristic arousal from loudness/onsets/brightness and valence 0; no Beat This! -> librosa beats. The app shows which fallbacks ran.

Two things to know:
- **MERT normalizes each window's loudness**, so a pure volume change does not move MERT arousal. The intensity curve keeps its loudness and onset terms for that reason.
- **Beat This! has been falling back to librosa since 2026-10-27** (its checkpoint host was down, and song analyses cached during the outage kept librosa beats). Keshav's PR #4 (`phase-13-beat-this-mirror`) fixes this; it touches the same cache check in `profile.py` as this branch (section 7).

## 3. Phase 14: what is built

All on `phase-14-temporal-emotion` (2 commits, rebased onto `main` on 2026-11-10 and pushed).

**Model** (`montage.music.emotion.make_temporal_head`, shared by training and inference):
```
per window:  softmax layer weights over MERT's 13 layers -> LayerNorm -> Linear(768, 128) -> GELU -> Dropout
across song: 1-layer bidirectional GRU (64 per direction)
per window:  [projection, GRU state] -> Dropout -> Linear -> tanh -> (valence, arousal)
```
- Bidirectional is fine because songs are analyzed offline; it can also learn the annotators' reaction lag.
- The skip connection lets it behave like the old per-window head where context does not help.
- About 120k trainable parameters on top of frozen MERT: cheap to train, hard to overfit badly.

**Loss** (summed over valence and arousal): `MSE + 0.5 x (1 - pooled CCC) + 0.5 x (1 - within-song CCC)`. Pooled CCC is dominated by differences between songs; the within-song term (each song's curves centered first) rewards following the shape.

**Training** (`training/emotion/temporal.py`): batches of 32 songs, one random 10-30 s crop per song (so the model cannot learn where clips start), AdamW, early stopping on validation, seeds 0/1/2. Same MERT shards and song split as 9B, so numbers are directly comparable.

**Baselines in the same job:** predict-mean, the 9B window head retrained on the same seeds, and the window head smoothed exactly as the app smooths it (`gauss_smooth`, 1 s). The smoothed window head is the bar to beat, because it is what the app actually uses.

**Main metric:** CCC per song, averaged over songs. Also pooled CCC and RMSE. Evaluated on the DEAM test clips and on 58 full songs (longer than any training crop: checks length generalization).

**Deploy gate:** writes `DATA_ROOT/weights/emotion_temporal.pt` only if the best temporal seed beats the best smoothed window head on validation. The 9B `emotion_head.pt` stays in place as the fallback.

**Inference fallback chain** (`predict_emotion_detail`): temporal head -> 9B window head -> heuristic. A missing or broken checkpoint falls through with a warning. `music.emotion.use_temporal: false` turns it off.

**Cache keying:** song analyses now store which model and weights produced them (`temporal:<hash>`, `window:<hash>`, or `heuristic`) and are recomputed when that changes. Before, deploying a retrained head silently kept old curves.

**No schema changes.** `MusicProfile.emotion_source` stays `mert` / `heuristic`. New config keys: `music.emotion.use_temporal`, `temporal_weights`, `temporal.{proj_dim, hidden, dropout}`.

**Tests** (`tests/test_temporal_emotion.py`, CPU, synthetic data, need `torch`): padding does not leak, loss behaves, crops stay inside a song, the temporal head learns a context-dependent synthetic task that the window head cannot, the fallback chain works, the model id tracks the weights, and the training job runs end to end on tiny shards. They skip on a laptop without `torch`; run them on Colab.

## 4. Plan

| Step | What | Why | Status |
|---|---|---|---|
| 14.0 | Push the branch; rebase on `main`; resolve the `PLAN.md` conflict | Colab only pulls from GitHub; nothing is backed up yet | done 2026-11-10 |
| 14.1 | Train on DEAM (Colab job), copy the table to `RESULTS.md` | The core result: does context improve within-song tracking? | done 2026-11-11: deployed; test arousal CCC per song 0.10 -> 0.18 (`RESULTS.md`) |
| 14.2 | Add PMEmo (794 pop choruses, dynamic V/A at 2 Hz) | More and more varied training data; report per dataset | after 14.1 |
| 14.3 | Annotator-lag study (shift targets 0-4 s, re-score the window head) | Shows how much of any gain is just learning the lag | after 14.1 |
| 14.4 | Rerun the phase 11 eval grid with the temporal head | Shows the effect on the edit itself (mood and pace coupling) | after 14.1 deploys |
| 14.5 | Architecture ablation: BiGRU vs small transformer vs dilated CNN | Is the GRU enough? | stretch |

EmoMusic is deliberately **not** used: it is the 2013 part of DEAM, so it would duplicate songs and could leak DEAM test songs into training.

## 5. Run book (Colab)

Prerequisites: the branch is pushed; you can see `My Drive/montage_data` (shared by Keshav, added as a shortcut); the 9B MERT shards exist at `montage_data/datasets/deam_mert/` (one `.npz` per song).

1. **Bootstrap** (run from the Colab UI, not remotely, so the Drive and Secrets prompts work):
   ```
   !git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null
   %run /content/bac_prototype/scripts/colab_bootstrap.py --branch phase-14-temporal-emotion --user andy
   ```
2. **Tests on the GPU runtime** (short):
   ```
   !cd /content/bac_prototype && python -m pytest -q tests/test_temporal_emotion.py
   ```
3. **Train** (long, so it runs as a background job; check `montage_data/jobs/*/` first that nobody else is running it):
   ```
   !cd /content/bac_prototype && python scripts/run_job.py --name emo_temporal -- python -m training.emotion.temporal
   ```
   Options: `--seeds 0 1 2`, `--epochs 60`, `--no-deploy` (report only), `--force-deploy`.
4. **Poll** every minute or so (keep the tab open and the laptop awake):
   ```
   !cd /content/bac_prototype && python scripts/job_status.py emo_temporal
   ```
5. **Results:** `montage_data/eval/emotion_temporal/report.md`, `metrics.json`, `curves_<song>.png`. Copy the tables and takeaways into `docs/RESULTS.md` and add a `docs/LOG.md` entry. If deployed, `montage_data/weights/emotion_temporal.pt` is picked up by everyone's app automatically (and their cached song analyses are recomputed once).

After pushing new code from the laptop, sync Colab with:
```
!cd /content/bac_prototype && git fetch -q && git checkout -q phase-14-temporal-emotion && git pull -q
```

## 6. What success looks like

- Test-set arousal CCC per song clearly above the smoothed window head's (today about 0.1), with pooled CCC not clearly worse.
- The full-song set does not collapse (length generalization).
- In the eval grid, mood coupling holds or improves.

Report whatever the numbers are; a clean negative result with the lag study is still a result.

## 7. Integration notes

- **Conflict with PR #4 (Beat This! mirror):** both branches change the "reuse the cached song analysis?" check in `music/profile.py: analyze_song`. This branch compares the emotion model id; PR #4 adds "redo it if it was cached with librosa beats and Beat This! now loads". The merge keeps both conditions.
- **Shared with other tracks:** song-pick sentiment matching (Track 1 of the 2-week plan) uses this module's valence/arousal, and the contrastive song model (Track 3) uses its MERT window features (`make_windows`, `mert_window_features`). Keep those function signatures stable.
- **Train/inference parity (golden rule 5):** training must use `montage.music.emotion` for windows, features and the model definition. Never use precomputed dataset features.

## 8. Glossary

- **Valence / arousal:** the two standard music-emotion axes: negative-positive and calm-energetic.
- **MERT-v1-95M:** a self-supervised music model (CC-BY-NC-4.0); we use its 13 hidden layers, frozen, at revision 12af15f.
- **DEAM:** 1,802 songs with valence/arousal annotated every 0.5 s (from 15 s onward); 45 s clips plus 58 full songs.
- **CCC (concordance correlation coefficient):** like Pearson correlation but also penalizes offset and scale; 1 is perfect, 0 is no agreement.
- **CCC per song:** CCC computed inside each song, then averaged; measures shape-following, ignoring the song's overall level.
