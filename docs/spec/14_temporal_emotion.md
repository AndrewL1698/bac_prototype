# Phase 14: Temporal music emotion model (Track 2, ROADMAP item 1)

Goal: emotion curves that follow changes **inside** a song, not just its overall mood. The 9B head scores each 5 s window on its own; it gets the song-level mood right (test CCC 0.74 valence, 0.82 arousal) but tracks movement within a song poorly (about 0.1 arousal CCC per song, RESULTS.md 9B). The planner's mood and pace terms read these curves shot by shot, so within-song tracking is what matters for the edit.

Depends on: 9B (MERT shards on Drive, `montage.music.emotion`). Everything heavy runs on Colab as a background job.

## 1. Data

**Now: DEAM only**, the same MERT shards and the same seeded song split as 9B (`training/emotion/dataset.py`), so test numbers are directly comparable with 9B. No new extraction is needed.

**Next (14.2): PMEmo.** 794 pop choruses with dynamic valence/arousal at 2 Hz (from the authors' GitHub / Zhejiang University page; Google Drive or Baidu links, so the download may need a manual step). Before mixing it with DEAM:
- Annotation scales differ (PMEmo is [0, 1], DEAM is [-1, 1]). Map PMEmo linearly to [-1, 1] and report results per dataset, never only pooled.
- Check whether the PMEmo release in use has dropped its first 15 s of annotations (the 2019 update does). Align times the same way as DEAM (annotation time = window center).
- Extract features with `training/emotion/extract_mert.py`'s code path (golden rule 5), into `datasets/pmemo_mert/`.
- Split by song, seeded, like DEAM. Train on DEAM-train + PMEmo-train, and report on each test set separately.

**Not EmoMusic.** EmoMusic ("1000 songs", MediaEval 2013, 744 after deduplication) is already the 2013 part of DEAM. Adding it would duplicate songs and could put DEAM test songs into training. (ROADMAP item 1 lists it; this replaces that.)

## 2. Model (`montage.music.emotion.make_temporal_head`, shared by training and inference)

```
per window:   w = softmax(layer_logits)                       # 13 learnable scalars, as in 9B
              x_t = Dropout(GELU(Linear(LayerNorm(sum_l w_l * feats_t[l]), proj_dim)))
across song:  h_t = BiGRU(x_1..x_T)                           # 1 layer, `hidden` per direction
per window:   (valence_t, arousal_t) = tanh(Linear(Dropout([x_t, h_t])))
```

- Bidirectional is fine: songs are analyzed offline, and it lets the model learn the annotators' reaction lag.
- The skip `[x_t, h_t]` lets it fall back to per-window behaviour where context does not help.
- Sizes come from `music.emotion.temporal` in `configs/default.yaml` (proj_dim 128, hidden 64, dropout 0.2), and are saved in the checkpoint (`arch`), so a config change never breaks loading an old checkpoint.
- Padded batches use `pack_padded_sequence`, so padding never leaks into the backward direction (tested).

## 3. Loss and training (`training/emotion/temporal.py`)

Loss, summed over valence and arousal:

```
MSE + 0.5 * (1 - pooled CCC) + 0.5 * (1 - within-song CCC)
within-song CCC = 2 cov(p_c, y_c) / (var(p_c) + var(y_c) + eps), with p_c, y_c centered per song
```

Pooled CCC is dominated by differences between songs; the within-song term rewards following the curve's shape.

- Batches of 32 songs, one random crop per song (20-60 steps = 10-30 s, random offset). The random crops keep the model from learning where a clip starts (every DEAM clip's annotations start at 15 s).
- AdamW lr 1e-3, weight decay 1e-2, gradient clip 1.0, up to 60 epochs, early stopping (patience 8).
- Model selection score: mean over valence and arousal of 0.5 * pooled CCC + 0.5 * per-song CCC, on validation.
- Seeds 0, 1, 2. Results are reported as mean +- std over seeds.

## 4. Baselines and evaluation

On test (45 s clips) and full (58 full songs; longer than any training crop, so this checks length generalization). All baselines are trained in the same job on the same split and seeds:

| Model | Why |
|---|---|
| predict-mean | floor (per-song CCC is 0 by construction) |
| window head (9B architecture, retrained) | what the app uses today |
| window head, smoothed | what the app actually shows: `profile.py` smooths with `gauss_smooth(x, 1.0 s)` |
| temporal head, and smoothed | the new model, raw and as the app would run it |

Metrics: per dimension, **CCC per song averaged over songs (main metric)**, pooled CCC, and RMSE.

Outputs in `DATA_ROOT/eval/emotion_temporal/`: `report.md`, `metrics.json`, and `curves_<song>.png` for 3 test and 2 full songs (annotated vs smoothed window head vs temporal head).

## 5. Deploy

`DATA_ROOT/weights/emotion_temporal.pt` (atomic): `{state_dict, arch, arch_type: "temporal", layer_logits, outputs, window_s, align, metrics, feature_version, created, trained_by, git_commit, seed}`.

It is written only if the best temporal seed's smoothed validation score beats the best smoothed window head's (`--force-deploy` overrides, `--no-deploy` skips). The 9B `emotion_head.pt` is left in place as the fallback.

## 6. Inference and fallbacks (`montage.music.emotion`)

`predict_emotion_detail` tries, in order: the temporal head over the whole song's window sequence -> the 9B window head -> the heuristic. A missing or broken checkpoint falls through with a warning, never an error. `music.emotion.use_temporal: false` turns the temporal head off.

`MusicProfile.emotion_source` stays `"mert" | "heuristic"` (no schema change). The new `emotion_model_id()` (`temporal:<weights hash>`, `window:<weights hash>` or `heuristic`) is stored in the cached song analysis (`FullAnalysis.emotion_model`, internal to `profile.py`). A cached analysis is reused only if it was made by the same model and weights, so deploying or retraining a head invalidates old analyses. (This also fixes 9B's cache, which only compared `mert` vs `heuristic`.)

## Acceptance tests (`tests/test_temporal_emotion.py`; all CPU, no data)

- Padding: a padded sequence gives the same outputs as the same sequence alone.
- Loss: zero for a perfect prediction; a flat curve at the right level scores clearly worse than a slightly offset curve with the right shape.
- Crops stay inside one song, are contiguous, and pad with -1.
- On a synthetic task where the target depends on earlier windows only, the temporal head learns it (per-song CCC > 0.35) and the window head cannot (< 0.15).
- Inference prefers temporal, then window, then heuristic; a corrupt temporal checkpoint falls back to the window head; `use_temporal: false` is respected.
- The model id changes when the weights change.
- The training job runs end to end on tiny synthetic shards: report, plots, deploy, and inference picks up the deployed head.

On Colab (after the run): test-set arousal CCC per song is clearly above the smoothed window head's, and pooled CCC is not clearly worse. Report whatever the numbers are, honestly.

## Open questions

- Annotator lag: DEAM annotators react a few seconds late. The BiGRU can learn a lag, but an explicit lag study (shift targets by 0-4 s, re-score the window head) would show how much of the gain is just lag.
- Is the BiGRU enough, or does a small transformer (or a dilated temporal CNN) do better? Ablate after PMEmo is in.
- Planner impact: rerun the phase 11 eval grid with the temporal head (mood coupling, pace coupling) and record it in RESULTS.md.
