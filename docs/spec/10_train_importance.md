# Phase 10A (and optional 10B): Learned importance ranker

Goal: a small ranker that orders moments within a collection the way people do, trained on SumMe + TVSum (optionally CUFED photo albums), evaluated honestly against the heuristic, then blended in (spec 04).

**Core idea:** annotator scores only mean something relative to other moments in the same video/album, and inference ranks items within one person's uploads. So train with a **pairwise ranking loss on pairs from the same group** and evaluate with within-group rank correlation.

## 1. Data (`training/importance/download.py`, started in Phase 0)

Zenodo record 4884870 contains the original SumMe (25) and TVSum (50) videos plus the standard h5 files (`eccv16_dataset_summe_google_pool5.h5`, `eccv16_dataset_tvsum_google_pool5.h5`). Per video in the h5: `gtscore` (importance per sampled frame), `picks` (sampled frame indices), `n_frames`, `user_summary`, `change_points`, sometimes `video_name`. **Use the h5 only for labels and frame indices. Ignore its GoogLeNet features.**

## 2. Mapping h5 entries to raw videos (`training/importance/labels.py`)

- If `video_name` exists, map by name.
- Otherwise map by frame count: count frames of each raw video (`ffprobe -count_packets`) and match `n_frames` within one dataset; break ties by duration. Log the full mapping to `DATA_ROOT/datasets/importance/mapping.json` and assert all 75 map one to one. If ambiguity remains, stop and report.
- Frame label time = pick / native fps.
- Normalize labels per video to [0, 1] (min-max).

## 3. Features with the inference pipeline (`training/importance/extract_clip.py`)

Run `montage.ingest`, `montage.candidates`, and `montage.features` on the raw dataset videos **unchanged** (same 2 fps frames, same 3 s / 1.5 s windows, same CLIP, aesthetic, quality, faces). This guarantees train/inference parity. Cache as usual (feature cache is keyed by file hash, so the app never mixes them up).
Window label = mean normalized label of picks whose time falls inside the window. Drop windows with no picks.
Save `DATA_ROOT/datasets/importance/windows.npz`: `X` (ranker_features, spec 04), `y`, `group` (video id), `dataset` (summe/tvsum), `heuristic` (spec 04 H computed within each video, for the baseline).

## 4. Model and training (`training/importance/train.py`)

- Inputs: `ranker_features` with scalar columns standardized (save mean/std).
- Models: `linear` (D -> 1) and `mlp` (D -> 128 -> GELU -> Dropout 0.3 -> 1).
- Pairs: per group per epoch, sample up to 2,000 pairs (i, j) with |y_i - y_j| >= 0.05; target 1 if y_i > y_j.
- Loss: RankNet, `BCEWithLogits(s_i - s_j, target)`. AdamW lr 1e-3 (linear) / 3e-4 (mlp), weight decay 1e-2, 30 epochs, early stopping on mean validation Kendall tau.

## 5. Evaluation (`training/importance/evaluate.py`)

5-fold cross-validation **by video**, each fold mixing SumMe and TVSum. For each held-out video compute Kendall tau and Spearman rho between predicted scores and window labels; average per dataset.

Compare:
- random (expected ~0)
- aesthetic only (LAION score)
- heuristic H (spec 04)
- learned: {SumMe only, TVSum only, both} x {linear, mlp}
- blend of learned and heuristic at alpha in {0, 0.25, 0.5, 0.75, 1} (rank-normalized within video)

Select the configuration with the best mean tau on **SumMe** held-out videos (closest to personal footage), unless it is clearly worse on TVSum; log the choice. Choose `blend_alpha` by the same CV. Retrain the chosen model on all videos.

Outputs in `DATA_ROOT/eval/importance/`: `report.md`, `metrics.json`, and 2 plots of predicted score vs label over time for held-out videos.

## 6. Deploy

Save `DATA_ROOT/weights/importance_ranker.pt` (atomic write; shared): `{type, state_dict, feat_mean, feat_std, blend_alpha, metrics, feature_version, created, trained_by, git_commit}`. `importance/blend.py` uses the checkpoint's `blend_alpha` when present. App status shows `Importance: learned+heuristic`.

Then update `docs/RESULTS.md` (ranker table + one-line takeaway, including an honest note if it did not beat the heuristic), append a `docs/LOG.md` entry with a **Heads-up** that new weights are live, and update README "What works today".

## 7. Phase 10B (optional): CUFED photo albums

Only if 9A, 9B, and 10A are done and more than 4 hours remain before the demo.
- CUFED: 1,883 Flickr event albums (23 event types, 30-100 photos each) with per-photo importance from crowd votes. Find the download from the dataset authors' page (Wang et al., CVPR 2016; ML-CUFED, BMVC 2017). If no working link is found within 15 minutes, skip and log it.
- Each album is a group; each photo is an item; run the photo path of `montage.features`; add groups to training; evaluate per-album Spearman (published inter-annotator agreement is about 0.4, a useful ceiling); re-select and redeploy only if photo-album rank correlation improves without hurting SumMe.

## Acceptance tests

- Mapping covers all 75 videos one to one.
- `test_parity`: features for a dataset video computed by the training script equal those from `montage.features.extract` (same cache entry).
- CV table produced with every row above; the chosen model and alpha are logged with reasons. If learned does not beat the heuristic, ship `alpha = 0` and say so in the report (that is an honest result).
- App status line reflects the deployed ranker; `test_blend_with_ranker` checks `source == "blend"`.
