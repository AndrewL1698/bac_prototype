# Phase 15: Photo-album importance (CUFED / ML-CUFED)

Track 2, ROADMAP item 2 ("Photo-album importance"). Supersedes the optional 10B section of `10_train_importance.md`; everything else in spec 10 (RankNet, pairs within a group, CV by group, deploy format) still applies.

Goal: the learned importance ranker (phase 10A) is trained on SumMe/TVSum **videos**, but the albums people upload are about 90% **photos**, and RESULTS.md has no photo-album number at all. Train and evaluate the same ranker on event photo albums with human importance labels and deploy it as a **separate photo ranker** only if photo-album ranking improves. The video ranker stays as it is (see section 5 for why).

Schedule (ROADMAP): days 1-3 sections 1-4 (download, label mapping, features); days 4-7 section 5 (training, evaluation, deploy decision); days 8-10 section 6 (personalization, only if time). Cut order: CUFED is the last ML item cut; if it fails, the 10A ranker stays deployed (that is the fallback).

## 1. Dataset

- **CUFED** (Wang et al., CVPR 2016, "Event-specific Image Importance"): 1,883 Flickr event albums, 23 event types, 30-100 photos per album. Each photo was rated by 5 people for its importance *within its album*; the score is the mean rating, discrete, in [-2, 2]. A blurred photo of a key moment can score high: it is not a quality score.
- **ML-CUFED** (Wang et al., BMVC 2017): the same albums and photos with **multi-label** event types (an album can have more than one). For importance the two are the same data; `event_type.json` holds a list per album, so ML-CUFED labels come for free.
- **File layout** (verified 2026-11-10 against two independent public loaders, `HVQuan02/peta_updated` and `blagoySimandov/narrative-image-recommender`):
  - `image_importance.json`: `{album_id: [[flickr_url, "album_id/photo_id", score], ...]}`
  - `event_type.json`: `{album_id: ["Wedding", ...]}`
  - `images/<album_id>/<photo_id>.jpg`
  - Album ids look like `0_64041142@N00` (index, underscore, Flickr owner id).
- **Not this:** the "CUFED" and "CUFED5" sets on Google Drive used by reference-based super-resolution (TTSR, SRNTT, C2-Matching, MASA-SR) are cropped image patches **without importance labels**. The layout check rejects any folder without `image_importance.json`, so this mix-up fails loudly.
- **Ceiling:** published inter-annotator agreement is about **0.4 Spearman**. A model near 0.4 is as consistent as people are with each other.
- **Terms:** CUFED's photos come from the Yahoo Flickr Creative Commons 100M dataset (YFCC100M); the authors' download required accepting its terms: no redistribution, no attempts to identify people, attribution in published results ("Yahoo Webscope", http://webscope.sandbox.yahoo.com, and Thomee et al., arXiv:1503.01817), and each photo's own Creative Commons license applies before showing it anywhere. We follow them whichever copy we use.
- **License:** research, non-commercial (the Hugging Face mirror states CC BY-NC 2.0). Like SumMe and MERT, it would need replacing before any commercial use. Datasets live only on the shared Drive, never in the repo.

## 2. Download (`training/importance/cufed.py`)

```
python scripts/run_job.py --name cufed_dl -- python -m training.importance.cufed
python -m training.importance.cufed --inspect      # report what is on disk; downloads nothing
```

Everything lives in `DATA_ROOT/datasets/importance/cufed/`. The **dataset root** is the folder that contains `image_importance.json`, found anywhere under `cufed/raw/` (archives and mirrors nest differently).

Sources, tried in the order of `importance.cufed.sources` on every run; the first one that has (or downloads) a valid root wins. Each remote source keeps its own folder, so an earlier mirror download never shadows a later official link or manual copy, and a source already on disk is reused without any network call:
1. **`manual`**: a valid root anywhere under `cufed/raw/` except `official/` and `hf/` (someone downloaded it in a browser and copied it there).
2. **`official`**: `cufed/raw/official/` if it already holds a root; otherwise `importance.cufed.official_url`, a direct link to the authors' archive from the dataset page (https://acsweb.ucsd.edu/~yuw176/event-curation.html, "Download CUFED"). **Dead as of 2026-11-10:** the authors' site times out (from the cloud workspace and from a normal browser); its last Wayback Machine capture (2024-04-20) shows the "Data" link going through a Yahoo YFCC100M terms page to a Google Drive file (`0BxBaqbArti0NMlF4aG05c3h2NU0`) that now answers "file does not exist", and the Wayback Machine has no copy of the file. Kept so a re-shared link from the authors can be plugged in. Resumable download to scratch (`training.common.download`), then zip/tar extraction into `cufed/raw/official/`. If the response is an HTML page (e.g. a Google Drive "can't scan for viruses" page), the job stops and says to download by hand into `cufed/raw/` (the `manual` source). A failed download (dead link, no network) is logged and the next source is tried.
3. **`hf`**: `cufed/raw/hf/` if it already holds a root; otherwise the Hugging Face mirror `importance.cufed.hf_repo` (`Shawn-Huang/CUFED-AlbumBench`) at `hf_revision`, via `huggingface_hub.snapshot_download` (resumable; pinned to the resolved commit sha; uses `HF_TOKEN` if the repo asks for a login) into `cufed/raw/hf/`, then any zip/tar inside is extracted in place. **As of 2026-11-10 this mirror holds images for the 133-album AlbumBench test split only** (5,665 photos; open issue byu-vision/albumbench#2 asks for the rest), so it is a fallback, not the full dataset. With the official download gone, it is the **main source**. Its `image_importance.json` labels exactly the 133 albums it has photos for, and every labeled photo is on disk. 133 albums is still more groups than SumMe + TVSum (74 videos) and enough for 5-fold CV.

Finding the root never walks `images/` folders (they hold the photos; on the Drive that would be slow). With several roots in one source, the one with the most album folders wins.

Archive extraction refuses members that would land outside the target folder (absolute paths on any OS: a leading `/` or `\` or a drive letter like `C:`; and `..`), skips links and `__MACOSX`/`._*` entries, and writes a `_extracted_<archive>.json` marker so reruns skip it.

Provenance: each remote source writes `_source.json` in its folder (url, or repo + revision + resolved sha, and the download time) after extraction finishes; a remote folder without it (a download that died half-extracted) does not count as on disk and is fetched again. `cufed/raw/_downloaded.json` records which root the last mapping used: source, provenance, root path, and counts (labeled albums, labeled photos, album folders, photo files).

**No Flickr fill.** An earlier version could fetch labeled photos missing on disk from their original Flickr URLs. It was removed in review (PR #6): the mirror's labels cover only albums whose photos are all present, and the official archive is gone, so it could never add a photo.

## 3. Label mapping (`cufed.py: build_mapping`)

- One **group** per album, one **item** per labeled photo. A photo's file is `images/<album>/<photo_id>.<ext>` with `<ext>` any extension ingest accepts (`montage.ingest.assets.PHOTO_EXT`).
- An album is **kept** if at least `importance.cufed.min_photos` (10, the app's `ingest.min_assets`) labeled photos are on disk and their scores are not all equal. Otherwise it is **dropped** with a reason: `no_images`, `few_photos`, or `tied_scores` (no ranking signal).
- Label: `y` = min-max of the raw score over the album's photos on disk (missing photos are left out before normalizing), so `y` is in [0, 1] within every kept album, as SumMe/TVSum labels are within a video.
- Albums without an `event_type.json` entry get `["unknown"]`.
- Checks (the job stops if any fails): every labeled album appears exactly once (kept or dropped); every kept photo maps to exactly one file and no file is used twice; every score is finite. Scores outside `importance.cufed.score_range` are counted and logged (not fatal).
- Writes `cufed/mapping.json`: `{"albums": {album: {events, photos: [{photo, path, score, y}], n_labeled, n_missing}}, "dropped": {album: {reason, n_labeled, n_present}}, "summary": {...}}`. Rebuilt on every run (cheap, and it must reflect what is on disk).

## 4. Features with the inference pipeline (`training/importance/cufed_features.py`)

```
python scripts/run_job.py --name cufed_feats -- python -m training.importance.cufed_features
python -m training.importance.cufed_features --limit 5     # smoke run on 5 albums first
```

- **One album = one upload.** Each album goes through `montage.candidates.build_media` + `montage.features.extract` **unchanged** (same analysis frame, CLIP ViT-L/14, aesthetic, quality, faces), with only the ingest count limits lifted (`ingest.min_assets: 1`, `ingest.max_assets: 10000`), as `extract_clip.py` does for videos. This is train/inference parity (golden rule 5).
- Albums are **never batched** into one pipeline call: ingest drops exact duplicate files within a call (a photo shared by two albums would vanish from the second), and the heuristic baseline is a z-score within the upload. One call per album reproduces both exactly as the app sees them.
- Candidates map back to photos by the asset's source path. Photos that ingest skips (exact duplicate files inside one album) are dropped and counted.
- Per album: `X = ranker_features` (768 CLIP + 8 scalars, spec 04/10, identical to inference), `y` re-normalized min-max over the photos that have features, `heuristic` = spec 04 H within the album (the baseline), `aesthetic`, `photo`, `event` (first event type). An album left with fewer than `min_photos` rows or tied `y` is skipped (`.skip.json` with the reason).
- Output per album: `cufed/per_album/<album>.npz` (or `.skip.json`); the job is resumable (finished albums are skipped). An album that raises (e.g. an unreadable photo) writes `.skip.json` with the error and the job continues; `importance.cufed.max_consecutive_failures` failures in a row stop the job (something systemic, like a missing model, not one bad album).
- Combined: `DATA_ROOT/datasets/importance/cufed_photos.npz` with keys `X, y, heuristic, aesthetic, group ("cufed/<album>"), dataset ("cufed"), event, photo` and a summary `cufed_photos.json`. The core keys match `windows.npz`, so training/evaluation can concatenate the two.
- The feature cache is keyed by file hash (as for uploads), so a rerun or a later app run on the same photos never recomputes.

## 5. Training, evaluation, deploy decision (days 4-7)

```
python scripts/run_job.py --name cufed_eval -- python -m training.importance.evaluate_cufed
python -m training.importance.evaluate_cufed --deploy      # only after reading the report
```

- **New module `training/importance/evaluate_cufed.py`**; `evaluate.py` (10A, videos only) is left unchanged so its published numbers stay reproducible. It reuses `evaluate.rank01_within`, `evaluate.ALPHAS` and the RankNet trainer `train.fit` as they are. Settings in `importance.cufed.eval`.
- Data: `windows.npz` + `cufed_photos.npz` concatenated on the shared keys (same `ranker_features` columns).
- CV: 5 folds **by group** (video or album), each dataset dealt on its own. SumMe and TVSum are drawn first with the same generator as `evaluate.folds`, so the video folds are exactly 10A's. Inner validation for early stopping as in 10A (10% of the training groups). Per held-out group: Kendall tau-b and Spearman rho (tie-aware; CUFED scores are discrete), averaged per dataset (`summe`, `tvsum`, `cufed`). Out-of-fold scores are cached per configuration (keyed by a fingerprint of the data and settings), so a rerun after a Colab disconnect skips finished configurations.
- **Photos get their own ranker.** The app already ranks photos only against photos and videos only against videos (`importance/blend.py`), so photos can use their own checkpoint: `weights/importance_ranker_photo.pt` (`importance.photo_ranker_weights`) scores photo candidates when it exists, with its own blend alpha; videos always use `importance_ranker.pt`; with no photo ranker, photos use the video ranker as before. Why (first run, 2026-11-15): one ranker for both forced a trade-off. Models trained on CUFED order photos about 4x better than the live ranker (CUFED tau 0.30 vs 0.08) but drop SumMe to about 0.10, so a SumMe guard ruled them all out, and the only model left was the live one with less weight on the learned part, which barely helped photos and dropped TVSum from 0.41 to 0.08 (a guard on SumMe alone missed that). With separate rankers there is no trade-off: this phase only ever changes the photo ranker.
- Rows: random, aesthetic only, heuristic H, **what scores photos today, as is** (`deployed_as_is`: the photo ranker if live, else the video ranker), **its configuration re-run in this CV** (`deployed_config_cv`), learned {summe, tvsum, both (= SumMe + TVSum), cufed, all} x {linear, mlp}, and blends of the selected model with H at alpha in {0, 0.25, 0.5, 0.75, 1} (rank-normalized within the group, as `importance/blend.py` does at inference).
- Selection: every (learned model, alpha) blend is a candidate (alpha 0 is the heuristic alone). Chosen: the best mean tau on **held-out CUFED albums**; ties prefer the smaller alpha. No video guard: the chosen model only scores photos, and the SumMe/TVSum columns are reported for information. The chosen model is retrained on all of its training data.
- Baseline ("what the app does today on a photo album"): what scores photos today, as is, when it never saw CUFED (today: the video ranker, trained on SumMe + TVSum, so CUFED is out of sample for it); once a CUFED-trained ranker is live, its as-is CUFED score is in sample, so its configuration re-run in this CV is the baseline instead; no live checkpoint: the heuristic.
- **Deploy rule:** the per-album tau gain (chosen minus baseline) must be above 0, and with `require_ci: true` its 95% bootstrap CI over albums (`bootstrap` resamples) must be above 0 too. With 133 albums a small average gain can be noise, and the live weights affect every teammate's app. Otherwise photos keep what they use today and the report says so.
- **A run never touches the live weights.** It writes `DATA_ROOT/eval/importance_cufed/`: `report.md` (table, decision, per-event breakdown), `metrics.json`, `per_album_tau.png` (chosen vs baseline per album) and `candidate.pt` (spec 10 checkpoint format plus `trained_on`, `modality: photo` and `phase: 15`). `--deploy` copies the candidate to `weights/importance_ranker_photo.pt` only if the decision says deploy and the candidate matches `metrics.json` (sha1), keeping the previous file as `importance_ranker_photo.prev.pt`. It never writes `importance_ranker.pt`.
- After a deploy: a CUFED column in the RESULTS.md ranker table, a one-line takeaway, a LOG **Heads-up** that a photo ranker is live, README "What works today". After a no-deploy: the same table and an honest takeaway.

## 6. Personalization (days 8-10, only if time)

Learn from each user's pins and excludes (`Session.pins` = shots marked Keep, `Session.excluded` = shots marked Remove) to nudge importance for that user, across albums. Today they only act inside one session (kept shots are forced in, removed shots are dropped); nothing carries over.

**Module `montage/importance/personalize.py`.** Settings in `importance.personalize`.

- **Evidence.** When a recut applies pins or excludes, `record(session_id, features, pins, excluded)` stores, for that session, the sum and count of the CLIP embeddings (`features.arrays["emb"]`, L2-normalized) of the kept shots (`pos`) and of the removed shots (`neg`). Re-recording a session replaces its evidence, so "Stop keeping" / "Put back" take the shot out again. File: `DATA_ROOT/prefs/<user>.json` (`paths.user()`), written atomically; the newest `max_sessions` (50) sessions are kept. Only these per-session sums are stored: no photos, ids or file names.
- **Preference score** for a candidate with embedding e: `cos(e, mean_pos) - cos(e, mean_neg)` over all stored sessions (each term only when that side has evidence).
- **Nudge** (`apply(final, features)`), within each modality (photos, videos), like the blend: `final' = rank01((1 - beta) * final + beta * rank01(pref))`, with `beta = beta_max * n / (n + half_evidence)`, n = number of kept + removed shots stored. Nothing changes below `min_events` (3) or without a prefs file. beta grows with evidence and never exceeds `beta_max`, so a few clicks cannot override the ranker.
- **Where.** `pipeline.recut` records before planning; `pipeline.plan_edit` applies the nudge to `importance.final` (not to `heuristic`) when `personalize=True` (the default) and adds an EDL note "personalized from N kept/removed shots". Evaluation callers (`eval/run_grid.py`, `eval/render_check.py`) pass `personalize=False`, so their numbers do not depend on who runs them. `python -m montage.importance.personalize --show` prints the evidence counts; `--reset` deletes the user's file.
- **Privacy.** The file holds averaged CLIP vectors of the user's kept and removed shots, in that user's namespace on the shared Drive; it cannot be turned back into photos. Nothing leaves the machine.

**Offline check (`training/importance/evaluate_personalize.py`, CUFED).** A simulated user edits albums of one event type: in each history album they keep the `k` (3) photos with the highest CUFED score and remove the `k` lowest. For each held-out album, the history is every other album of the same event type (leave one album out). Base scores: the chosen photo ranker's out-of-fold predictions from the phase 15 CV (`eval/importance_cufed/oof/cufed_linear.npz`), so the held-out album was never trained on. Report: CUFED tau base vs personalized per held-out album, mean gain with a 95% bootstrap CI over albums, for (a) a history of the same event type and (b) a control history from other event types (a random user's taste), at `beta_max` in {0.1, 0.2, 0.3, 0.5}. Default `beta_max`: the largest value whose control gain is not below -0.01 and whose same-event gain CI is above 0; if none, personalization ships disabled (`enabled: false`). Output `DATA_ROOT/eval/personalize/report.md` + `metrics.json`.

**Tests (`tests/test_personalize.py`).** Record and re-record a session (replace, put back, session cap); no file or fewer than `min_events` leaves scores unchanged; with evidence, shots like the kept ones move up and shots like the removed ones move down, within each modality; beta follows the formula; `recut` records and `plan_edit` applies with a note, and `personalize=False` gives exactly the plain scores; the offline check on synthetic albums finds a gain for same-type history and none for the control.

## 7. Acceptance tests (`tests/test_cufed.py`, `tests/test_cufed_eval.py`)

- Layout: a valid root is found under nested folders; a folder without `image_importance.json` (the super-resolution CUFED) is rejected with a clear message; archive extraction refuses paths outside the target.
- Sources: an official HTML answer stops with instructions; the mirror is pinned to its resolved sha; a gated mirror falls through; a source on disk is reused without a download; a later official link or manual copy wins over an earlier mirror download; a dead official link falls through to the mirror.
- Mapping: every labeled album appears exactly once (kept or dropped with the right reason), photos map one to one, `y` is min-max in [0, 1] over present photos only.
- Reruns: with the mirror already in `raw/hf/`, the manual check does not report it as unknown files (no false super-resolution warning) and nothing is downloaded again.
- Feature assembly: per-album rows, `y` re-normalized after ingest drops a duplicate, resume skips finished albums, consecutive failures stop the job (pipeline mocked).
- Evaluation (`tests/test_cufed_eval.py`): video folds identical to 10A and unchanged by adding albums; selection on CUFED alone with the tie rule; the bootstrap gain and CI; end to end on synthetic data where importance is learnable (it learns, decides, writes the candidate but not the live weights; `--deploy` makes it the live photo ranker and never writes the video ranker; a rerun reuses the CV cache and compares against the right baseline; a non-deploy decision or a mismatched candidate is refused); a video-trained live ranker is today's photo baseline, compared as is. Blend (`tests/test_training.py`): with a photo ranker, photos use it and video scores are exactly unchanged; without one, scores equal the old single-ranker formula.
- `test_cufed_parity` (`gpu`, `media`): features for an album from `cufed_features` equal `build_media` + `extract` on the same photos (same cache entries).
