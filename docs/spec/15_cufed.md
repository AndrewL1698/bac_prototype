# Phase 15: Photo-album importance (CUFED / ML-CUFED)

Track 2, ROADMAP item 2 ("Photo-album importance"). Supersedes the optional 10B section of `10_train_importance.md`; everything else in spec 10 (RankNet, pairs within a group, CV by group, deploy format) still applies.

Goal: the learned importance ranker (phase 10A) is trained on SumMe/TVSum **videos**, but the albums people upload are about 90% **photos**, and RESULTS.md has no photo-album number at all. Train and evaluate the same ranker on event photo albums with human importance labels, keep the same model and blend, and deploy a new checkpoint only if photo-album ranking improves without hurting SumMe.

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
- **License:** research, non-commercial (the Hugging Face mirror states CC BY-NC 2.0). Like SumMe and MERT, it would need replacing before any commercial use. Datasets live only on the shared Drive, never in the repo.

## 2. Download (`training/importance/cufed.py`)

```
python scripts/run_job.py --name cufed_dl -- python -m training.importance.cufed
python -m training.importance.cufed --inspect      # report what is on disk; downloads nothing
```

Everything lives in `DATA_ROOT/datasets/importance/cufed/`. The **dataset root** is the folder that contains `image_importance.json`, found anywhere under `cufed/raw/` (archives and mirrors nest differently).

Sources, tried in the order of `importance.cufed.sources` on every run; the first one that has (or downloads) a valid root wins. Each remote source keeps its own folder, so an earlier mirror download never shadows a later official link or manual copy, and a source already on disk is reused without any network call:
1. **`manual`**: a valid root anywhere under `cufed/raw/` except `official/` and `hf/` (someone downloaded it in a browser and copied it there).
2. **`official`**: `cufed/raw/official/` if it already holds a root; otherwise `importance.cufed.official_url`, a direct link to the authors' archive from the dataset page (https://acsweb.ucsd.edu/~yuw176/event-curation.html, "Download CUFED"). `null` until a person copies the link (the page cannot be reached from the cloud workspace). Resumable download to scratch (`training.common.download`), then zip/tar extraction into `cufed/raw/official/`. If the response is an HTML page (e.g. a Google Drive "can't scan for viruses" page), the job stops and says to download by hand into `cufed/raw/` (the `manual` source). A failed download (dead link, no network) is logged and the next source is tried.
3. **`hf`**: `cufed/raw/hf/` if it already holds a root; otherwise the Hugging Face mirror `importance.cufed.hf_repo` (`Shawn-Huang/CUFED-AlbumBench`) at `hf_revision`, via `huggingface_hub.snapshot_download` (resumable; pinned to the resolved commit sha; uses `HF_TOKEN` if the repo asks for a login) into `cufed/raw/hf/`, then any zip/tar inside is extracted in place. **As of 2026-11-10 this mirror holds images for the 133-album AlbumBench test split only** (5,665 photos; open issue byu-vision/albumbench#2 asks for the rest), so it is a fallback, not the full dataset. 133 albums is still more groups than SumMe + TVSum (74 videos) and enough for 5-fold CV.

Finding the root never walks `images/` folders (they hold the photos; on the Drive that would be slow). With several roots in one source, the one with the most album folders wins.

Archive extraction refuses members that would land outside the target folder (absolute paths, `..`), skips links and `__MACOSX`/`._*` entries, and writes a `_extracted_<archive>.json` marker so reruns skip it.

Provenance: each remote source writes `_source.json` in its folder (url, or repo + revision + resolved sha, and the download time). `cufed/raw/_downloaded.json` records which root the last mapping used: source, provenance, root path, and counts (labeled albums, labeled photos, album folders, photo files).

**Optional Flickr fill** (`importance.cufed.flickr.enabled: false` by default; turn on with `--set importance.cufed.flickr.enabled=true`): `image_importance.json` stores each photo's original Flickr URL, and it may cover more albums than the images on disk (e.g. the mirror). For labeled photos missing on disk, fetch the URL (`flickr.workers` threads, `flickr.delay_s` between requests per worker, `flickr.timeout_s`), check that the bytes decode as an image, and write `images/<album>/<photo>.jpg` atomically. Dead links are recorded in `cufed/raw/_flickr_failed.json` and skipped on later runs (`--retry-failed` tries them again). Only image bytes are downloaded; nothing about the user leaves the machine.

## 3. Label mapping (`cufed.py: build_mapping`)

- One **group** per album, one **item** per labeled photo. A photo's file is `images/<album>/<photo_id>.<ext>` with `<ext>` any extension ingest accepts (`montage.ingest.assets.PHOTO_EXT`).
- An album is **kept** if at least `importance.cufed.min_photos` (10, the app's `ingest.min_assets`) labeled photos are on disk and their scores are not all equal. Otherwise it is **dropped** with a reason: `no_images`, `few_photos`, or `tied_scores` (no ranking signal).
- Label: `y` = min-max of the raw score over the album's photos on disk (missing photos are left out before normalizing), so `y` is in [0, 1] within every kept album, as SumMe/TVSum labels are within a video.
- Albums without an `event_type.json` entry get `["unknown"]`.
- Checks (the job stops if any fails): every labeled album appears exactly once (kept or dropped); every kept photo maps to exactly one file and no file is used twice; every score is finite. Scores outside `importance.cufed.score_range` are counted and logged (not fatal).
- Writes `cufed/mapping.json`: `{"albums": {album: {events, photos: [{photo, path, score, y}], n_labeled, n_missing}}, "dropped": {album: {reason, n_labeled, n_present}}, "summary": {...}}`. Rebuilt on every run (cheap, and it must reflect photos added by a Flickr fill).

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

- `evaluate.py` is generalized from the hardcoded `summe`/`tvsum` to any dataset tag in the data: 5-fold CV **by group** (album or video), Kendall tau-b and Spearman rho (tie-aware; CUFED scores are discrete) within each held-out group, averaged per dataset (`summe`, `tvsum`, `cufed`).
- Rows: random, aesthetic only, heuristic H, **the deployed 10A ranker as is**, learned {summe, tvsum, cufed, videos (summe+tvsum), all} x {linear, mlp}, and blends of the selected learned model with H at alpha in {0, 0.25, 0.5, 0.75, 1}.
- Selection: the configuration with the best mean tau on **held-out CUFED albums**, among those whose SumMe tau is not clearly worse than the deployed ranker's (same `CLEARLY_WORSE` margin as spec 10); alpha by the same CV. Deploy only if it beats the deployed ranker on CUFED; otherwise keep the deployed checkpoint and report that honestly.
- Deploy as spec 10 section 6 (`weights/importance_ranker.pt`, atomic, shared), plus a CUFED column in the RESULTS.md ranker table, a one-line takeaway, and a LOG **Heads-up** that new weights are live.

## 6. Personalization (days 8-10, only if time)

Learn from each user's pins and excludes (`Session.pins`, `Session.excluded`) to nudge importance for that user. Specified separately if reached; first thing dropped if time runs short.

## 7. Acceptance tests (`tests/test_cufed.py`)

- Layout: a valid root is found under nested folders; a folder without `image_importance.json` (the super-resolution CUFED) is rejected with a clear message; archive extraction refuses paths outside the target.
- Sources: an official HTML answer stops with instructions; the mirror is pinned to its resolved sha; a gated mirror falls through; a source on disk is reused without a download; a later official link or manual copy wins over an earlier mirror download; a dead official link falls through to the mirror.
- Mapping: every labeled album appears exactly once (kept or dropped with the right reason), photos map one to one, `y` is min-max in [0, 1] over present photos only.
- Flickr fill: only missing photos are fetched, non-images are rejected, failures are recorded and skipped on the next run (network mocked).
- Feature assembly: per-album rows, `y` re-normalized after ingest drops a duplicate, resume skips finished albums, consecutive failures stop the job (pipeline mocked).
- `test_cufed_parity` (`gpu`, `media`): features for an album from `cufed_features` equal `build_media` + `extract` on the same photos (same cache entries).
