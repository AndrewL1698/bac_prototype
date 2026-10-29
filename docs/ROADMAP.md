# ROADMAP (proposal by Andrew): 8-week extension

> **Status: DRAFT proposal by Andrew, 2026-10-29. Not agreed yet.** Nothing here is claimed or scheduled.
> Keshav and the two new devs: please review and comment (open questions at the bottom). Once agreed, the phases
> move into `docs/PLAN.md` with owners, and each gets a spec in `docs/spec/`. Until then `docs/PLAN.md` stays the plan of record.

## Context
The MVP (phases 0-11, built in ~2 days by Keshav and Andrew) works end to end: ingest -> CLIP features -> importance (heuristic + SumMe/TVSum ranker) -> music (beats, MERT emotion head on DEAM) -> Claude director -> beam-search planner -> FFmpeg render -> Gradio app. The project is for an **ML cohort**, and four developers (Andrew, Keshav, plus two more) now have **8 weeks**. The extension needs much more ML depth: new trained models, real datasets, baselines, ablations, and human evaluation, built in the same style as 9B/10A (our own feature pipeline, a non-learned fallback, numbers in RESULTS.md).

Headline new feature: **"Pick the song for me"**. The user uploads photos plus their Spotify listening history. The app picks a song they actually listened to around when the photos were taken that matches the photos' vibe, then makes the montage. Spotify cannot supply audio. Its SDKs only play inside Spotify's DRM player, preview clips and audio features were removed for new apps in Nov 2024, and its developer terms forbid syncing its content with video or training ML on it. So the app **shortlists** songs from the history and the **user supplies the audio file**, either by upload or by pointing the app at a local music folder that it matches automatically. A future mobile app could use Spotify login (top tracks plus the last 50 plays) instead of the history export, and in-app playback to preview the shortlist.

Final feature (weeks 7-8, mostly non-ML): a simple timeline editor on top of the EDL.

## Part 1: what to improve in the current pipeline (from inspection)

| # | Weakness (evidence) | Improvement | ML? |
|---|---|---|---|
| 1 | Music emotion tracks each song's overall mood but barely follows changes inside a song: arousal CCC about 0.1 per song (RESULTS.md 9B). The head scores each 5 s window on its own. | Temporal emotion model: a small GRU/transformer over the sequence of MERT window features, with a loss on per-song-centered targets plus CCC. Add PMEmo and EmoMusic data next to DEAM. Report per-song CCC as the main metric. | Yes |
| 2 | The importance ranker is trained on video summarization data (SumMe/TVSum), but eval albums are about 90% photos. CUFED (10B) was skipped. | Photo-album importance: train on CUFED / ML-CUFED (event albums with human importance labels), keeping the same RankNet and blend. Then personalize from pins, excludes, recut notes and Rate-tab votes. | Yes |
| 3 | The planner's ~25 cost weights are hand-tuned, and the only human signal is the live vote. Proxy metrics trade off (w_dir sweep). | Learn editing preferences: log pairwise preferences (A/B votes, recut-before vs recut-after). Fit a Bradley-Terry reward model over EDL-level features (the per-shot `ShotScores` aggregates plus eval metrics), then use it to tune planner weights or re-rank beam outputs. | Yes |
| 4 | The full cut is less varied than random picks (pairwise CLIP similarity 0.61 vs 0.59). Pairwise terms only score neighbouring shots. | Global coverage term: event segmentation (time + CLIP clustering + EXIF GPS), with a reward for covering events/people and a DPP- or MMR-style diversity bonus in the beam's end scoring. | Partly |
| 5 | The app doesn't know who is in each shot, so a brief like "more of Sam" can't work. YuNet only counts faces. | Face embeddings (ArcFace via insightface) -> cluster per album -> user names the clusters in the UI -> director queries and must-include can reference people. | Yes (pretrained + clustering) |
| 6 | Beat This! depends on a host that was down (HTTP 500) during the whole demo, so librosa beats were used everywhere. | Mirror the checkpoint to Drive / an HF repo at a pinned revision, and add a check to `smoke_models.py`. | No |
| 7 | Warm generate takes 29.8 s against a 30 s target, and every recut re-renders every shot. | Per-shot render cache keyed by hash(candidate, src_in/out, duration, framing, kenburns, resolution). A recut then encodes only the changed shots and re-concats. | No |
| 8 | Evaluation uses 2 photo-heavy albums, 1 generic brief, and proxy metrics only. | A larger eval set (6-10 albums from team members, kept on Drive and never committed), 5 briefs of different kinds, VLM-judged brief adherence, and a formal blind human study protocol. | Eval |
| 9 | Output is fixed at 16:9 and 45-60 s, with one song. | 9:16 vertical output with saliency/face-aware crop paths, and a variable target length (full song). | Partly |
| 10 | Session state lives in process memory, and only Colab + Gradio can run the app. | A thin FastAPI layer over `montage/pipeline.py` (generate/recut/export/EDL endpoints). The timeline editor and any future mobile front end need it. | No |

## Part 2: new features

### F1. Song selection from Spotify history (flagship, the deepest ML piece)
New package `montage/songpick/`, used as an optional pre-step before the existing pipeline (`pipeline.generate` stays unchanged once a song path exists).

1. **Ingest history** (`songpick/history.py`): parse Spotify "Extended streaming history" JSON (`ts`, `ms_played`, `master_metadata_track_name`, `master_metadata_album_artist_name`, `spotify_track_uri`, `skipped`, `reason_end`). Store it under the user's per-session DATA_ROOT only. It is personal data: never commit it or log raw track lists, and extend `test_repo_hygiene.py` to block `Streaming_History*.json`.
2. **Time period** (`songpick/period.py`): the album's time window comes from `Asset.taken_at` (robust min/max with padding; config keys). If timestamps are missing, it falls back to the whole history weighted by recency.
3. **Period affinity score (non-learned fallback):** plays in the window weighted by `ms_played`, with a penalty for skips. Add a "signature song" lift: the track's play share in the window divided by its share across the whole history, TF-IDF style. This surfaces "your song of that summer", not just your all-time favourites.
4. **Vibe match, stage A (metadata only, before any audio):** turn each track into text (Last.fm/MusicBrainz tags + genre; only track/artist names leave the machine) -> CLIP text embedding -> similarity to the album's pooled image embedding. Also compare the album's mood valence/arousal (already computed in `features/mood.py`) with the track's valence/arousal predicted from its tags.
5. **Vibe match, stage B (learned, audio):** an **image-set <-> music contrastive model**. Frozen CLIP ViT-L/14 frame embeddings go through attention pooling into a projection. Frozen MERT window features (reusing `montage.music.emotion` code, so train and inference match) go through a projection. Train with InfoNCE on music-video pairs (candidates: HIMV-200K, the YouTube-8M music-video subset, MusicCaps/AudioSet clips with video), with features extracted through our own pipeline. Baselines: valence/arousal distance, tag-CLIP. Metrics: Recall@K / MRR on held-out pairs, plus a human study ("which song fits this album?"). It runs when audio is available: the user's local music folder (matched by ID3 tags/fuzzy filename via `mutagen` + `rapidfuzz`), or audio the user uploads for the shortlist.
6. **Final ranking:** `alpha * period_affinity + (1 - alpha) * vibe`, with the weights in `configs/default.yaml`. It also drops tracks too short for a 45 s excerpt. The output is a shortlist of 3-5 with a "why this song" line from Claude (text only), plus the existing excerpt chooser.
7. **App:** a "Pick a song for me" tab with history upload -> shortlist with reasons -> choose -> upload/match the audio -> the existing Generate flow.
8. **Schemas:** new pydantic models `ListeningEvent`, `TrackCandidate`, `SongPick` in `montage/types.py` + `docs/spec/schemas.md` (shared-contract commit plus a LOG heads-up).

### F2. People-aware editing (pipeline item 5)
Face clustering, naming UI, director/plan support for people (`EditPlan.people_queries` or reuse of `must_include`), plus an evaluation of clustering purity on a labeled team album.

### F3. Learned editing preferences (pipeline item 3)
Preference logging in the app, a reward model, planner weight tuning, and an ablation against hand-tuned weights in the eval grid.

### F4. Timeline editor (last, weeks 7-8)
Edits the EDL directly: reorder, swap a shot from ranked alternatives (the beam already scores them), trim or extend with snapping to beats, and lock shots. "Re-solve around my edits" runs the planner with locked shots as constraints (it extends the pins/excludes mechanism to slot-level locks). The front end is a custom Gradio component, or a small React page against the FastAPI layer (pipeline item 10). The EDL stays the single source of truth; the renderer is unchanged and uses the per-shot cache from item 7.

## Part 3: four-person, eight-week schedule

| Week | Track 1: Song pick (F1) | Track 2: Music + emotion | Track 3: Visual understanding | Track 4: Platform, eval, editor |
|---|---|---|---|---|
| 1 | History parser, period window, affinity score, specs | Beat This! mirror; temporal emotion spec + PMEmo/EmoMusic downloads | CUFED/ML-CUFED download + label mapping | Per-shot render cache; expanded eval set; FastAPI skeleton |
| 2 | Stage A tag/CLIP vibe match; shortlist UI (fallback complete end to end) | Temporal emotion model training | Photo importance ranker training (CUFED) | Eval grid v2: multiple briefs, VLM adherence judge |
| 3 | Music-video dataset download + CLIP/MERT feature extraction | Emotion eval (per-song CCC) + deploy | Face embeddings + clustering + naming UI | Preference logging (votes, recut pairs) in app |
| 4 | Contrastive model v1 training + Recall@K | 9:16 output + variable length (with Track 4) | People in director/plan; event segmentation | Mid-point human study #1 |
| 5 | Model iteration, ablations, local library matching | Section/structure improvements using the temporal model | Coverage/diversity term in the planner + ablation | Reward model over EDLs (with Track 2) |
| 6 | Integrate final ranking; Claude "why this song" | Preference-tuned planner weights | Personalization from pins/excludes | Timeline editor v1 (reorder, swap, trim) |
| 7 | Song-pick human study | Full eval grid with all new components | Clustering/importance eval write-ups | Editor: locks + re-solve around edits |
| 8 | RESULTS.md, demo polish | RESULTS.md | RESULTS.md | DEMO.md v2, final human study, freeze |

Rules carried over from CLAUDE.md:
- Every learned model ships with a fallback and a RESULTS.md table: baseline vs fallback vs learned, on held-out data.
- No precomputed dataset features: everything is extracted through `montage.features` / `montage.music.emotion`.
- New config keys go in `configs/default.yaml`.

## Open questions for the team
- Track ownership: who takes Tracks 1-4?
- Dataset access for the music-video contrastive model (HIMV-200K, YouTube-8M music subset, MusicCaps): which is actually downloadable at our scale?
- Eval albums: who can contribute personal albums (kept on Drive only, never committed)?
- Timeline editor front end: custom Gradio component vs a small React page over the FastAPI layer.

## Next steps once agreed
1. Add phases 12+ (one per row of work above) with owners and dependencies to `docs/PLAN.md`. Record the decisions there: Spotify audio comes from the user; the song-pick fallback is period affinity plus valence/arousal match; the editor goes last.
2. Write `docs/spec/12_songpick.md` (F1 in full, with schema additions and privacy rules). Track owners write the specs for the other phases.
3. Acceptance tests for F1, at minimum:
   - The history parser works on a synthetic fixture from `scripts/make_fixtures.py` (never real data).
   - The period window is correct when EXIF dates are missing.
   - The fallback shortlist works with no network and no weights.
   - A chosen pick hands a valid song path to `pipeline.generate`.
