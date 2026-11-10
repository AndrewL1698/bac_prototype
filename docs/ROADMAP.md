# ROADMAP: 2-week extension (automatic montage maker)

> **Status: agreed scope, 2026-11-07 (Andrew, for the group).** Replaces the 8-week draft of 2026-10-29. Track owners are still TBD (open questions at the bottom). Phases, owners and decisions live in `docs/PLAN.md`; each phase gets a spec in `docs/spec/`.

## What we are building
An **automatic montage maker**, not a video editor. Anyone drops in a set of photos and videos and gets a short montage that tells the story of those moments:
- a **song that fits the vibe** of the photos (picked for them, or uploaded);
- the **right photos at the right musical moments** (cuts on beats, calm shots in calm parts, big moments on the drop);
- **in story order** (the trip or event unfolds as it happened, covering each part of it).

One click makes a montage. The director's brief and notes/recut stay as **optional** extras for people who want to steer. There is no timeline editor. A simple **"My montages" gallery** saves finished montages.

The team is 4 people for **2 weeks** (10 working days). The project is for an ML cohort, so each learned piece ships like 9B/10A: our own feature pipeline, a non-learned fallback, baselines, and a RESULTS.md table.

## Context
The MVP (phases 0-11) works end to end: ingest -> CLIP features -> importance (heuristic + SumMe/TVSum ranker) -> music (beats, MERT emotion head on DEAM) -> Claude director -> beam-search planner -> FFmpeg render -> Gradio app. Since then: per-shot render cache (phase 13, merged 2026-11-05) and the song-pick fallback with Connect Spotify and opt-in YouTube audio (phase 12, on branch `phase-12-songpick`).

Spotify cannot supply audio. Its SDKs only play inside Spotify's DRM player, preview clips and audio features were removed for new apps in Nov 2024, and its developer terms forbid syncing its content with video or training ML on it. So the app uses Spotify only for listening **data** and gets the audio from an opt-in YouTube fetch (yt-dlp, off by default) or an upload.
- **Spotify Development Mode** (since Feb/Mar 2026): the app owner needs Premium; at most **5 allowlisted users**; sign-in shows only the last 50 plays and top tracks for about 4 weeks, 6 months and 1 year.

## ML priority (all attempted, in this order)
1. **Song-vibe matching** (F1): tag/CLIP + mood match, then sentiment match with the MERT emotion head once audio exists. Uses models we already have; no new training required.
2. **Photo-album importance** (item 2): retrain the importance ranker on CUFED / ML-CUFED.
3. **Image-set <-> music contrastive model** (F1 stage B): the most novel piece and the biggest schedule risk.
4. **Story/event ordering** (item 4): event segmentation plus a coverage term in the planner.

## F1. Song pick: three modes (flagship)
Package `montage/songpick/`, an optional pre-step before the existing pipeline (`pipeline.generate` stays unchanged once a song path exists). The user picks a mode; all three end in a shortlist of 3-5 songs with a "why this song" line.

### Modes
1. **Connect Spotify ("songs from back then"):** sign-in only reaches about a year back, so the candidates are songs the user **listened to in the past year** (recent plays + top tracks for the 3 time ranges) whose **release date is near the photos' median capture date**. They are ranked by vibe.
   - Release dates are noisy: Spotify's `album.release_date` is often a remaster, compilation or "Best of" date, so old songs can look new. Use the **earliest release** (MusicBrainz first-release-date, fetched with the tags) when available, and widen the window when few songs qualify.
   - Release window: centered on the album's median capture date, width in config (`songpick.release_window_days`), widened step by step when there are too few candidates.
2. **Best vibe fit:** ignore dates and rank the user's own songs (recent plays + top tracks, plus the export if uploaded) purely by vibe. The pool is the user's own songs, not an open catalog: there is no catalog with audio we may use, and Spotify has no audio features anymore.
3. **Upload the full Spotify export** ("Extended streaming history" JSON): gives the real "what you played back then". Candidates come from an **age-scaled window** around the photos' dates (below), scored with period affinity plus vibe. The export also enlarges the pool for modes 1 and 2.

### Pieces
1. **Listening data:** Spotify sign-in (`songpick/spotify.py`: OAuth PKCE, scopes recently-played + top tracks only) and the optional export parser (`songpick/history.py`: `ts`, `ms_played`, `master_metadata_track_name`, `master_metadata_album_artist_name`, `spotify_track_uri`, `skipped`, `reason_end`). Personal data: kept in the session only, never committed or logged except counts (`test_repo_hygiene.py` blocks `Streaming_History*.json`).
2. **Time windows** (`songpick/period.py`):
   - Mode 1: the release window above.
   - Mode 3: an **age-scaled listening window** centered on the album's median capture date. Its width grows with the photos' age: about a month for photos from a year ago, a season for photos from five years ago, up to about half a year for older ones (piecewise-linear table `songpick.window_by_age_days`, e.g. `[[0, 21], [1, 30], [5, 90], [10, 180]]`). The window always contains the album's own date span and never shrinks below a minimum. If the history does not reach back that far, it widens, then falls back to the whole history weighted by recency. Missing capture dates also fall back to the whole history.
3. **Period affinity** (mode 3; non-learned): plays in the window weighted by `ms_played`, a penalty for skips, and a "signature song" lift (the track's play share in the window divided by its share across the whole history, TF-IDF style), so "your song of that summer" beats your all-time favourites. It decides **eligibility** (a minimum number of real plays) and breaks ties; vibe and sentiment do the ranking.
4. **Vibe match, stage A (metadata only, before any audio):** each track becomes text (Last.fm/MusicBrainz tags + genre; only track/artist names leave the machine) -> CLIP text embedding -> similarity to the album's pooled image embedding. Also compare the album's valence/arousal (`features/mood.py`) with the track's valence/arousal predicted from its tags.
5. **Sentiment match with audio:** once audio exists (YouTube fetch or upload), the song's valence/arousal comes from the existing MERT emotion head (`montage.music.emotion`, the same code path as the editor). Sentiment match = 1 minus the normalized valence/arousal distance; it can also compare curve shape (an energetic album prefers a song with a high-arousal peak).
6. **Vibe match, stage B (learned, audio): image-set <-> music contrastive model.** Frozen CLIP ViT-L/14 frame embeddings -> attention pooling -> projection; frozen MERT window features (via `montage.music.emotion`, so train and inference match) -> projection. InfoNCE on music-video pairs (candidates: HIMV-200K, the YouTube-8M music-video subset, MusicCaps/AudioSet clips with video), features extracted through our own pipeline. Baselines: valence/arousal distance, tag-CLIP. Metrics: Recall@K / MRR on held-out pairs. **Go/no-go on day 3:** if no dataset is downloadable at our scale by then, ship stages A + sentiment and report baselines only.
7. **Final score:** `w_sent * sentiment_match + w_vibe * vibe + w_aff * period_affinity` (affinity only in mode 3, a small weight). All weights live in `configs/default.yaml`. Drop tracks too short for a 45 s excerpt. Shortlist of 3-5 with a "why this song" line from Claude (text only), plus the existing excerpt chooser.
8. **App:** a "Pick a song for me" panel: choose a mode -> Suggest songs -> choose -> Use this song (opt-in YouTube fetch, or upload) -> Generate.
9. **Schemas:** `ListeningEvent`, `TopTrack`, `TrackCandidate`, `SongPick` in `montage/types.py` + `docs/spec/schemas.md` (on the phase 12 branch; a release-date field is added there for mode 1).

## Pipeline improvements kept

| # | Weakness (evidence) | Improvement | ML? |
|---|---|---|---|
| 2 | The importance ranker is trained on video summarization data (SumMe/TVSum), but eval albums are about 90% photos. CUFED (10B) was skipped. | Photo-album importance: train on CUFED / ML-CUFED (event albums with human importance labels), keeping the same RankNet and blend. Personalization from pins/excludes only if time remains. | Yes |
| 4 | The full cut is less varied than random picks (pairwise CLIP similarity 0.61 vs 0.59), and pairwise terms only score neighbouring shots, so the story can skip whole parts of an event. | Story/event ordering: event segmentation (time + CLIP clustering + EXIF GPS), a reward for covering each event, and a diversity bonus in the beam's end scoring. Chronological ordering already exists (`planner.w_chrono`). | Partly |
| 6 | Beat This! depends on a host that was down (HTTP 500) during the whole demo, so librosa beats were used everywhere. | Mirror the checkpoint to Drive / an HF repo at a pinned revision, and add a check to `smoke_models.py`. Small. | No |
| 7 | Warm generate took 29.8 s and every recut re-rendered every shot. | Per-shot render cache. **Done** (phase 13, merged 2026-11-05). | No |
| 8 | Evaluation uses 2 photo-heavy albums, 1 generic brief, and proxy metrics only. | A few more eval albums from team members (Drive only, never committed), runs with and without a brief, and a small song-pick human study. | Eval |

## F5. My montages gallery
A simple saved-montages list: title, date, song, thumbnail, video link, and "reopen" to recut. It builds on the existing `Session` already persisted at `DATA_ROOT/sessions/<id>.json` (`montage/pipeline.py`). Index in SQLite on the shared Drive, written only by the app process (one writer at a time), per-person by `MONTAGE_USER`, no login. Personal media stays on Drive; the database is never committed.

## Two-week schedule

| Days | Track 1: Song vibe (ML #1) | Track 2: Photo importance (ML #2) | Track 3: Contrastive song model (ML #3) | Track 4: Story + product (ML #4) |
|---|---|---|---|---|
| 1-3 | Three song modes; release-date window; tag/CLIP + mood vibe match | CUFED/ML-CUFED download + label mapping; features via our pipeline | Dataset access check; **go/no-go on day 3** | Event segmentation spec + code; brief optional in the UI (one-click generate) |
| 4-7 | Sentiment match with the MERT head once audio exists; shortlist UI | RankNet training + blend; eval vs the SumMe/TVSum ranker | CLIP + MERT feature extraction; model code | Coverage term in the beam's end scoring; gallery (SQLite + Drive) |
| 8-10 | Claude "why this song"; song-pick mini study | Personalization from pins/excludes if time | Training; Recall@K/MRR vs tag-CLIP and V/A baselines; plug into the final score | Eval grid with the new components; DEMO.md v2; freeze |

The Beat This! mirror is a half-day task for whoever finishes first.

**Cut in this order if time runs short:** 1. story/event coverage term (chronological ordering stays), 2. gallery, 3. contrastive model (ship the vibe fallback), 4. CUFED (keep the SumMe/TVSum ranker). Song-vibe matching and the three song modes are never cut.

Rules carried over from CLAUDE.md:
- Every learned model ships with a fallback and a RESULTS.md table: baseline vs fallback vs learned, on held-out data.
- No precomputed dataset features: everything is extracted through `montage.features` / `montage.music.emotion`.
- New config keys go in `configs/default.yaml`.

## Out of scope (was in the 8-week draft)
- Temporal emotion model (GRU/transformer over MERT windows, PMEmo/EmoMusic).
- Learned editing preferences (preference logging, Bradley-Terry reward model, tuned planner weights).
- People-aware editing (face embeddings, clustering, naming, people in the director).
- 9:16 vertical output and variable length.
- FastAPI layer and the timeline editor.

## Open questions for the team
- Track ownership: who takes Tracks 1-4?
- Contrastive dataset: which of HIMV-200K, the YouTube-8M music subset, or MusicCaps is actually downloadable at our scale (answer by day 3)?
- Eval albums: who can contribute personal albums (kept on Drive only, never committed)?
