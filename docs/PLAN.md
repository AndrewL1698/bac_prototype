# PLAN: phases, owners, current work, decisions

This is the plan of record. Progress history goes in `docs/LOG.md`; this file shows where things stand right now.

How to use it: claim a phase in **Current work** before starting (person, branch, start time), tick the box and clear your row when it is merged to `main`, and record any judgment calls under **Decisions**.

> **Now a 4-person, 2-week project (re-scoped 2026-11-07).** The product is an **automatic montage maker**: photos/videos in, a story-ordered montage with a song that fits their vibe out, in one click (brief and notes optional; no timeline editor). The MVP (phases 0-11) is done. The extension follows [`ROADMAP.md`](ROADMAP.md). Track owners are TBD.

## Goal and schedule

### Extension (4 people, 2 weeks = 10 working days; see ROADMAP.md)
Team: Andrew, Keshav, and two more, one per track (owners TBD). An ML cohort project: every learned model ships with a fallback and a RESULTS.md table, as in 9B/10A. ML priority: 1. song-vibe matching, 2. CUFED photo importance, 3. image-set <-> music contrastive model, 4. story/event ordering.
- **Days 1-3:** three song modes + tag/CLIP and mood vibe match; CUFED download + labels; contrastive dataset access (**go/no-go on day 3**); event segmentation; one-click generate with the brief optional.
- **Days 4-7:** MERT sentiment match + shortlist UI; CUFED ranker training; contrastive feature extraction + model code; coverage term; "My montages" gallery.
- **Days 8-10:** "why this song" + song-pick mini study; contrastive training + Recall@K; eval grid with the new components; DEMO.md v2; freeze.

**Cut in this order if time runs short:** 1. story/event coverage term (chronological ordering stays), 2. gallery, 3. contrastive model (ship the vibe fallback), 4. CUFED (keep the SumMe/TVSum ranker). Song-vibe matching and the three song modes are never cut.

### MVP (done 2026-10-27)
A live workshop demo of the AI music video editor in two days.
- **Day 1:** working app end to end using fallbacks only (no trained models, rule-based director).
- **Day 2:** LLM director, trained models, evaluation, demo prep.

**Never cut:** phases 0-8 and 9A (core pipeline, app, director).
**Cut in this order if time runs short:**
1. CUFED (10B)
2. Thumbnail contact sheet for the director
3. Extra ablations (keep full vs naive vs beat_random)
4. Pin/exclude buttons (notes still work)
5. Learned importance (10A; ship heuristic only)
6. MERT emotion head (9B; ship heuristic arousal)

## Tracks

### Extension tracks (4 people in parallel; owners TBD)

| Track | Scope | Week-1 deliverables | Owner |
|---|---|---|---|
| **1: Song vibe (F1, ML #1)** | three song modes (Spotify sign-in by release date, best vibe fit, full export) -> tag/CLIP and mood vibe match -> MERT sentiment match -> shortlist UI + "why this song" | three modes, release-date window, stage A vibe match | TBD |
| **2: Photo importance (ML #2)** | CUFED / ML-CUFED ranker + blend, personalization if time | download + label mapping, features | TBD |
| **3: Contrastive song model (ML #3)** | image-set <-> music contrastive model (CLIP + MERT, InfoNCE), Recall@K vs baselines, plug into song ranking | dataset access, go/no-go on day 3 | TBD |
| **4: Story + product (ML #4)** | event segmentation + coverage term, one-click generate (brief optional), "My montages" gallery, eval, DEMO.md v2 | event segmentation, optional brief | TBD |

The Beat This! mirror (half a day) goes to whoever finishes first.

### MVP tracks (two people in parallel; history)

| Track | Phases | Depends on |
|---|---|---|
| **A: pipeline and demo** | 0 (scaffold) -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9A -> 11 (demo mode, DEMO.md) | nothing external |
| **B: data, models, evaluation** | 0 (downloads + smoke tests) -> 9B -> 10A -> (10B) -> 11 (eval grid, metrics, report) | 9B needs only the window helpers in `montage/music/emotion.py` (B writes them first, per spec 09). 10A feature extraction needs phase 3 merged; until then B writes download, label mapping, and training code. 11 eval needs 5-7 merged. |

Phase 0 is shared: whoever starts first builds the scaffold (config, types, paths, jobs, fixtures) and merges it fast; the other starts dataset downloads as soon as the job runner is on `main`.

Assign people to tracks here (proposed; swap if you prefer):
- Track A: Keshav (demoing, so owns the app and demo flow)
- Track B: Andrew

## Phases

Day 1
- [x] **0. Setup and Colab ops** (`spec/00_setup.md`) (merged 2026-10-26; downloads are Track B)
  - [x] scaffold: pyproject, requirements, config loader, paths, types, log, `test_repo_hygiene.py`
  - [x] job runner + status, `colab_bootstrap.py`, `smoke_models.py`
  - [x] fixtures generator; pytest green
  - [x] Colab connected, bootstrapped, GPU verified
  - [x] background downloads: DEAM (Hugging Face mirrors), SumMe/TVSum (Track B, 2026-10-27)
- [x] **1. Ingest and candidates** (`spec/01_ingest_candidates.md`)
- [x] **2. Music analysis**, heuristic emotion (`spec/02_music.md`)
- [x] **3. Visual features, tags, heuristic importance, dedup** (`spec/03_visual_features.md`, `spec/04_importance.md` heuristic part)
- [x] **4. Director: rule-based fallback + plan compilation** (`spec/05_director.md` sections 1-4)
- [x] **5. Slots, costs, beam search, EDL** (`spec/06_planner.md`)
- [x] **6. Render**: preview + export, framing, Ken Burns (`spec/07_render.md`)
- [x] **7. Pipeline + CLI** end to end on fixtures and one real album (`spec/08_app.md` section 1)
- [x] **8. Gradio app v1** (`spec/08_app.md` sections 2-4)

Day 2
- [x] **9A. LLM director**, notes/recut, pin/exclude, editor's note (`spec/05_director.md` sections 5-8, `spec/08_app.md` section 5)
- [x] **9B. DEAM emotion head** trained + plugged in (`spec/09_train_emotion.md`)
- [x] **10A. Importance ranker** (SumMe/TVSum) trained + plugged in (`spec/10_train_importance.md`) (merged 2026-10-27)
- [ ] **10B. (optional) CUFED**, only if 9A-10A are done and more than 4 h remain (skipped, see Decisions)
- [x] **11. Evaluation, naive-cut compare, demo mode, backups, DEMO.md, RESULTS.md** (`spec/11_eval_demo.md`)
  - [x] demo parts (Track A): demo mode, backups, Rate tab + vote analysis, DEMO.md (merged 2026-10-27; real-album prep run waits for the demo album)
  - [x] evaluation parts (Track B): eval grid, metrics, render check, report, RESULTS.md (merged 2026-10-27; the live vote runs at the demo)

Extension (2 weeks, see ROADMAP.md)
- [ ] **12. Song pick: three modes + vibe/sentiment match** (Track 1; ROADMAP F1; `spec/12_songpick.md`; claimed by Andrew, provisional; spec needs the release-date and best-vibe modes)
- [ ] **13. Per-shot render cache + Beat This! mirror** (ROADMAP items 6-7)
  - [x] per-shot render cache (Keshav; `spec/13_render_cache.md`; merged 2026-11-05, PR #1)
  - [ ] Beat This! mirror (whoever finishes first; spec TBD)
- [ ] **15. Photo-album importance (CUFED)** (Track 2; ROADMAP item 2; spec TBD)
- [ ] **17. Story/event ordering: event segmentation + coverage term** (Track 4; ROADMAP item 4; spec TBD)
- [ ] **18. Eval v2 (slim): more albums, with/without brief, song-pick mini study** (Track 4; ROADMAP item 8; spec TBD)
- [ ] **22. Image-set <-> music contrastive model** (Track 3; ROADMAP F1 stage B; go/no-go on day 3; spec TBD)
- [ ] **23. "My montages" gallery** (Track 4; ROADMAP F5; spec TBD)
- Dropped in the 2-week re-scope (2026-11-07): 14 temporal emotion, 16 people-aware editing, 19 learned editing preferences, 20 9:16 and variable length, 21 FastAPI + timeline editor.

## Current work

| Phase | Person | Branch | Started (ET) | Notes |
|---|---|---|---|---|
| 13. Per-shot render cache (render half; spec 13_render_cache.md) | Keshav | phase-13-render-cache | 2026-11-05 | Track 4; PR for review |
| 12.8 Song vibe match, stage A (tags -> CLIP + mood vs the photos) | Andrew | phase-12-vibe | 2026-11-10 | Track 1; 12.1-12.7b merged (PR #5) |

## Decisions

(append: `YYYY-MM-DD | who | decision | why`)

- 2026-10-26 | Keshav | Output is 16:9 1080p; portrait/square media uses blurred-background fill; near-16:9 media is face-aware cropped when the crop loses <= 20% | demo looks consistent and never crops heads
- 2026-10-26 | Keshav | Director = user's free-text brief + notes, compiled by Claude (Haiku 5.5, structured outputs) into an EditPlan; rule-based fallback | user acts as director, system does the rest
- 2026-10-26 | Keshav | Colab bootstrap runs via `%run scripts/colab_bootstrap.py` (not `!python`) | Drive mount and Secrets only work inside the notebook kernel
- 2026-10-26 | Keshav | Colab Secrets are only released to cells run from the Colab UI, not to cells run remotely through colab-mcp. Bootstrap takes `--user` and keeps values a UI run saved in `/content/montage_env.sh` | remote runs time out on `userdata.get`
- 2026-10-26 | Keshav | Requirements: pin packages Colab does not ship (open_clip 3.3.0, gradio 6.30.0, anthropic 2.3.0, beat-this 1.1.0, scenedetect 0.7.2, ...); lower bounds for Colab-shipped ones; no opencv wheel (Colab ships opencv 5.0 with FaceDetectorYN) | pinning Colab's own packages forces reinstalls that break CUDA/numpy
- 2026-10-26 | Keshav | First requirements install runs as background job `pip_install` | keeps the bootstrap cell under ~25 s
- 2026-10-26 | Keshav | Repo-hygiene secret check matches a prefix plus a realistic token body (e.g. `sk-ant-` + 20 chars), not the bare prefix | CLAUDE.md itself mentions the prefixes
- 2026-10-26 | Keshav | CLIP ViT-L-14 with OpenAI weights loads with `force_quick_gelu=True` | open_clip's plain ViT-L-14 config omits QuickGELU; without it embeddings (and the LAION aesthetic head) are off
- 2026-10-26 | Keshav | Colab runtime is Python 3.13, torch 2.11/cu130, transformers 5.20; MERT-v1-95M fails to load under transformers 5.20 (smoke test). Track B: find a working transformers pin for MERT | recorded for 9B
- 2026-10-26 | Keshav | V-JEPA not used in v1 (no text alignment; photos dominate; tiny training set); possible later ablation | scope and timeline
- 2026-10-26 | Keshav | Exact duplicate files (same content hash) are skipped at ingest with a note; very short clips (<1 s usable) still give one candidate if >= 0.5 s | avoids wasting slots, keeps every clip usable
- 2026-10-27 | Keshav | Energy curves average frames in 0.5 s bins centered on each grid time; MusicProfile.tempo is the detected tempo before half/double adjustment (beat_mode says which) | aligns with centered emotion windows; BPM shown to users matches the song
- 2026-10-27 | Keshav | Tags and mood are computed from candidate embeddings at load time using cached text embeddings (one shared npz of CLIP text embeddings on Drive), not stored per asset | changing the tag vocabulary or mood prompts never re-runs the image encoder
- 2026-10-27 | Keshav | Dedup: inside a cluster, the first window of each video keeps the cluster and its other windows become singletons | guarantees windows of one video never share a cluster even in mixed clusters
- 2026-10-27 | Keshav | Rule-based vibe query is the brief minus its avoid/length clauses (not the raw brief) | the raw brief would pull CLIP toward things the director asked to skip
- 2026-10-27 | Keshav | Framing keeps max_crop_loss 0.2 (4:3 photos -> fill_blur, 2.2:1 video -> crop), contradicting spec 07's table on those two rows | the table is internally inconsistent; the PLAN decision wins until Keshav decides otherwise
- 2026-10-27 | Keshav | Video crop placement uses the candidate's face center from features, passed to the renderer as a side table (not in the EDL schema) | EDL schema stays unchanged
- 2026-10-27 | Keshav | App/CLI user errors raise gr.Error / exit 2 with the friendly message (spec said gr.Warning) | a Warning toast does not stop the handler or reach gradio_client, so the error path could not be tested
- 2026-10-27 | Keshav | Beat snapping uses error diffusion (remainder carried to the next shot, clipped to +/-0.5 in log2) | snapping to 1/2/4/8 alone made pacing changes below a factor of 2 invisible
- 2026-10-27 | Keshav | A whole-cut speed note shifts global pacing and every explicit section pacing by the same step; if the LLM revision leaves pacing unchanged, the rule step (+/-0.4) is applied | section pacing overrides global pacing, so a global-only change could leave the cut unchanged
- 2026-10-27 | Keshav | The live vote (eval/vote.py) and Rate tab are built with the demo parts of phase 11 (Track A); eval grid/metrics/report stay with Track B | the vote is part of the demo flow
- 2026-10-27 | Keshav | render.max_crop_loss raised to 0.25: 4:3 photos are face-aware cropped to fill the frame (supersedes the 20% rule above); 5:4 and taller keep the blurred fill | most phone photos are 4:3; full-frame looks better in the montage
- 2026-10-27 | Keshav | HDR videos render from a cached SDR proxy (built once in demo prep; originals used when no proxy exists); tonemapping otherwise happens after downscale/framing | per-frame full-resolution tonemapping made warm generate 49 s on an iPhone album; now 28 s
- 2026-10-27 | Keshav | Keshav takes over Track B (downloads, 9B, 10A, 11 evaluation) | Andrew is unavailable until the demo
- 2026-10-27 | Keshav | DEAM comes from public Hugging Face mirrors (audio: herrjyj/herrjyj-deam-assets DEAM_audio.zip; dynamic 2 Hz annotations: Rehead/DEAM_continuous_annotations, annotation columns only), validated against the DEAM layout | the official host cvml.unige.ch does not respond (from Colab or locally); no Kaggle credentials
- 2026-10-27 | Keshav | Optional app password via Colab Secret APP_PASSWORD (Gradio login, username montage); one-command start scripts/start_app.py waits up to 120 s for the link | anyone with a share link could otherwise use the app and the API key; the start cell is human-run, so a longer wait is fine
- 2026-10-27 | Keshav | MERT layer outputs are captured with forward hooks when transformers returns no hidden_states (5.20 with MERT's remote code); revision pinned to 12af15f | keeps the pinned model working across transformers versions
- 2026-10-27 | Keshav | SumMe/TVSum features run 8 videos per pipeline call with the ingest limits lifted; SumMe "Uncut_Evening_Flight" is damaged inside the published Zenodo archive (md5 verified) and is left out (74 of 75 videos) | one video per call was too slow for the session; the archive is the only source
- 2026-10-27 | Keshav | Ranker selection: best SumMe tau over {SumMe, TVSum, both} x {linear, MLP} in 5-fold CV by video (unless clearly worse on TVSum), then the blend alpha by the same CV: TVSum-trained MLP at alpha 0.75 | spec 10 picks by SumMe, the dataset closest to personal footage
- 2026-10-27 | Keshav | Eval set: two albums from a separate trip that is not in the demo album, split by capture time (`scripts/make_eval_set.py`); songs: the two demo songs plus the most energetic DEAM full song | held out from the demo album; no extra media needed
- 2026-10-27 | Keshav | Scarcity counts distinct shots (one per dup cluster, each video at most its non-overlapping windows), not candidates; in reuse mode a repeated dup cluster costs w_reuse per earlier use and is never adjacent | the first eval grid planned 47 cuts for 44 distinct shots, and reuse mode then filled them with near-identical photos (up to 69 duplicate pairs per cut)
- 2026-10-27 | Keshav | planner.w_dir 1.0 -> 0.6 | eval sweep after the scarcity fix: brief adherence 0.85 -> 0.84, mood coupling 0.23 -> 0.37, quality lift 0.00 -> 0.01
- 2026-10-27 | Keshav | The no_pairwise ablation is judged on mean adjacent CLIP similarity (new eval metric), not on mean similarity over all shot pairs | the pairwise term only scores neighbouring shots
- 2026-10-27 | Keshav | 10B (CUFED) skipped | optional; the learned ranker already beats the heuristic, and the remaining time goes to the demo
- 2026-10-27 | Keshav | A removal note that names a time ("lose the shot at 0:12") excludes the shot playing at that time in the current cut, also when Claude revises the plan (Claude's own exclude edits for that note are dropped) | Claude excluded the wrong shot in test_note_lose_shot; the cut knows exactly which shot plays when
- 2026-10-27 | Keshav | Beat This! tests skip when its checkpoint cannot load (host down, no copy on Drive); the synthetic-song BPM and peak-excerpt checks only run with Beat This! | librosa reads the synthetic song at half tempo, so those checks measure the missing model, not our code
- 2026-11-04 | Andrew | Project reframed as 4 people x 8 weeks following ROADMAP.md; track owners TBD; extension phases 12-21 listed as proposed | MVP is done; the extension needs ML depth
- 2026-11-04 | Andrew | Song pick (phase 12) only shortlists from the user's own Spotify export; the user supplies the audio; the fallback (period affinity) uses no network and no weights | Spotify provides no audio and its terms forbid syncing its content with video
- 2026-11-04 | Andrew | Song-pick track key is "artist|track" (lowercased), not the Spotify uri; capture dates (local) and history times (UTC) are compared as is | the same song has different uris on single and album; the time-zone error (< 1 day) is far below the 7-day padding
- 2026-11-04 | Andrew | Listening-history exports are personal data: never committed (hygiene test blocks Streaming_History*/StreamingHistory*/endsong* JSON), never logged except counts; tests use the synthetic history in scripts/fixtures_history.py | public repo
- 2026-11-05 | Keshav | Shot cache key leaves out the shot's slot and timeline position, so an unchanged shot that moves in a recut is reused | swaps and reorders are the common recut
- 2026-11-05 | Keshav | Shot cache lives on local scratch by default (`render.shot_cache.location: scratch`, LRU-pruned to `max_gb` 5); `drive` is an option | recuts run in the same runtime; writing every shot to Drive would slow cold renders
- 2026-11-05 | Keshav | The encoded song excerpt (AAC) is cached with the shots | it was ~3.4 s of the ~4.2 s fixed assembly cost, and most recuts keep the excerpt
- 2026-11-05 | Andrew | Song pick default flow is Connect Spotify (OAuth PKCE, scopes recently-played + top tracks only) -> shortlist -> opt-in YouTube audio (yt-dlp); the data export stays optional for photos older than about a year; upload stays the fallback | uploading an export plus the MP3 by hand was a poor experience
- 2026-11-05 | Andrew | YouTube audio fetch is off by default and runs only after a per-session opt-in click; Andrew accepts YouTube's terms and copyright risk for the class demo; yt-dlp has a lower bound, not a pin | YouTube changes break old yt-dlp releases; demo-only use
- 2026-11-05 | Andrew | Spotify redirect goes through a fixed GitHub Pages page (docs/spotify/callback.html) that sends the browser back only to *.gradio.live or localhost | share links change on every restart, and Spotify needs a fixed redirect URI
- 2026-11-05 | Andrew | Spotify Development Mode limits accepted: owner on Premium, at most 5 allowlisted users | extended quota needs about 250k monthly users

## Setup checklist

Shared
- [x] Spec pack committed on `main` (Keshav, 2026-10-26)
- [x] Drive folder `montage_data` created in Keshav's My Drive with `demo/`, `demo/songs/`, `eval/albums/`, `eval/songs/`
- [x] colab-mcp registered for Claude Code at project scope: Keshav runs the `claude mcp add --scope project` command from `KICKOFF.md` step 4 and commits `.mcp.json` and `.claude/settings.json` (each person then approves the server once)
- [ ] `montage_data` shared with Andrew (Editor); Andrew adds a shortcut in My Drive

Keshav
- [x] Push access to the repo; local clone; `uv` installed
- [x] Colab Secrets: `ANTHROPIC_API_KEY`, `MONTAGE_USER=keshav`
- [x] Demo album in `montage_data/demo/<album>/`, songs in `montage_data/demo/songs/` (Summer Trip (picked), 2 songs; 2026-10-27)

Andrew
- [x] Local clone; `uv` installed
- [ ] Colab Secrets: `ANTHROPIC_API_KEY`, `MONTAGE_USER=andrew`
- [x] Eval: 2 albums in `montage_data/eval/albums/`, 3 songs in `montage_data/eval/songs/` (Track B; done by Keshav, 2026-10-27)

New dev 1 (name TBD)
- [ ] Push access to the repo; local clone; `uv` installed
- [ ] Colab Secrets: `ANTHROPIC_API_KEY`, `MONTAGE_USER=<name>`
- [ ] Drive shortcut to `montage_data` (shared as Editor)

New dev 2 (name TBD)
- [ ] Push access to the repo; local clone; `uv` installed
- [ ] Colab Secrets: `ANTHROPIC_API_KEY`, `MONTAGE_USER=<name>`
- [ ] Drive shortcut to `montage_data` (shared as Editor)
- 2026-11-07 | Andrew | Project re-scoped to 4 people x 2 weeks; ROADMAP.md rewritten; phases 14, 16, 19, 20, 21 dropped; new phases 22 (contrastive model) and 23 (gallery) | the group's timeline is 2 weeks, not 8
- 2026-11-07 | Andrew | The product is an automatic montage maker, not a video editor: one-click generate, the brief and notes/recut stay optional, no timeline editor | the premise is effortless storytelling from a set of photos and videos
- 2026-11-07 | Andrew | Song pick has three user-chosen modes: Connect Spotify (songs played in the past year whose release date is near the photos' median capture date), best vibe fit (ignore dates), full export upload (period affinity + vibe) | Spotify sign-in only reaches about a year back
- 2026-11-07 | Andrew | Release dates prefer the earliest release (MusicBrainz first-release-date) over Spotify's album date, and the release window widens when few songs qualify | Spotify dates are often remaster/compilation dates
- 2026-11-07 | Andrew | "Best vibe fit" ranks the user's own songs (recent plays, top tracks, export if given), not an open catalog | no catalog with usable audio; Spotify has no audio features for new apps
- 2026-11-07 | Andrew | ML priority: 1. song-vibe matching, 2. CUFED photo importance, 3. contrastive song model (go/no-go on day 3 on dataset access), 4. story/event ordering; cut order is the reverse, with the gallery cut after story ordering | all four are wanted; two weeks is tight
- 2026-11-07 | Andrew | "My montages" gallery: SQLite index + files on the shared Drive, written only by the app process, per MONTAGE_USER, no login; builds on the persisted Session json | simplest durable store on Colab; personal media never leaves Drive
- 2026-11-08 | Andrew | Mode "Songs from back then" (sign-in): eligible = past-year songs whose album release date falls in an age-scaled window around the photos' median date (asymmetric: `release.before_days_by_age`, after = 0.25 x before; widened until 5 songs), or songs actually played during the album (recent plays; the 4-week list for photos that recent); ranked by listening affinity, ties by release closeness | sign-in has no play dates older than the last 50 plays; the 1-year list made every pick "your top 5 of the year"
- 2026-11-08 | Andrew | Remaster/compilation dates are skipped by a name heuristic (`release.unreliable_words`, `album_type: compilation`); a MusicBrainz first-release lookup is deferred | keeps this PR small; MusicBrainz is 1 request/s
- 2026-11-08 | Andrew | "Best vibe fit" ships as a placeholder ranked by listening affinity, with a note; the stage A vibe match (tags -> CLIP text vs album CLIP/mood) is the next PR | keeps the YouTube and time-window fixes reviewable on their own
- 2026-11-08 | Andrew | YouTube on Colab: optional Colab Secret `YOUTUBE_COOKIES` from a spare Google account (raw or base64 cookies.txt), written to local VM disk (`/content/.yt_cookies.txt`, mode 600), never Drive or the repo (hygiene blocks `*cookies*.txt`); `deno` pinned for yt-dlp's YouTube challenges; when still blocked the app links to a YouTube search and asks for an upload | YouTube blocks cloud IPs; the cookies are a full Google login, so never a personal account
- 2026-11-08 | Andrew | Export mode uses the age-scaled listening window (`window_by_age_days`, widened up to `max_window_days` 365) instead of a fixed 7-day pad (`max_pad_days` removed) | the roadmap's "a month for last year, a season for 5 years ago"
- 2026-11-08 | Andrew | `start_app.py` switches to `--branch` first and then loads that branch's own bootstrap (re-running itself once if it changed) | a fresh runtime started on main ran main's bootstrap, which silently ignored new secrets such as SPOTIFY_CLIENT_ID
- 2026-11-10 | Andrew | 12.8 vibe match (stage A): song tags come from Last.fm (Colab Secret `LASTFM_API_KEY`; track tags, artist tags when a track has few) with MusicBrainz artist tags for the gaps (no key, 1 request/s, capped at 10 per Suggest); tags are cached in process memory only, never on disk | Last.fm has the richest mood tags; MusicBrainz keeps it working without a key; a tag list hints at what the user listens to
- 2026-11-10 | Andrew | Song valence/arousal from tags uses a hand-set lexicon (`configs/tag_mood.tsv`, about 100 mood + genre tags, Russell's circumplex), not a learned model or a third-party norms list | no labelled tag data; simple, inspectable, and the MERT head replaces it once audio exists (12.9)
- 2026-11-10 | Andrew | Album mood for song matching = tanh(mean raw CLIP mood-pole score / `album_mood_scale`), pooled per asset; the editor's batch z-scored visual mood is not used | the z-scored album mean is always about 0, so it cannot tell a party album from a calm one
- 2026-11-10 | Andrew | Suggest songs runs the editor's ingest + feature path (cached), not a lighter CLIP-only pass | train/inference parity and a warm Generate afterwards
- 2026-11-10 | Andrew | Ranking: score = alpha * affinity + (1 - alpha) * vibe with `vibe.alpha_by_mode` (release 0.3, vibe 0.15, export 0.5; replaces `songpick.alpha`); the vibe is computed for the best 40 songs by affinity (release: the eligible songs, so eligibility is unchanged); untagged songs get the median vibe | listening still means "back then" in release/export; tag lookups are slow; a missing tag should not decide the rank
