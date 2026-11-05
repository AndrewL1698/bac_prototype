# PLAN: phases, owners, current work, decisions

This is the plan of record. Progress history goes in `docs/LOG.md`; this file shows where things stand right now.

How to use it: claim a phase in **Current work** before starting (person, branch, start time), tick the box and clear your row when it is merged to `main`, and record any judgment calls under **Decisions**.

> **Now a 4-person, 8-week project (2026-11-04).** The MVP (phases 0-11) is done. The extension (Spotify-history song picking, deeper ML, timeline editor) follows [`ROADMAP.md`](ROADMAP.md). Track owners are TBD until the team agrees; nothing in the Extension section below is claimed yet.

## Goal and schedule

### Extension (4 people, 8 weeks; proposed, see ROADMAP.md)
Team: Andrew, Keshav, and two new devs, one per track (owners TBD). The goal is an ML cohort project with real depth: new trained models, datasets, baselines, ablations, and human evaluation. Every model ships with a fallback and a RESULTS.md table, as in 9B/10A.
- **Weeks 1-2:** fallbacks complete end to end. Song-pick shortlist from listening history, Beat This! mirror, per-shot render cache, expanded eval set, dataset downloads (PMEmo/EmoMusic, CUFED).
- **Weeks 3-4:** first learned models. Temporal emotion, photo importance (CUFED), music-video contrastive v1, face clustering. Preference logging, mid-point human study.
- **Weeks 5-6:** integration. Coverage/diversity term, people in the director, reward model and preference-tuned weights, final song ranking. Timeline editor v1.
- **Weeks 7-8:** editor locks and re-solve, full eval grid, human studies, RESULTS.md, DEMO.md v2, freeze.

**Cut in this order if time runs short:** 1. 9:16 output and variable length, 2. timeline editor locks/re-solve (keep v1), 3. learned editing preferences (F3), 4. people-aware editing (F2), 5. contrastive song model (ship the period-affinity + tag/mood fallback). F1's fallback shortlist is never cut.

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
| **1: Song pick (F1)** | Spotify history -> period affinity -> tag/CLIP and mood vibe match -> image-set <-> music contrastive model -> shortlist UI | history parser, album time window, affinity score, spec 12 | TBD |
| **2: Music + emotion** | Beat This! mirror, temporal emotion model (DEAM + PMEmo/EmoMusic), structure, preference-tuned planner weights | Beat This! mirror; temporal emotion spec; dataset downloads | TBD |
| **3: Visual understanding** | CUFED photo importance, face clustering + naming, people in the director, event coverage/diversity, personalization | CUFED/ML-CUFED download + label mapping | TBD |
| **4: Platform, eval, editor** | per-shot render cache, eval grid v2 + VLM judge, preference logging, reward model, FastAPI layer, timeline editor | render cache; expanded eval set; FastAPI skeleton | TBD |

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

Extension (phases 12+, proposed; numbering and specs set when the team agrees, see ROADMAP.md)
- [ ] **12. Song pick from listening history** (Track 1; ROADMAP F1; spec TBD: `spec/12_songpick.md`)
- [ ] **13. Beat This! mirror + per-shot render cache** (Tracks 2 and 4; ROADMAP items 6-7; spec TBD)
- [ ] **14. Temporal emotion model** (Track 2; ROADMAP item 1; spec TBD)
- [ ] **15. Photo-album importance (CUFED) + personalization** (Track 3; ROADMAP item 2; spec TBD)
- [ ] **16. People-aware editing** (Track 3; ROADMAP F2 / item 5; spec TBD)
- [ ] **17. Coverage and diversity term** (Track 3; ROADMAP item 4; spec TBD)
- [ ] **18. Eval v2: larger eval set, multiple briefs, VLM judge, human studies** (Track 4; ROADMAP item 8; spec TBD)
- [ ] **19. Learned editing preferences** (Tracks 2 and 4; ROADMAP F3 / item 3; spec TBD)
- [ ] **20. 9:16 output and variable length** (Tracks 2 and 4; ROADMAP item 9; spec TBD)
- [ ] **21. FastAPI layer + timeline editor** (Track 4; ROADMAP item 10 / F4; spec TBD)

## Current work

| Phase | Person | Branch | Started (ET) | Notes |
|---|---|---|---|---|
| _(none: all phases merged 2026-10-27; 10B skipped)_ | | | | |

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
