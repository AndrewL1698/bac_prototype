# PLAN: phases, owners, current work, decisions

This is the plan of record. Progress history goes in `docs/LOG.md`; this file shows where things stand right now.

How to use it: claim a phase in **Current work** before starting (person, branch, start time), tick the box and clear your row when it is merged to `main`, and record any judgment calls under **Decisions**.

## Goal and schedule

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

## Two tracks (so two people can work in parallel)

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
  - [ ] background downloads started: DEAM, SumMe/TVSum
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
- [ ] **9B. DEAM emotion head** trained + plugged in (`spec/09_train_emotion.md`)
- [ ] **10A. Importance ranker** (SumMe/TVSum) trained + plugged in (`spec/10_train_importance.md`)
- [ ] **10B. (optional) CUFED**, only if 9A-10A are done and more than 4 h remain
- [ ] **11. Evaluation, naive-cut compare, demo mode, backups, DEMO.md, RESULTS.md** (`spec/11_eval_demo.md`)
  - [x] demo parts (Track A): demo mode, backups, Rate tab + vote analysis, DEMO.md (merged 2026-10-27; real-album prep run waits for the demo album)
  - [ ] evaluation parts (Track B): eval grid, metrics, render check, report, RESULTS.md

## Current work

| Phase | Person | Branch | Started (ET) | Notes |
|---|---|---|---|---|
| _(none: Keshav's Track A phases are merged)_ | | | | |

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

## Setup checklist

Shared
- [x] Spec pack committed on `main` (Keshav, 2026-10-26)
- [x] Drive folder `montage_data` created in Keshav's My Drive with `demo/`, `demo/songs/`, `eval/albums/`, `eval/songs/`
- [x] colab-mcp registered for Claude Code at project scope: Keshav runs the `claude mcp add --scope project` command from `KICKOFF.md` step 4 and commits `.mcp.json` and `.claude/settings.json` (each person then approves the server once)
- [ ] `montage_data` shared with Andrew (Editor); Andrew adds a shortcut in My Drive

Keshav
- [x] Push access to the repo; local clone; `uv` installed
- [x] Colab Secrets: `ANTHROPIC_API_KEY`, `MONTAGE_USER=keshav`
- [ ] Demo album in `montage_data/demo/<album>/`, songs in `montage_data/demo/songs/`

Andrew
- [ ] Local clone; `uv` installed
- [ ] Colab Secrets: `ANTHROPIC_API_KEY`, `MONTAGE_USER=andrew`
- [ ] Eval: 2 albums in `montage_data/eval/albums/`, 3 songs in `montage_data/eval/songs/` (Track B)
