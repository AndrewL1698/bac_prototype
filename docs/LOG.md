# LOG: progress history

Append-only. Newest entries at the bottom. Never edit or delete someone else's entry. This file uses git's `union` merge driver (`.gitattributes`), so two people appending at the same time never get a merge conflict locally; if GitHub's web UI reports a conflict on a PR, keep both entries.

Write an entry when you finish a phase, hit a blocker, make a decision that affects the other person, finish a training/eval run, or end a session.

Format:
```
## YYYY-MM-DD HH:MM ET | <person> | phase <N> | branch <name>
- Done: what now works (be concrete: commands, tests passing)
- Next: the next step you or someone else should take
- Blockers: what is stuck and what is needed (or none)
- Heads-up: changes to shared contracts, config keys, data layout, Drive folders (omit if none)
- Results: key numbers from runs (omit if none)
```

---

## 2026-10-26 21:15 ET | Keshav | setup | branch main
- Done: Committed the spec pack: README (how to use the repo), CLAUDE.md (architecture, collaboration and logging rules, Colab procedure), KICKOFF.md (setup + start/resume prompts), docs/PLAN.md, docs/LOG.md, docs/RESULTS.md, docs/spec/ (phases 00-11 + schemas), configs/default.yaml, configs/tags.txt, .gitignore, .gitattributes. No code yet.
- Done: Created the shared Drive folder `montage_data` in Keshav's My Drive with `demo/`, `demo/songs/`, `eval/albums/`, `eval/songs/`.
- Next: Keshav: add Colab Secrets (ANTHROPIC_API_KEY, MONTAGE_USER=keshav), put the demo album and songs in Drive, register colab-mcp at project scope (KICKOFF.md step 4) and commit .mcp.json + .claude/settings.json, then open a local Claude Code session on the clone and paste the start prompt (Track A, Phase 0). Andrew: confirm Track B in docs/PLAN.md, accept the Drive share and add a shortcut, add Colab Secrets (MONTAGE_USER=andrew), install uv, read README and KICKOFF.md.
- Blockers: none.
- Heads-up: tracks proposed as Keshav = A (pipeline, app, demo), Andrew = B (data, training, eval). Swap in docs/PLAN.md if you prefer.

## 2026-10-26 22:30 ET | Keshav | phase 0 | branch phase-0-setup
- Done: Committed `.mcp.json` (colab-mcp, project scope) and `.claude/settings.json` (MCP timeouts) on main. On `phase-0-setup`: pyproject, requirements, `montage/{config,paths,types,log,jobs,cache,weights}.py`, `scripts/{run_job,job_status,colab_bootstrap,make_fixtures,smoke_models}.py`, CLIP + aesthetic loaders, tests (config, types roundtrip, atomic write, job runner, job namespacing, repo hygiene). `pytest -q` green locally (15 passed).
- Next: verify on Colab (bootstrap, fixtures, smoke_models), merge phase 0 to main, then phase 1.
- Blockers: colab-mcp connects, but its notebook-cell tools did not appear in this Claude Code session; the project-scope server shows "pending approval". Keshav to approve it / restart the session.
- Heads-up: Colab bootstrap now runs with `%run` (Drive mount and Secrets only work in the notebook kernel): `!git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null` then `%run /content/bac_prototype/scripts/colab_bootstrap.py --branch <branch>`. Requirements install runs as background job `pip_install` when requirements.txt changes. Andrew: the job runner is on `phase-0-setup` until merged; downloads can start from that branch.

## 2026-10-26 23:20 ET | Keshav | phase 0 | branch phase-0-setup (merged)
- Done: Phase 0 merged to main. On Colab A100 (Python 3.13, torch 2.11 cu130, FFmpeg 6.1.1 with zscale/tonemap): bootstrap summary OK (branch, user keshav, GPU, CUDA, API key, shared Drive ok); `make_fixtures.py` writes all 21 fixtures + truth.json; `pytest -q` 16 passed; smoke: CLIP PASS, aesthetic PASS, Beat This! PASS (99% of beats within 30 ms on click_120), YuNet PASS, Claude API PASS (claude-haiku-5-5 structured output), FFmpeg PASS.
- Next: Keshav: phase 1 (ingest + candidates). Andrew: dataset downloads (DEAM, SumMe/TVSum) can start now; the job runner is on main.
- Blockers: MERT smoke FAIL under transformers 5.20 ("object of type 'NoneType' has no len()"); Track B to find a working pin (see PLAN Decisions). colab-mcp's notebook tools do not load in the Claude desktop app; Keshav's session drives Colab through a small local bridge built on the colab-mcp package instead.
- Heads-up: bootstrap is `%run scripts/colab_bootstrap.py --branch <b>` (see README). Colab Secrets cannot be read by remotely-run cells; run the bootstrap cell once from the Colab UI to save `ANTHROPIC_API_KEY`/`MONTAGE_USER` into `/content/montage_env.sh`. CLIP loads with QuickGELU (`force_quick_gelu=True`).
- Results: smoke {clip PASS (sims 0.27/0.31), aesthetic PASS (4.9/4.5), mert FAIL, beat_this PASS, yunet PASS, claude PASS, ffmpeg PASS}.

## 2026-10-26 23:45 ET | Keshav | phase 1 | branch phase-1-ingest (merged)
- Done: `montage.ingest` (probe with display sizes/rotation/HDR/EXIF dates, probe cache by content hash, count validation, duplicate-file skip), `montage.ingest.frames` (one FFmpeg pass per video: 2 fps analysis frames at 448 px, 15 fps motion pairs at 320 px, 360p proxy; HDR tonemap chain), `montage.candidates` (PySceneDetect on the proxy, 3 s / 1.5 s windows with edge trim and `src_max_end`, thumbnails, `build_media()` orchestrator). Phase 1 tests 12/12 pass on Colab; cold ingest + candidates of all 15 fixtures well under 30 s.
- Next: phase 2 (music analysis).
- Blockers: none.
- Heads-up: frames, thumbnails and proxies live on scratch (`/content/scratch/frames/<hash>/`, `thumbs/`); probe and scene results are cached on Drive under `cache/assets/` and `cache/scenes/`.

## 2026-10-27 00:20 ET | Keshav | phase 2 | branch phase-2-music (merged)
- Done: `montage.music`: load (librosa, FFmpeg fallback), beats (Beat This! `Audio2Beats` on GPU, librosa fallback with low-frequency downbeat phase, half/double mode), energy curves on the 2 Hz grid, structure (beat-sync MFCC+chroma, Foote novelty, peaks snapped to downbeats), emotion (heuristic arousal; MERT path ready), intensity, section labels, excerpt selection with hints, `build_profile` -> `MusicProfile`, `summarize`. Full analysis cached at `cache/music/<song_hash>.json` + `_grid.npz`. Phase 2 tests 17/17 on Colab, including Beat This! F >= 0.95 and librosa F >= 0.9 on 90/120/150 BPM clicks, both `song_sections.wav` boundaries, and a 3.5 min song analyzed in under 20 s.
- Next: phase 3 (visual features, tags, heuristic importance, dedup).
- Blockers: none.
- Heads-up (Andrew, 9B): `montage/music/emotion.py` already has the shared helpers from spec 09: `make_windows(y24, times)` (5 s windows centered on each time, zero-padded) and `mert_window_features(windows)` -> (n, 13, 768) fp16, plus an `EmotionHead` loader matching the spec-09 architecture (reads `state_dict` from `weights/emotion_head.pt`). Import them in `training/emotion/extract_mert.py`; change the head loader if your checkpoint format differs. Cached analyses are recomputed automatically once the head file exists (emotion_source changes to `mert`).

## 2026-10-27 01:10 ET | Keshav | phase 3 | branch phase-3-visual (merged)
- Done: `montage.features` (CLIP ViT-L/14 with parallel preprocessing and a Drive-backed text-embedding cache, LAION aesthetic, quality, Farneback motion + shake, YuNet faces, CLIP mood poles + batch combine, zero-shot tags, near-dup clustering) with per-asset per-frame caching at `cache/features/f1/<hash>.npz`; `extract()` returns a `Features` object (FeatureTable + candidate arrays + per-frame views for in/out). `montage.importance` (heuristic score, hard rejects with relaxation, per-modality rank blend, `ranker_features` + `Ranker.load` matching spec 10's checkpoint format). Phase 3 tests 13/13 on Colab plus 4 local importance tests.
- Next: phase 4 (rule-based director, plan compilation), phase 5 (planner).
- Blockers: none.
- Heads-up (Andrew, 10A): use `montage.importance.ranker.ranker_features(arrays)` for training features (arrays from `montage.features.extract.extract(...).arrays`, or the same keys from your own per-frame aggregation). `Ranker.load` expects `{type, state_dict, feat_mean, feat_std, blend_alpha, ...}` as in spec 10.
- Results: cold features for 60 photos + 20 videos x 30 s (420 candidates): 39.9 s on the A100 (target < 90 s); ingest + candidates for those 80 files: 28.3 s.
