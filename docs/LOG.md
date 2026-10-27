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

## 2026-10-27 01:20 ET | Keshav | phase 4 | branch phase-4-director (merged)
- Done: `montage.director.rules.plan_from_rules` (pacing/chronology/avoid/open/close/length parsing; vibe query = brief minus avoid and length clauses, plus tag words found in the brief), `montage.director.context` (song summary, media inventory, EDL summary, tag-vocabulary hint), `montage.director.apply.compile_plan` -> `Directives` (anchor -> section mapping, per-section pacing and mood, target arousal per slot, z-scored query relevance, director score matrix D (N x K), plan/UI excludes, must-include, pins), `montage.director.make_plan` (LLM hook for 9A, rules fallback recorded as `director:rules`). Spec 05 Day-1 tests (rules parse, anchor mapping, target arousal, D shape + exclude) pass.
- Next: phase 5 (slots, costs, beam search, EDL).
- Blockers: none.

## 2026-10-27 02:00 ET | Keshav | phase 5 | branch phase-5-planner (merged)
- Done: `montage.planning`: slot grid (intensity-driven beats per shot, section pacing, snap to 1/2/4/8 beats, min/max shot length, forced cuts at section boundaries with merge, scarcity lengthening calmest sections first, reuse mode), precomputed costs (importance, mood vs target arousal, valence when MERT, director D, infeasible short windows, photo terms, must/pin bonus, seeded Gumbel variety; pairwise repetition/same-source/visual-jump vs boundary contrast/chronology), vectorized beam search with all spec constraints (no reuse, dup clusters, max windows per video, no overlapping windows, photo band, must-include pruning) and reuse fallback, in/out point search, EDL with whole-timeline frame quantization, framing + Ken Burns decision (`render/framing.py`), per-shot reasons, baselines (naive, beat_random, greedy). 16 synthetic tests pass locally (incl. K=40, M=120, B=128 in well under 2 s; beam <= greedy; mood alignment; boundary contrast) and the real-fixture integration test passes on Colab.
- Next: phase 6 (render).
- Blockers: none.
- Heads-up: `render.max_crop_loss` stays 0.2 per the PLAN decision, so 4:3 photos (25% loss) get fill_blur and 2.2:1 video (19%) gets crop. Spec 07's framing table says the opposite for both, and no single threshold satisfies both; flagged to Keshav as a taste call. Local Windows test runs set `KMP_DUPLICATE_LIB_OK=TRUE` in `tests/conftest.py` (numpy MKL vs torch OpenMP clash).

## 2026-10-27 02:45 ET | Keshav | phase 6 | branch phase-6-render (merged)
- Done: `montage.render`: photo shots in Python (fill_blur composition at 1.15x, largest-16:9 face-centered base rect, smoothstep subpixel Ken Burns via warpAffine, panorama pans) piped as rgb24 to FFmpeg; video shots with seek + crop/native/fill_blur filter graphs, HDR tonemap chain, tpad to guarantee exactly n frames; identical libx264 settings (veryfast, yuv420p, 30 fps, timescale 15360) so `concat -c copy` works; song excerpt with fade-out; mux + ffprobe verification (video packets == frames, audio within 1 frame). Phase 6 tests 7/7 on Colab (framing table, exact shot frame counts incl. past-end tpad, fill_blur no black, rotated upright vs reference, Ken Burns monotonic zoom, scene-detected cuts at EDL frames +/- 1).
- Next: phase 7 (pipeline + CLI), phase 8 (Gradio app).
- Blockers: `montage_data/demo/` is empty: Keshav to add a demo album folder and songs for the phase 7 real-album check and the demo.
- Results: 40-shot render (1575 frames): preview 13.7 s (target < 25 s), 1080p export 22.0 s (target < 60 s).

## 2026-10-27 03:15 ET | Keshav | phase 7 | branch phase-7-pipeline (merged)
- Done: `montage/pipeline.py` (Session persisted under `sessions/`, in-process state with rebuild-from-cache, `analyze`, `make_plan`, `plan_edit` with systems full/naive/beat_random/greedy/no_learned/no_mood/no_pairwise/no_director/no_forced, `render_edl`, `generate`, `recut` with notes/pins/excludes, `export`), `montage/cli.py` (generate/recut/export, `--set`), run artifacts (plan.json, edl.json, meta.json, result.json, preview.mp4, curves.png, costs.png, storyboard.jpg), rule-based notes (`director/notes.py`: faster/slower overall or by part, more X, less/no X, "lose the shot at m:ss") and the editor's-note template (`director/editor_note.py`; LLM versions come in 9A). Phase 7 tests on Colab with every optional component off (rules director, heuristic emotion, librosa beats, heuristic importance): generate end to end, recut "faster" (pacing up, shots >= before), "lose the shot at 0:ss" removes that asset, 1080p export, 3-file upload -> friendly error.
- Next: phase 8 (Gradio app).
- Blockers: real-album timing check waits for Keshav's demo album in `montage_data/demo/<album>/` (+ songs in `demo/songs/`).
- Heads-up: new config key `music.use_beat_this` (default true; false = librosa only). Shared contract: `montage.pipeline` API as in spec 08 section 1 (`generate` returns RunResult; `session_for_run(run_id)` gives its Session).
- Results: fixtures cold generate (15 files, 50 s preview): ingest 3.1 s, features 13.4 s, music 4.2 s, plan 0.2 s, render 9.7 s.

## 2026-10-27 03:40 ET | Keshav | phase 8 | branch phase-8-app (merged)
- Done: `montage/app/gradio_app.py`: inputs (multi-file upload or demo-album dropdown, song upload or demo-song dropdown, brief, advanced: seed/length/excerpt start/thumbnail toggle/system), Generate, Export 1080p, preview player, honest status line (director/emotion/importance/beats fallbacks, shots, BPM), editor's note, readable plan + raw JSON, notes + Recut, storyboard gallery with why-this-shot and Pin/Exclude (applied on the next recut), Under-the-hood tab (curves with beats/sections/cuts, per-shot score bars, stage timings), Compare tab (naive cut beside the current cut). Queue with concurrency 1; user errors show as friendly gr.Error. Phase 8 tests on Colab via gradio_client: generate on a demo album returns a video + plan, recut "faster" raises pacing and records the note, 3 uploaded files -> "Need at least 10" error. App started as job `app`; `app.url` appeared in ~20 s.
- Next: Keshav: open the share link and try your own media + song. Me: phase 9A (LLM director, notes via LLM, editor's note, contact sheet), then demo parts of 11.
- Blockers: none for the app; `montage_data/demo/` still has no album/songs (needed for demo mode and backups in phase 11).
- Heads-up: `paths.load_colab_env()` no longer overrides variables already set in the environment (explicit env wins over `/content/montage_env.sh`).
- Heads-up (security): never commit share URLs (anyone with the link can use the app). The app serves files only from `runs/` and scratch thumbnails; `/content/montage_env.sh`, `sessions/`, `jobs/` are blocked (verified 403).
- Results: share link written to `montage_data/jobs/keshav/app.url` (not in the repo).

## 2026-10-26 23:05 ET | Keshav | session end | branch phase-9a-llm
- Done: Phases 0-8 merged to main (Day 1 goal: Gradio share link, upload media + song + brief -> beat-synced video with fallbacks only). Full write-up for Andrew: `docs/reports/2026-10-26_keshav_track_a.md`.
- Done: Phase 9A built on `phase-9a-llm` (Claude structured-output director, LLM notes, LLM editor's note, contact sheet, error-diffusion beat snapping). 14/15 acceptance checks pass on Colab. Untested phase 11 demo drafts (`demo_mode --prepare`, `eval/vote.py`) are on `wip-phase-11-demo`.
- Next: fix "faster -> more shots" (Claude raised pacing 0 -> 0.3; the cut stayed at 32 shots), merge 9A, then the demo parts of 11.
- Blockers: `montage_data/demo/` has no album/songs yet (Keshav). MERT does not load under Colab's transformers 5.20 (Track B, 9B).
- Heads-up: the times on tonight's phase 1-8 entries above were estimated and ran ahead of the clock (some say 2026-10-27). All of that work happened on 2026-10-26 between about 21:30 and 23:05 ET.

## 2026-10-27 11:45 ET | Keshav | phase 9A | branch phase-9a-llm (merged)
- Done: Claude director (`director/llm.py`, `schema.py`, `prompts.py`): `claude-haiku-5-5` with structured outputs (`messages.create` + `output_config` json_schema from `anthropic.transform_schema`, effort low), enum lowercasing, repair (unknown ids dropped, pacing/energy clamped, length 0 or 45-60, <= 6 queries of <= 8 words), one retry then rules. LLM notes (`revise_from_llm`) and LLM editor's note (`note_from_llm`); optional thumbnail contact sheet (only when `director.send_thumbnails` / the app toggle is on). Pins/excludes already wired through `recut`.
- Done: two fixes for "faster" notes. (1) Error-diffusion snapping in `planning/slots.py`: shots still snap to 1/2/4/8 beats, but the average follows pacing smoothly. (2) A whole-cut speed note moves global pacing AND every explicit section pacing (in the rules and in Claude's revise instructions), with a guard that applies the rule step if Claude leaves pacing unchanged. Root cause on the test album: Claude had set intro/outro pacing -0.5 from "slower at the start", which overrides the global value.
- Done: tests on Colab 17/17: mocked fallbacks (timeout, refusal, max_tokens, bad JSON, wrong-case enums), 3 live briefs valid, plan source llm + LLM editor's note, "faster" 36 -> 39 shots, "lose the shot at 0:05" removes that asset, pin forces inclusion.
- Next: phase 11 demo parts (branch `wip-phase-11-demo`: demo mode, Rate tab, DEMO.md).
- Blockers: `montage_data/demo/` still empty (needs Keshav's album + songs).

## 2026-10-27 12:10 ET | Keshav | phase 11 (demo parts) | branch wip-phase-11-demo (merged)
- Done: `montage/app/demo_mode.py` (`--prepare` warms song and album caches; per album generates + exports a 1080p backup and a naive-cut backup to `demo/backups/`; pre-renders the vote pair to `demo/vote/` + `pairs.json`; times a warm generate; prints a readiness checklist: GPU, Claude key, emotion/ranker weights, Beat This! checkpoint, backups, vote pairs, warm time, app URL). Per-album `brief.txt` and optional `song.txt`. `montage/eval/vote.py` (one CSV per user under `eval/votes/`, exact binomial test, Wilson 95% interval, `python -m montage.eval.vote --analyze`). App Rate tab (Load a pair, left/right order randomized, Left better / Tie / Right better, one vote per loaded pair, live tally). `docs/DEMO.md` run-of-show with the app's real button names, pre-demo checklist, and fallback table.
- Done: Beat This! checkpoint is now cached on Drive (`weights/beat_this-final0.ckpt`) after any successful download and preferred on load; a load failure is remembered per process.
- Tests on Colab: demo prepare on the fixture album (backups + vote pair written, warm generate 16.7 s < 30 s), vote analysis (15/5/2 -> 0.75, correct binomial p), Rate tab via gradio_client, plus the earlier app tests.
- Next: Keshav: put the demo album + songs on Drive, then run `demo_mode --prepare` as a job and start the `app` job (see docs/DEMO.md). Track B: eval grid, metrics, report (spec 11 sections 1-4) and RESULTS.md.
- Blockers: Beat This! checkpoint host (cloud.cp.jku.at) returned HTTP 500 today, so fresh runtimes use librosa beats until it recovers once (then the Drive copy is used forever). `montage_data/demo/` still empty.
- Heads-up (Andrew): `montage/eval/vote.py` is in place (spec 11 section 5); the eval grid/metrics/report modules in `montage/eval/` are still yours.
