# Track A report: 2026-10-26 (Keshav)

Written for Andrew at the end of Keshav's session (about 23:05 ET). Phases 0 through 8 are merged to `main`. That is the Day 1 goal: a Gradio app where you upload photos, videos, a song and a brief, and get back a beat-synced montage using only the fallbacks. Phase 9A (Claude as director) is built on a branch but not merged. The Phase 11 demo tools exist only as untested drafts.

Timestamps: everything here happened on 2026-10-26, between about 21:30 and 23:05 ET. The times on tonight's `docs/LOG.md` entries for phases 1-8 were estimated and ran ahead of the clock (some say 2026-10-27). The same goes for a few dates under PLAN "Decisions". The work and the order are right; only those times are off.

## 1. Where things stand

| Phase | State | Where |
|---|---|---|
| 0 Setup, job runner, fixtures, smoke tests | merged | `main` |
| 1 Ingest and candidates | merged | `main` |
| 2 Music analysis (heuristic emotion) | merged | `main` |
| 3 Visual features, heuristic importance, dedup | merged | `main` |
| 4 Rule-based director, plan compilation | merged | `main` |
| 5 Slots, costs, beam search, EDL | merged | `main` |
| 6 Render | merged | `main` |
| 7 Pipeline + CLI | merged | `main` |
| 8 Gradio app v1 | merged | `main` |
| 9A LLM director, notes, editor's note | built; 14/15 acceptance tests pass; one open item (section 4) | branch `phase-9a-llm` |
| 11 (demo parts) | drafts, not tested | branch `wip-phase-11-demo` (built on `phase-9a-llm`) |
| 0 downloads, 9B, 10A, 11 eval | Track B, not started | - |

On `main`, `pytest -q` passes locally (48 passed, the rest skip without a GPU, FFmpeg or fixtures). Every phase's GPU/media tests passed on the Colab A100.

## 2. What works on `main`

End to end, on Colab:
```
!git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null
%run /content/bac_prototype/scripts/colab_bootstrap.py --branch main
!cd /content/bac_prototype && python scripts/run_job.py --name app -- python -m montage.app.gradio_app --share
!cd /content/bac_prototype && python scripts/job_status.py app      # prints the share URL
```
or from the command line: `python -m montage.cli generate --media DIR --song FILE --brief "..."`.

By phase:
- **0:** `montage.config` (YAML -> pydantic, dotted `--set` overrides, `config.override({...})` for ablations), `montage.types` (exactly `schemas.md`), `montage.paths` (DATA_ROOT/SCRATCH/user, `atomic_write`), background jobs (`scripts/run_job.py`, `scripts/job_status.py`, `montage.jobs.report`), `scripts/colab_bootstrap.py`, `scripts/make_fixtures.py` (21 synthetic files + `truth.json`), `scripts/smoke_models.py`, and `tests/test_repo_hygiene.py`.
- **1:** `montage.candidates.build_media(paths)` covers ingest, probe, frames, scenes and candidates. Probing gives display size, rotation, HDR and EXIF dates, cached by content hash. Each video gets one FFmpeg pass that writes 2 fps analysis frames, 15 fps motion pairs and a 360p proxy. Scene detection runs PySceneDetect on the proxy. Videos are cut into 3 s windows on a 1.5 s hop, plus thumbnails.
- **2:** `montage.music.profile.analyze_song(path)` gives beats and downbeats (Beat This! on GPU, librosa fallback), 2 Hz energy, intensity and arousal curves, and Foote-novelty section boundaries, all cached per song. `music.excerpt.choose_excerpt` + `build_profile` produce the 45-60 s whole-bar excerpt as a `MusicProfile`. `summarize` writes the text the director sees.
- **3:** `montage.features.extract.extract(...)` gives CLIP ViT-L/14 embeddings (with QuickGELU), the LAION aesthetic score, sharpness, brightness, contrast, exposure and colorfulness, Farneback motion and shake, YuNet faces, CLIP mood poles, zero-shot tags and near-duplicate clusters. Features are cached per asset and per frame at `cache/features/f1/<hash>.npz`. `montage.importance.blend.importance(features)` gives the heuristic score rank-normalized within modality, and blends in the learned ranker automatically when `weights/importance_ranker.pt` exists.
- **4:** `montage.director.make_plan(brief)` returns an `EditPlan`, rule-based on `main`. `director.apply.compile_plan` turns the plan into `Directives`: section mapping, pacing and mood per section, target arousal per slot, z-scored CLIP query relevance, the director score matrix D, excludes, must-includes and pins.
- **5:** `montage.planning.plan_edit(PlanInputs, run_id, seed, system)` builds the beat grid (section boundaries force cuts), precomputes the costs, and runs a vectorized beam search with every spec constraint. It then picks in/out points and writes an EDL with frame-exact cuts and a reason for each shot. Baselines: `naive`, `beat_random`, `greedy`.
- **6:** `montage.render.assemble.render_edl(...)` renders Ken Burns photos in Python (subpixel `warpAffine`) and video shots with FFmpeg (crop, native or blurred fill, HDR tonemap, `tpad` for an exact frame count). All shots use identical encode settings, so they concat with stream copy; the song excerpt fades out, and frame and audio counts are verified with ffprobe.
- **7:** `montage/pipeline.py` provides `analyze`, `make_plan`, `plan_edit`, `render_edl`, `generate`, `recut` and `export`, plus every system named in spec 11 (`full`, baselines, ablations). It also includes the rule-based notes ("faster", "slower at the start", "more X", "no X", "lose the shot at m:ss") and the editor's-note template. Each run writes `runs/<run_id>/{plan,edl,meta,result}.json`, `preview.mp4`, `curves.png`, `costs.png` and `storyboard.jpg`.
- **8:** `montage/app/gradio_app.py` has upload or demo-album/demo-song pickers, the brief, advanced options, Generate, an honest status line, the editor's note, the plan (readable and JSON), notes + Recut, and a storyboard with why-this-shot and Pin/Exclude. There are also an under-the-hood tab (curves, score bars, timings), a Compare tab that renders the naive cut, and Export 1080p.

## 3. Key numbers (Colab A100)

| What | Result | Target |
|---|---|---|
| Smoke tests | CLIP, aesthetic, Beat This!, YuNet, Claude API, FFmpeg pass; **MERT fails** | CLIP, aesthetic, FFmpeg |
| Beats on click tracks (90/120/150 BPM) | Beat This! F >= 0.95, librosa F >= 0.9 | same |
| Full-song analysis, 3.5 min song | < 20 s | < 20 s |
| Cold features, 60 photos + 20 videos x 30 s (420 candidates) | 39.9 s (ingest + candidates for those 80 files: 28.3 s) | < 90 s |
| Beam search, K=40, M=120, B=128 | well under 2 s (CPU) | < 2 s |
| 40-shot render (1575 frames) | preview 13.7 s, 1080p export 22.0 s | < 25 s / < 60 s |
| Cold generate on the 15 fixtures, 50 s preview | about 31 s (features 13.4 s, render 9.7 s, music 4.2 s, ingest 3.1 s) | - |
| App start to `app.url` | about 20 s | < 60 s |

## 4. Phase 9A (branch `phase-9a-llm`, not merged)

Built so far:
- `director/schema.py`, `director/prompts.py` and `director/llm.py`. Claude (`claude-haiku-5-5`, the model set in config) is called with structured outputs (`messages.create` + `output_config` json_schema, effort `low`). Enums are lowercased before validation. The plan is repaired: unknown asset ids dropped, numbers clamped, query lists capped. Each call is retried once, then falls back to rules.
- `revise_from_llm` (notes) and `note_from_llm` (editor's note), plus an optional thumbnail contact sheet, sent only when `director.send_thumbnails` is on.
- Error-diffusion beat snapping in `planning/slots.py`, so small pacing changes still change the cut on average.

Tests on Colab: mocked failure paths (timeout, refusal, max_tokens, bad JSON, wrong-case enums) all pass. Three live briefs produce valid plans. The plan source shows "llm" and the editor's note comes from Claude. "Lose the shot at 0:05" removes that asset, and a pin forces the pinned shot into the cut.

**Open item:** "faster increases the shot count" still fails. On the 9A test album, Claude raised pacing 0.0 -> 0.3 and the cut stayed at 32 shots, even with the snapping fix. Next step: print the per-section pacing and beats per shot before and after. Likely causes are segments with explicit pacing overriding the global value, loud sections already at the 1-beat minimum, or rule-based "faster" (+0.4) needing a bigger step.

## 5. Things Andrew needs to know (Track B)

- **Job runner is on `main`.** Dataset downloads (DEAM, SumMe/TVSum) can start: `python scripts/run_job.py --name deam_dl -- python -m training.emotion.download_deam`. Check `python scripts/job_status.py --all-users` first.
- **Bootstrap with `%run`, not `!python`.** Drive mount and Secrets only work inside the notebook kernel. **Colab Secrets cannot be read by cells run remotely (colab-mcp)**: run the bootstrap cell once yourself from the Colab UI, which saves `ANTHROPIC_API_KEY` and `MONTAGE_USER` into `/content/montage_env.sh` for later remote runs. You can also pass `--user andrew`.
- **MERT fails to load under Colab's transformers 5.20** ("object of type 'NoneType' has no len()"). 9B needs a working transformers pin, or a separate venv for MERT. Colab runs Python 3.13, torch 2.11 + cu130 and FFmpeg 6.1.1 (with zscale).
- **Emotion helpers already exist** in `montage/music/emotion.py`: `make_windows(y24, times)` (5 s windows centered on each time, zero-padded) and `mert_window_features(windows)` returning (n, 13, 768) fp16. There is also an `EmotionHead` loader matching the spec-09 architecture (`state_dict` in `weights/emotion_head.pt`). Import them in `training/emotion/extract_mert.py`, and change the loader if your checkpoint differs. Cached song analyses recompute automatically once the head file exists.
- **Ranker:** train on `montage.importance.ranker.ranker_features(arrays)`. `Ranker.load` expects `{type, state_dict, feat_mean, feat_std, blend_alpha, ...}` as in spec 10. Learned importance blends in automatically when the file exists.
- **Eval (yours in 11):** `pipeline.plan_edit(session, plan, seed, system)` already supports `full`, `naive`, `beat_random`, `greedy`, `no_learned`, `no_mood`, `no_pairwise`, `no_director` and `no_forced`. EDLs are written to `runs/<run_id>/edl.json`. `planning.beam.objective(cm, cons, seq)` scores any sequence.
- **Live vote:** I drafted `montage/eval/vote.py` (one CSV per user, exact binomial test, Wilson interval) as part of the demo Rate tab. It is on `wip-phase-11-demo` and untested. Tell me if you'd rather own it.
- **Shared-contract changes today:** new config key `music.use_beat_this` (default true). `paths.load_colab_env()` no longer overrides variables already set in the environment. No schema changes.
- **colab-mcp in the Claude desktop app:** the server connects, but its notebook-cell tools never load (the app doesn't pick up `tools/list_changed`). Keshav's session drove Colab through a small local bridge built on the colab-mcp package itself, kept in his scratch folder. If you hit the same problem in your own session, ask, or run the `claude` CLI in a terminal instead.
- **Security:** never commit Gradio share URLs; anyone with the link can use the app. The app serves files only from `runs/` and scratch thumbnails. `/content/montage_env.sh` (holds the API key), `sessions/` and `jobs/` are blocked; this was checked with HTTP 403s.
- **Local tests on Windows/conda:** `tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE`, because numpy (MKL) and torch ship clashing OpenMP runtimes that abort the interpreter.

## 6. Decisions made today

All are recorded under PLAN "Decisions":
- Bootstrap runs via `%run`; the first requirements install runs as background job `pip_install`.
- Requirements: pin only packages Colab doesn't ship; lower bounds for Colab's own packages; no extra opencv wheel.
- The repo-hygiene secret check matches a prefix plus a realistic token body, since CLAUDE.md itself mentions the prefixes.
- CLIP loads with `force_quick_gelu=True` (the OpenAI weights need it).
- Tags and mood are computed from cached embeddings, using one shared npz of CLIP text embeddings on Drive.
- Dedup never puts two windows of one video in the same cluster.
- The rule-based vibe query is the brief minus its avoid, open, close and length clauses.
- Framing keeps `max_crop_loss` 0.2, so 4:3 photos get blurred fill and 2.2:1 video gets cropped. Spec 07's table says the opposite for both rows and is internally inconsistent. **Taste call pending for Keshav:** raise to 0.25 so phone photos fill the frame?
- User errors show as `gr.Error`, not `gr.Warning`, so the error path can be tested.
- Video crop placement uses face centers passed to the renderer in `meta.json`; the EDL schema is unchanged.

## 7. Open items and next steps

1. **Keshav:** add a demo album to `montage_data/demo/<album>/` (optionally with `brief.txt`) and songs to `montage_data/demo/songs/`. Both folders are empty, and demo mode, backups and the real-album timing check need them.
2. **Keshav:** decide the 4:3 framing question in section 6.
3. **Track A, next session:** fix the "faster" item, then merge 9A. After that, test and merge the Phase 11 demo parts on `wip-phase-11-demo` (`demo_mode --prepare`, the Rate tab, `docs/DEMO.md`).
4. **Track B:** downloads, 9B (with a transformers pin that loads MERT), 10A, and the eval parts of 11.
