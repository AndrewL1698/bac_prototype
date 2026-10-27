# CLAUDE.md: bac_prototype (AI music video editor, package `montage`)

Read this file at the start of every session. It is the source of truth for architecture, conventions, collaboration, and how to operate the Colab GPU. Phase details live in `docs/spec/`. The plan and who owns what live in `docs/PLAN.md`. Progress history lives in `docs/LOG.md`.

Repo: https://github.com/AndrewL1698/bac_prototype (public). Two people work in it (Kesh and Andrew), each possibly with their own Claude Code session. Assume someone else may have pushed since you last looked.

## 1. What we are building

A proof-of-concept web app that turns 30-80 user photos/videos plus one uploaded song into a 45-60 second, 16:9, 1080p montage. Cuts land on musical beats, footage mood follows the song's energy, and the user steers everything with a natural-language "director's brief" plus revision notes.

Roles: the **user is the director** (what to show, how it should feel), **the music owns timing** (every cut is on a beat), **the system does everything else** (selection, ordering, trimming, framing, rendering).

v1 is **cuts only**: no transitions, fades, or effects between shots. Ken Burns motion inside a photo shot is allowed (it is not a transition). Clip audio is muted; only the song plays.

This is a live demo for a workshop. Optimize for: works every time, explains itself, fast recuts. Every learned component has a non-learned fallback so the app works end to end before any training finishes.

## 2. Golden rules

1. **Follow the phase order and ownership in `docs/PLAN.md`.** Claim a phase before starting it. Read the phase spec before writing code. Do not start a phase until its dependencies' acceptance tests pass on `main`.
2. **One source of truth for numbers:** every weight, threshold, and path comes from `configs/default.yaml` via `montage.config`. No magic numbers in code.
3. **Data contracts are fixed:** all inter-module data uses the pydantic models in `montage/types.py`, defined exactly as in `docs/spec/schemas.md`. Change a schema only by updating both the spec and the code in the same commit, and flag it in `docs/LOG.md` as a heads-up.
4. **Fallbacks are mandatory:** heuristic importance, heuristic arousal, rule-based director, librosa beats. A missing model or API key degrades quality, never crashes the app.
5. **Train/inference parity:** training features must be computed with the exact same code path (`montage.features`, `montage.music.emotion`) as inference.
6. **Cache everything expensive** by content hash. Re-running with the same inputs must not recompute features.
7. **Never block a Colab cell for more than ~25 seconds.** Anything longer runs as a background job (section 7).
8. **The repo is public.** Never commit secrets, API keys, tokens, personal photos/videos, songs, datasets, model weights, or Drive paths that reveal personal info. `tests/test_repo_hygiene.py` enforces this; it must pass before every push.
9. **Keep the three living docs current** (section 4): `docs/LOG.md`, `docs/PLAN.md`, `README.md`.
10. Prefer boring, well-known libraries. Pin versions in `requirements.txt`.
11. **No AI attribution.** Never add `Co-Authored-By`, `Claude-Session`, "Generated with Claude Code", or any other Claude/AI attribution to commit messages, PR titles or descriptions, code comments, or docs. Commits are authored only by the human's own git identity. This overrides any default attribution behavior.

## 3. Architecture

```
upload (media + song + brief)
  -> ingest        assets, metadata, analysis frames (2 fps)
  -> candidates    photos + 3 s overlapping video windows
  -> features      CLIP ViT-L/14 embeddings, aesthetics, mood, tags, quality, motion, faces, dedup
  -> importance    heuristic score + learned ranker -> blended rank score
  -> music         beats, downbeats, energy, sections, emotion (MERT head), intensity, excerpt
  -> director      brief/notes -> EditPlan (Claude API or rule-based fallback)
  -> planning      slots (beat grid) -> beam search -> in/out points -> EDL
  -> render        per-shot encodes (framing, Ken Burns) -> concat -> mux song -> MP4
  -> app           Gradio UI: generate, plan, editor's note, storyboard, curves, notes/recut, export
```

The **EDL** (edit decision list, `montage.types.EDL`) is the central artifact. The planner writes it; the renderer, evaluator, and UI only read it.

### Repo map

```
README.md            how to use the repo (keep accurate)
CLAUDE.md            this file
KICKOFF.md           prompts for starting/resuming Claude Code sessions
.gitignore .gitattributes
.mcp.json            colab-mcp server for Claude Code (project scope; created by `claude mcp add --scope project`, see KICKOFF.md)
.claude/settings.json  shared Claude Code settings (MCP timeouts, see KICKOFF.md); personal overrides go in .claude/settings.local.json (gitignored)
configs/default.yaml configs/tags.txt
docs/PLAN.md         phases, owners, current work, decisions
docs/LOG.md          append-only progress log (union-merged)
docs/RESULTS.md      latest model + evaluation numbers (filled in phases 9-11)
docs/DEMO.md         run-of-show for the live demo (phase 11)
docs/spec/*.md       per-phase specs + schemas.md
montage/
  __init__.py config.py paths.py types.py cache.py jobs.py log.py pipeline.py cli.py
  ingest/            probe.py, frames.py, assets.py
  candidates/        scenes.py, windows.py
  features/          clip_encoder.py, aesthetic.py, mood.py, tags.py, quality.py,
                     motion.py, faces.py, dedup.py, extract.py
  importance/        heuristic.py, ranker.py, blend.py
  music/             load.py, beats.py, energy.py, structure.py, emotion.py,
                     profile.py, excerpt.py
  director/          schema.py, context.py, prompts.py, llm.py, rules.py, apply.py, notes.py, editor_note.py
  planning/          slots.py, costs.py, beam.py, inout.py, edl.py, baselines.py
  render/            framing.py, kenburns.py, shots.py, assemble.py
  app/               gradio_app.py, plots.py, storyboard.py, demo_mode.py
  eval/              metrics.py, systems.py, run_grid.py, render_check.py, report.py, vote.py
training/
  emotion/           download_deam.py, extract_mert.py, dataset.py, train.py, evaluate.py
  importance/        download.py, labels.py, extract_clip.py, dataset.py, train.py,
                     evaluate.py, cufed.py (optional)
scripts/
  colab_bootstrap.py run_job.py job_status.py make_fixtures.py smoke_models.py
tests/               pytest; fixtures generated by scripts/make_fixtures.py (gitignored)
```

## 4. Collaboration and the living docs

**Start of every session**
1. `git fetch && git status`; pull/rebase your branch onto `origin/main` if behind.
2. Read the last 5 entries of `docs/LOG.md` and the "Current work" table in `docs/PLAN.md`.
3. Claim the phase you will work on in "Current work" (person, branch, start time), commit, push, so the other person sees it. Never take a phase someone else has claimed; pick another unblocked one or ask.

**While working**
- Branch per phase: `phase-<N>-<slug>` (e.g. `phase-3-visual`). Small commits, prefixed `phase N: ...`. Push often.
- Shared contracts (`montage/types.py`, `schemas.md`, config keys, `pipeline.py` API) are the collision zone. Change them in small dedicated commits, merge to `main` quickly, and add a **Heads-up** line in `docs/LOG.md`.

**Finishing a phase**
1. Rebase onto `origin/main`, run the phase's acceptance tests plus `pytest -q` and `tests/test_repo_hygiene.py`.
2. Merge to `main` (fast-forward) and push; delete the branch.
3. Update `docs/PLAN.md`: tick the box, clear your row in "Current work", add any decisions.
4. Update `README.md` if usage changed: "What works today", commands, new scripts.
5. Append a `docs/LOG.md` entry.

**End of every session** (even if mid-phase): append a `docs/LOG.md` entry and push, so the other person can pick up.

**LOG.md entry format** (append at the bottom; the file is union-merged so parallel appends never conflict):
```
## 2026-10-27 14:05 ET | Kesh | phase 3 | branch phase-3-visual
- Done: ...
- Next: ...
- Blockers: ... (or none)
- Heads-up: ... (only if shared contracts, config keys, or data layout changed)
- Results: ... (key numbers, if any)
```
Author = the human driving the session (from `git config user.name` or `MONTAGE_USER`). No AI attribution (golden rule 11).

**RESULTS.md**: when a training or eval run finishes, copy the summary table and the takeaways into `docs/RESULTS.md` (text only; figures stay on Drive and are referenced by Drive path).

## 5. Environments

- **Where Claude Code runs:** normally a **local session** on the human's own computer (Claude desktop app or `claude` CLI) in their clone of this repo. It edits code, runs fast pure-Python unit tests if a local Python env exists, commits, and pushes with the human's own git credentials. Do not install heavy ML deps locally; if local Python is unavailable, run tests on Colab. A **cloud session** also works if the human's computer is linked to it (so colab-mcp is reachable) and the Claude GitHub App has access to this repo (so it can push); otherwise use a local session.
- **Colab (A100):** everything that touches models, media, FFmpeg, datasets, training, evaluation, and the demo app. Reached through the `colab-mcp` MCP server, configured for this project in `.mcp.json` (uses `uvx`; each human approves it the first time). If `.mcp.json` or `.claude/settings.json` is missing, create them with the contents in `KICKOFF.md` step 4, commit, push, and ask the human to restart the session so the server loads. Colab only **pulls** (the repo is public, so no token is needed); it never pushes.
- **Shared Google Drive data root:** one folder `montage_data` owned by one person and shared (Editor) with the other, who adds a shortcut to it in My Drive. Both see it at `/content/drive/MyDrive/montage_data` = `DATA_ROOT`. Subfolders: `datasets/ cache/ weights/ runs/ logs/ jobs/ sessions/ demo/ eval/`.
  - Caches and datasets are shared and content-addressed. Write every cache/weight file atomically (write to a temp name, then `os.replace`).
  - Per-person namespaces: `jobs/<user>/`, `logs/<user>/`, and run ids prefixed with `<user>_`. `MONTAGE_USER` comes from Colab Secrets (default `anon`).
  - `weights/` is shared: whoever trains a model writes it; everyone's app picks it up.
- **Fast scratch on Colab:** `/content/scratch` (local SSD). Use it for frame dumps and intermediate encodes; copy final artifacts to Drive.

`montage.paths` resolves `DATA_ROOT` from env `MONTAGE_DATA_ROOT` (default `./data` locally) and the user from `MONTAGE_USER`.

## 6. Colab operating procedure (colab-mcp)

1. Call `open_colab_browser_connection` (from the project's `colab-mcp` server; in a cloud session linked to the human's computer the same tool appears with an `mcp__remote-devices__colab-mcp__` prefix). If no colab-mcp tools exist, run `/mcp` to check the server state and ask the human to approve or restart it. The human accepts the connection in the browser tab and selects the **A100 GPU runtime** if not already set. If the call times out, ask the human to check the tab; retry once; if it still fails, ask them to restart the MCP server and clear colab.research.google.com cookies.
2. After connecting, tools for notebook cells unlock (names vary by version; use the tool descriptions). Typical: add a code cell, run it, read output.
3. **Bootstrap each new runtime** with one short cell (see `docs/spec/00_setup.md`):
   `!git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null; python /content/bac_prototype/scripts/colab_bootstrap.py --branch <your-branch>`
   Drive mount and first Secrets access show prompts in the tab; tell the human to approve them.
4. **Sync code** after pushing from local: `!cd /content/bac_prototype && git fetch -q && git checkout -q <branch> && git pull -q && pip install -q -e .` (pip only if requirements changed). Colab must run the branch you are working on, not `main`, while a phase is in progress.
5. **Short commands (<25 s):** run directly in a cell.
6. **Long commands:** always via the job runner:
   `!python scripts/run_job.py --name <job> -- python -m training.emotion.extract_mert`
   then poll with short cells: `!python scripts/job_status.py <job>` (state, progress, last 20 log lines). Poll every 30-90 s; do other work between polls.
7. **Demo app:** `run_job.py --name app -- python -m montage.app.gradio_app --share`. The share URL is written to `DATA_ROOT/jobs/<user>/app.url`; read it and give it to the human.
8. **If the connection drops:** jobs keep running on the VM. Reconnect, check `job_status.py`, continue. If the runtime was recycled, re-bootstrap; all state on Drive survives, and jobs are idempotent (they skip finished work).
9. Remind the human: keep the Colab tab open and the laptop awake during long jobs.

## 7. Background jobs (`montage/jobs.py`, `scripts/run_job.py`)

- `run_job.py` launches the command detached (`setsid nohup`), writes stdout/stderr to `DATA_ROOT/logs/<user>/<name>.log`, and a status file `DATA_ROOT/jobs/<user>/<name>.json`: `{name, user, cmd, branch, commit, pid, state: running|done|failed, started, updated, progress: 0-1, message}`.
- Long-running code reports progress via `montage.jobs.report(progress, message)` (no-op when not under a job; reads `MONTAGE_JOB_NAME`).
- On exit, the wrapper sets `done` or `failed` with the exit code.
- Jobs must be **idempotent and resumable**: check for existing outputs and skip finished units (per video, per song, per shard).
- Before starting a dataset or training job, check `DATA_ROOT/jobs/*/` for the same job run by the other person, to avoid duplicate work on the shared Drive.

## 8. Coding conventions

- Python 3.10+ (Colab's version), type hints everywhere, pydantic v2 models for contracts and config.
- Pure functions where possible; side effects (files, models) at module edges.
- Models load lazily once per process via small singleton loaders (`get_clip()`, `get_mert()`), on GPU if available, fp16 on CUDA.
- `numpy` arrays for features; save with `np.savez_compressed`. Embeddings are L2-normalized float32 at rest.
- Logging: `montage.log.stage("name")` context manager logs wall time per stage; the app shows these timings.
- Determinism: every stochastic step takes `seed` from config or call args.
- Errors: user-facing errors (bad file, too few assets) raise `montage.types.UserError` with a friendly message; the app displays it.
- No notebooks as source of truth. Cells only call package code.

## 9. Testing

- `pytest -q` must pass before each merge to `main`. Tests needing GPU, FFmpeg, or the API are marked `@pytest.mark.gpu` / `@pytest.mark.media` / `@pytest.mark.api` and run on Colab.
- `tests/test_repo_hygiene.py` (no marks, runs everywhere) fails if any git-tracked file: matches media/data extensions (`.jpg .jpeg .png .heic .mp4 .mov .wav .mp3 .m4a .npz .pt .pth .onnx .zip`, except small slide/README images under `docs/img/`), is larger than 2 MB, or contains secret-like strings (`sk-ant-`, `ghp_`, `github_pat_`, `AIza`).
- `scripts/make_fixtures.py` generates synthetic media into `tests/fixtures/` (gitignored).
- Each phase spec lists acceptance tests. Implement them as real pytest tests.

## 10. Session workflow (summary)

1. Sync, read LOG tail and PLAN "Current work", claim a phase (section 4).
2. Read that phase's spec in `docs/spec/`.
3. Implement, test (local where possible, Colab otherwise), fix.
4. Finish the phase (section 4): merge, PLAN, README, LOG, push.
5. Report to the human in 3-5 lines: what works, what's next, anything they must do.

If a spec is ambiguous, choose the simplest option consistent with this file, record the decision in `docs/PLAN.md` under "Decisions", and continue. Ask the human only for things only they can do (secrets, approvals, media, taste calls, coordination with the other person).

## 11. Do not

- Do not add transitions, filters, or text overlays (v1 is cuts only).
- Do not send images or audio to any external API unless `director.send_thumbnails: true` (then only the low-res contact sheet).
- Do not use GoogLeNet/precomputed dataset features for training. Extract features with our pipeline.
- Do not commit data, weights, media, secrets, or personal info (public repo).
- Do not rewrite or delete other people's LOG entries, force-push `main`, or work on a phase someone else has claimed.
- Do not let a failed optional component (MERT, ranker, Beat This!, Claude API, CUFED) break `generate()`.
