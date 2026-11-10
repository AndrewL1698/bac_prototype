# bac_prototype

Prototype for the BAC project: an **automatic montage maker**. Drop in 30-80 photos and videos and get back a 45-60 second, 16:9, 1080p montage that tells their story: a song that fits the photos' vibe (picked from your Spotify listening, or uploaded), cuts on the beat, and the right photos at the right musical moments. If you want to steer, an optional plain-language brief ("open slow on the beach, save the party for the drop") and notes recut it in seconds. The current 2-week plan is in [`docs/ROADMAP.md`](docs/ROADMAP.md).

- **Status:** see [`docs/PLAN.md`](docs/PLAN.md) (phases, who is on what) and [`docs/LOG.md`](docs/LOG.md) (progress history).
- **Latest numbers:** [`docs/RESULTS.md`](docs/RESULTS.md).
- **Demo script:** [`docs/DEMO.md`](docs/DEMO.md) (written in phase 11).

## What works today

_Keep this list current. Update it when a phase merges to `main`._

- **Phase 0 (scaffold):** config (`montage.config`, dotted `--set` overrides), data contracts (`montage.types`), shared-Drive paths with atomic writes, background jobs (`scripts/run_job.py`, `scripts/job_status.py`), Colab bootstrap, synthetic test media (`scripts/make_fixtures.py`), model smoke tests (`scripts/smoke_models.py`: CLIP, LAION aesthetic, Beat This!, YuNet, Claude API, FFmpeg pass; MERT pending), `pytest -q` green.
- **Phase 1 (ingest + candidates):** `montage.candidates.build_media(paths)` probes, orders, and validates uploads, extracts analysis frames/motion pairs/proxies in one FFmpeg pass per video, detects scenes, and makes photo + 3 s video-window candidates with thumbnails (all cached).
- **Phase 2 (music):** `montage.music.profile.analyze_song(path)` gives beats/downbeats (Beat This!, librosa fallback), 2 Hz energy/intensity/arousal curves, section boundaries (cached by song hash); `montage.music.excerpt.choose_excerpt` + `build_profile` produce a whole-bar 45-60 s `MusicProfile`; `summarize` writes the director's song summary.
- **Phase 3 (visual features + heuristic importance):** `montage.features.extract.extract(assets, candidates, frames)` gives CLIP embeddings, aesthetic, quality, motion/shake, faces, mood, tags, near-dup clusters per candidate (per-asset cache on Drive); `montage.importance.blend.importance(features)` gives rank-normalized importance (heuristic now, learned ranker blended automatically when its weights exist).
- **Phase 4 (director, rules):** `montage.director.make_plan(brief)` turns the brief into an `EditPlan` (rule-based for now); `montage.director.apply.compile_plan` turns a plan into planner directives (section intents, pacing, target arousal, CLIP query scores, excludes/must-includes).
- **Phase 5 (planner):** `montage.planning.plan_edit(PlanInputs, run_id, seed, system)` builds the beat grid, scores every candidate/slot pair, runs the constrained beam search (or a naive/beat_random/greedy baseline), picks in/out points, and writes an `EDL` with frame-exact cuts and a reason per shot.
- **Phase 6 (render):** `montage.render.assemble.render_edl(edl, assets, song_path, preview)` renders frame-exact 720p previews or 1080p exports (Ken Burns photos, face-aware crop or blurred fill for odd aspect ratios, HDR tonemapping), muxes the song excerpt, and verifies frame and audio counts. Encoded shots and the song excerpt are cached (`render.shot_cache`), so a recut only encodes the shots that changed (phase 13).
- **Phase 7 (pipeline + CLI):** `python -m montage.cli generate --media DIR --song FILE --brief "..."` (and `recut --session ID --note ...`, `export --run ID`) runs the whole thing; outputs land in `montage_data/runs/<run_id>/` (preview.mp4, edl.json, plan.json, curves.png, storyboard.jpg).
- **Phase 8 (Gradio app):** `python -m montage.app.gradio_app --share` (as job `app` on Colab): upload media + song + brief -> beat-synced preview, editor's note, plan, storyboard with why-this-shot and pin/exclude, notes + Recut, Export 1080p, under-the-hood curves/timings, naive-cut comparison. Day 1 uses fallbacks only (rule-based director).
- **Phase 9A (LLM director):** with `ANTHROPIC_API_KEY` set, Claude (claude-haiku-5-5, structured outputs) turns the brief into the edit plan, revises it from notes ("faster", "more of the beach", "lose the shot at 0:12"), and writes the editor's note; any failure falls back to the rules and the status line says so. Optional thumbnail contact sheet (off by default).
- **Phase 11 (demo parts):** `python -m montage.app.demo_mode --prepare` (as a job) warms caches, renders 1080p and naive-cut backups to `montage_data/demo/backups/`, pre-renders the vote pair, times a warm generate, and prints a readiness checklist; the app's Rate tab collects blind A/B votes (`python -m montage.eval.vote --analyze`); run-of-show in [`docs/DEMO.md`](docs/DEMO.md).
- **Demo media tools:** `scripts/pick_subset.py --src BIG_ALBUM --dest demo/<album> --n 60` copies a demo-sized subset (longest clips up to 60 s + photos spread over time); demo prep builds cached SDR proxies for HDR phone videos so warm generates stay under 30 s (28 s on the 60-item demo album).
- **Phase 9B (music emotion):** a MERT-based head trained on DEAM (`weights/emotion_head.pt`) gives per-moment valence and arousal; test CCC 0.74 valence / 0.82 arousal. The app's status line shows `Emotion: MERT`.
- **Phase 10A (importance ranker):** a RankNet ranker trained on SumMe/TVSum with features from our own pipeline (`weights/importance_ranker.pt`), blended 0.75 learned + 0.25 heuristic; in cross-validation it beats the heuristic (Kendall tau SumMe 0.17 vs 0.12, TVSum 0.41 vs -0.02). The status line shows `Importance: learned+heuristic`.
- **Phase 12 (song pick, branch `phase-12-songpick`, not merged yet):**
  - In the app, open **Pick a song for me**: **Connect Spotify** (read-only: recent plays and top songs), choose **How to pick**, then **Suggest songs** (3-5 songs with a reason each):
    - **Songs from back then** (default): songs you listened to this year that came out around when the photos were taken (or that you played during them);
    - **Best vibe fit**: dates ignored (for now ranked by how much you play them; vibe matching is next);
    - **My full Spotify history**: upload the Spotify data export to get songs you actually played back then.
  - Opt-in: tick **Fetch the song's audio from YouTube** and press **Use this song** to fill the Song box. On Colab this needs the `YOUTUBE_COOKIES` secret (below); otherwise the app links to a YouTube search and you upload the file.
  - CLI for the export: `python -m montage.songpick --history DIR --start YYYY-MM-DD --end YYYY-MM-DD [--mode vibe]`.
- **Phase 15 (photo-album importance, branch `phase-15-cufed`, not merged yet; days 1-3 of 10):** the CUFED / ML-CUFED event albums (per-photo human importance) are downloaded and turned into ranker training data with our own pipeline. `python -m training.importance.cufed` finds or fetches the dataset (a copy under `montage_data/datasets/importance/cufed/raw/`, the authors' link via `--set importance.cufed.official_url=...`, or the Hugging Face mirror, which only has the 133-album test split) and writes the label mapping; `python -m training.importance.cufed_features` runs every album through the app's own feature pipeline into `cufed_photos.npz`. Training and the evaluation against the SumMe/TVSum ranker come next (spec [`15_cufed.md`](docs/spec/15_cufed.md)).
- **Phase 11 (evaluation):** `scripts/make_eval_set.py` builds held-out eval albums and songs; `python -m montage.eval.run_grid` (as a job) plans every album x song with the full editor, three baselines and five ablations (3 seeds each) and writes metrics, slide figures, and `montage_data/eval/report.md`; `python -m montage.eval.render_check` renders the full cuts and checks with scene detection that every cut lands on its planned frame. Headline numbers in [`docs/RESULTS.md`](docs/RESULTS.md).

## How it works (one paragraph)

The app **sees** your media (CLIP embeddings, an aesthetic score, mood, tags, sharpness, motion, faces, duplicate detection), **listens** to the song (beats and downbeats with Beat This!, energy, section boundaries, and a valence/arousal curve from a MERT-based model trained on DEAM), and turns your brief into a structured **edit plan** (Claude, with a rule-based fallback). An optimizer builds a beat-aligned cut grid, then a beam search picks and orders shots to balance importance, mood match, your direction, variety, and visual flow. FFmpeg renders a frame-exact video with the song. Every learned piece has a non-learned fallback, so the app always works. Full design: [`CLAUDE.md`](CLAUDE.md) and [`docs/spec/`](docs/spec/).

## Run the demo app (Colab, A100)

1. Open a new Colab notebook, Runtime -> Change runtime type -> **A100 GPU**.
2. In the key icon (Secrets) add, with notebook access enabled:
   - `MONTAGE_USER`: your name, lowercase.
   - `ANTHROPIC_API_KEY` (optional): without it the director uses rules.
   - `APP_PASSWORD` (optional): password-protects the app (username `montage`). Recommended, because anyone with the link can use the app and your API credits.
   - `SPOTIFY_CLIENT_ID` (optional): turns on **Pick a song for me** (Connect Spotify). One-time setup is in [`docs/spec/12_songpick.md`](docs/spec/12_songpick.md) ("Human setup"). Spotify's Development Mode allows 5 users, who must be added in the Spotify dashboard.
   - `YOUTUBE_COOKIES` (optional): lets **Use this song** fetch audio from YouTube on Colab, which YouTube otherwise blocks. Use a **spare** Google account, never your own: the cookies work as that account's full login, and YouTube may flag it. Sign in to YouTube in a private window, export youtube.com cookies with a cookies.txt extension, close the window, and paste `base64 -i cookies.txt` output as the secret. Re-export if the app says they expired.
3. Make sure the shared `montage_data` Drive folder is in your My Drive (see "Data and secrets").
4. Run one cell **from the Colab UI** (Secrets are only readable from cells you run yourself):
   ```
   !git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null
   %run /content/bac_prototype/scripts/start_app.py
   ```
   Add `--branch <name>` to run a branch instead of `main`. It mounts Drive, syncs the code, installs requirements (first time), reads Secrets, stops any running app, starts the `app` job, and prints the link (about 1-2 minutes on a fresh runtime). The link is also saved to `montage_data/jobs/<you>/app.url`. Rerun the cell any time to restart the app; each restart gives a new link.
5. In the app: pick a demo album and song (or upload), type a brief, **Generate**. Give notes and **Recut**. **Export 1080p** when happy.

Keep the Colab tab open while you use the app.

## Command line

_Available after Phase 7._ From `/content/bac_prototype` on Colab:

```
python -m montage.cli generate --media /content/drive/MyDrive/montage_data/demo/<album> \
    --song /content/drive/MyDrive/montage_data/demo/songs/<song>.mp3 \
    --brief "Warm summer recap, end on the sunset" --seed 0
python -m montage.cli recut --session <session_id> --note "slower at the start, more of the beach"
python -m montage.cli export --run <run_id>
```
Outputs go to `montage_data/runs/<run_id>/` (video, EDL, plan, curves plot, storyboard).

## Training and evaluation

_Available after phases 9-11._ All long jobs run in the background on Colab:

```
python scripts/run_job.py --name deam_dl     -- python -m training.emotion.download_deam
python scripts/run_job.py --name mert_feats  -- python -m training.emotion.extract_mert
python scripts/run_job.py --name emo_train   -- python -m training.emotion.train
python scripts/run_job.py --name vsum_dl     -- python -m training.importance.download
python scripts/run_job.py --name vsum_feats  -- python -m training.importance.extract_clip
python scripts/run_job.py --name rank_train  -- python -m training.importance.train
python scripts/run_job.py --name cufed_dl    -- python -m training.importance.cufed            # phase 15
python scripts/run_job.py --name cufed_feats -- python -m training.importance.cufed_features   # phase 15
python scripts/run_job.py --name eval_grid   -- python -m montage.eval.run_grid
python scripts/run_job.py --name demo_prep   -- python -m montage.app.demo_mode --prepare
python scripts/job_status.py                 # all your jobs
```
Trained weights land in the shared `montage_data/weights/` and are picked up by everyone's app automatically.

## Working in this repo (two people + Claude Code)

**Read first:** `docs/PLAN.md` (current work), the last few entries of `docs/LOG.md`, then `CLAUDE.md`.

**Living docs**
| File | Purpose | When to update |
|---|---|---|
| `docs/PLAN.md` | phases, tracks, who is on what, decisions | claim a phase before starting; tick it when merged |
| `docs/LOG.md` | append-only progress history | end of every session, phase completion, blockers, run results |
| `docs/RESULTS.md` | latest model/eval numbers | after training or eval runs |
| `README.md` | how to use the repo | when commands or "what works" change |

**Git workflow**
- `main` always works. Work on a branch per phase: `phase-<N>-<slug>`.
- Claim the phase in `docs/PLAN.md` "Current work" and push that change first, so the other person sees it.
- Before merging: rebase on `origin/main`, run `pytest -q` (and the phase's acceptance tests on Colab), then fast-forward merge to `main` and push.
- Changes to shared contracts (`montage/types.py`, `docs/spec/schemas.md`, config keys, `montage/pipeline.py`) go in small commits, merge quickly, and get a **Heads-up** line in `docs/LOG.md`.
- `docs/LOG.md` is union-merged (`.gitattributes`), so parallel log appends do not conflict locally.

**Using Claude Code:** see [`KICKOFF.md`](KICKOFF.md) for setup, the start prompt, and the resume prompt. Open a local Claude Code session on your clone; it edits locally, pushes with your git credentials, and runs everything heavy on your Colab A100 through colab-mcp, configured for the project in `.mcp.json` (approve it once; needs `uv`; see `KICKOFF.md` step 4 if it is missing).

**Two tracks** (details in `docs/PLAN.md`): Track A builds the pipeline, app, and demo; Track B handles datasets, model training, and evaluation.

## Data and secrets

- **This repo is public.** Never commit API keys, tokens, photos, videos, songs, datasets, or weights. `.gitignore` blocks them and `tests/test_repo_hygiene.py` fails the test run if anything slips through.
- **Secrets** live only in each person's Colab Secrets: `ANTHROPIC_API_KEY`, `MONTAGE_USER`.
- **Shared data** lives in one Google Drive folder, `montage_data`, owned by Keshav and shared with Andrew (Editor). The other person adds a shortcut to it in My Drive, so both see `/content/drive/MyDrive/montage_data`. Layout:
  ```
  montage_data/
    demo/<album>/   demo/songs/   demo/backups/
    eval/albums/<album>/   eval/songs/   eval/report.md   eval/figures/
    datasets/   cache/   weights/   runs/   sessions/   jobs/<user>/   logs/<user>/
  ```
  Note that the demo and eval albums are visible to both collaborators.

## Repo map

```
README.md  CLAUDE.md  KICKOFF.md
configs/      default.yaml (every weight and threshold), tags.txt
docs/         PLAN.md  LOG.md  RESULTS.md  DEMO.md  spec/
montage/      the app: ingest, candidates, features, importance, music, director, planning, render, app, eval
training/     emotion (MERT + DEAM), importance (SumMe/TVSum ranker, CUFED photo albums)
scripts/      colab_bootstrap, run_job, job_status, make_fixtures, smoke_models
tests/        pytest suite (fixtures are generated, not committed)
```

## Licenses and data notes

Prototype for non-commercial demo use. MERT-v1-95M is CC-BY-NC-4.0, SumMe is non-commercial, and CUFED is for non-commercial research (its Hugging Face mirror states CC BY-NC 2.0); all would need replacing before any commercial use. DEAM audio is Creative Commons. Datasets and weights are downloaded at runtime and never stored in this repo.
