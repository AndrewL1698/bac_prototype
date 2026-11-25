# Phase 24: Simple everyday UI

ROADMAP F6. Owner Andrew; Track 4 (Keshav) reviews.

Goal: someone with a folder of trip photos and no editing background opens the app, does three obvious things, and gets a montage. Today's page shows every control at once (Spotify connect and code paste, upload/demo radios, the song-pick accordion, Advanced, a three-column workspace, the Revise panel and the Lab). Phase 24 keeps everything the pipeline does and changes only what the user sees.

The work comes in three stages, and each one ships on its own:
- **24.0** simplifies the page inside Gradio (layout, `theme.py`, CSS). No new stack.
- **24.1** adds a React (TypeScript) frontend that calls the running Gradio backend through `@gradio/client`.
- **24.2** is a full FastAPI + React switch, done only if a trigger in section 7 is hit.

## 1. Goals and non-goals

Goals:
- A guided 3-step flow on the main page: add photos/videos, pick a song, generate.
- A calmer look in the existing Print Table style.
- Team tools (plots, ablations, votes) moved off the main page but still one click away.

Non-goals (unchanged from CLAUDE.md and ROADMAP):
- No timeline editor. This is a montage maker, not a video editor. Revising stays notes + keep/remove + recut.
- No transitions, filters or text overlays (v1 is cuts only).
- No relabelling: buttons and fields keep their current names (Generate, Recut, Director's brief, Export 1080p, ...).
- No pipeline, model or EDL changes. The UI only calls `pipeline.generate()` / `pipeline.recut()` through the existing handlers.

## 2. Everyday flow (`/`)

Before the first cut:

```
+--------------------------------------------------------------+
| Montage                                                      |
| Have Spotify? Connect first for song picks.   [Connect]      |
|                                                              |
| 1  Add photos and videos                                     |
|    [ drop files here ]          Try with sample photos       |
|                                                              |
| 2  Pick a song                                               |
|    [ upload a song ]            Use a sample song            |
|    > Pick a song for me (Spotify)                            |
|                                                              |
| 3  Director's brief (optional)                               |
|    [ ............................................. ]         |
|                                   [ Generate ]               |
+--------------------------------------------------------------+
```

After the first cut, the steps collapse into a one-line summary (photos, song, brief; click to edit) and the result fills the page:

```
+--------------------------------------------------------------+
| 42 photos . "Song name" . brief          [edit]              |
|                                                              |
| +--------------------------------+  Director                 |
| |                                |  editor's note            |
| |           video                |  notes history            |
| |                                |  > The plan               |
| +--------------------------------+                           |
|          [ Export 1080p ]                                    |
|                                                              |
| Revise   pick a shot, keep or remove it, add a note, recut   |
| [storyboard strip ..........................................]|
| selected shot [Keep][Remove] | note box | Changes [Recut]    |
|                              |          | Put back removed   |
+--------------------------------------------------------------+
```

Rules:
- **Steps.** Generate stays disabled until step 1 has media and step 2 has a song. The brief is optional, as it already is in the pipeline.
- **Spotify.** Connecting reloads the page (Spotify sends the browser back), so in 24.0 the Connect line sits above step 1, where today's button already is. 24.1 removes the reload (section 6).
- **Sample media.** "Try with sample photos" and "Use a sample song" appear only when `DATA_ROOT/demo` has albums or songs (`demo_albums()`, `demo_songs()`). They replace today's Upload/Demo radios, and upload stays the default. Live demos run from `/`.
- **Export.** One Export 1080p button under the video renders the 1080p file through the existing `export` endpoint, shows progress on the video, then downloads the MP4. It replaces the top-bar button and the separate file box.
- **Revise and Director.** Both stay complete: the Revise panel as today (storyboard, selected shot with Keep/Remove, note box, Changes, Recut, Put back removed shots) and the Director column (editor's note, notes history, "The plan" accordion). Only their styling changes.
- **Hidden on `/`:** the Lab (curves, per-shot costs, stage timings, plan JSON, Compare, Rate) and Advanced (seed, target length, excerpt start, thumbnails, system). Generate uses the config defaults for those.

## 3. Team lab (`/lab`)

Today's full app moves unchanged to a second page at `/lab` (Gradio multipage: `Blocks.route`; check the exact API in the pinned `gradio==6.30.0` at the start of 24.0). Everything the team uses for demos and evaluation stays there: Advanced, Under the hood, Compare, Rate.

Each page has its own `gr.State`, so the lab doesn't see the session from `/`. The lab gets a "Run id" box that reopens a run with `pipeline.session_for_run(run_id)`. The everyday page shows the run id in small muted text under the video for the team.

Both pages share one server, so there's one share link and one `served_paths()` allow/block list.

## 4. Look

Keep **Print Table** (`montage/app/theme.py`): neutral black table, white instant-print frames, one signal red, Archivo, contrast already checked. Make it calmer:
- more whitespace between steps, fewer panel borders (steps are separated by space and the step number, not boxes);
- one primary (red) button visible at a time: Generate before the first cut, Recut after it;
- larger step titles and primary buttons; secondary actions as text links;
- keep the existing contrast pairs; any new color gets its ratio in the `theme.py` comment.

24.1 reuses the same tokens as CSS variables (`--ground`, `--panel`, `--accent`, ...), generated from or copied next to `theme.py`, so both UIs match.

## 5. Stage 24.0: Gradio cleanup

What changes, by today's component in `montage/app/gradio_app.py`:

| Today | 24.0 on `/` |
|---|---|
| Top bar: wordmark + Export 1080p | Wordmark only; Export moves under the video |
| Media panel: Connect Spotify, source radios, uploads, song-pick accordion, Advanced | Steps 1 and 2; Connect line above step 1; sample links replace the radios; Advanced moves to `/lab` |
| Director panel: brief, Generate, editor's note, notes, The plan | Step 3 holds the brief and Generate; after the first cut, the Director column (note, notes, The plan) sits beside the video |
| Monitor column: video, readout, export file box | Video and readout; the file box goes away (Export downloads directly) |
| Revise panel | Unchanged, restyled; shown only after the first cut |
| Lab accordion | Moves to `/lab` |

Contract: the `generate`, `recut`, `export`, `naive`, `load_pair` and `vote_*` endpoints keep their names, inputs and outputs. In particular, the first 11 outputs of generate/recut (`tests/test_app.py`) stay as they are. 24.1 depends on them. Hidden components that are still endpoint inputs (seed, system, ...) stay in the Blocks with `visible=False` and default values.

## 6. Stage 24.1: React on `@gradio/client`

- **Stack:** React + TypeScript + Vite in a new top-level `web/` folder. `@gradio/client` calls the named endpoints above, and its job/status events drive the progress display. No new backend code except serving the build.
- **Hosting:** the built `web/dist/` is served by the same Colab VM and share link as the Gradio backend (same origin: no CORS, and the Spotify callback allowlist `*.gradio.live` / localhost stays the same). The mounting method (a static route on Gradio's FastAPI app, or similar) is a half-day spike at the start of 24.1. If it can't be done cleanly, that's trigger (a) in section 7.
- **Spotify without the reload:** the sign-in opens in a popup. `docs/spotify/callback.html` passes the code back to its opener with `postMessage` to the exact app origin taken from `state`, and only for the hosts `spotify.safe_return_url` allows. The current redirect stays as the fallback when popups are blocked. This change is security-sensitive (open redirect, message origin) and is reviewed on its own.
- **Privacy carried over (CLAUDE.md section 11):** the Spotify token and listening data stay in the backend session (`gr.State`), never in the browser's localStorage/sessionStorage and never logged. Videos and thumbnails come only from paths `served_paths()` allows. Nothing new leaves the machine.
- **Same flow and look** as sections 2 and 4. The team lab stays the Gradio page at `/lab`.
- Once `web/` exists, CLAUDE.md's repo map and README gain it.

## 7. Stage 24.2: full switch (only on a trigger)

Stay on Gradio as the backend unless one of these happens:
- **(a) `@gradio/client` limits:** progress, uploads or video playback can't be done cleanly through it.
- **(b) Real users or scale:** more than a few people at once, accounts or login beyond Gradio's basic auth, or hosting off Colab.
- **(c) Team bandwidth:** someone has 3+ free days after the extension freeze. Never during the 2 weeks.

If triggered, the switch is a FastAPI app with endpoints wrapping `pipeline.generate` / `recut` / export, a background job with a job id and progress (polling or server-sent events), uploads, video serving with the same allow/block rules as `served_paths()`, server-side session state holding the Spotify token in memory, login, and hosting plus a tunnel to the GPU. The React app from 24.1 switches its client calls. The pipeline doesn't change. A trigger and the decision to act on it go into PLAN Decisions first.

## 8. Screens

- **24.0:** designed for laptop width. At phone width, columns stack into one with no horizontal scroll (as far as Gradio's layout allows).
- **24.1:** works fully on phones: upload from the camera roll, watch, export.

## 9. Acceptance

E2e tests use **Playwright for Python** (`pytest-playwright`) under a new `e2e` marker in `pyproject.toml`, with synthetic media from `scripts/make_fixtures.py` only (public repo). They select by role, label or `elem_id`, never by Gradio's generated class names. Each stage runs against a local app with the pipeline stubbed, plus one full smoke run on Colab.

24.0:
- e2e: on `/`, Generate is disabled until media and a song are added. Upload fixtures, then upload a song, then Generate: the video appears, Revise and the Director column appear, Keep on a shot, then Recut: a new video appears.
- e2e: Export 1080p downloads an MP4.
- e2e: sample links show only when demo media exists.
- e2e: `/lab` shows the Lab tabs and Advanced; a run id from `/` reopens there.
- e2e: no horizontal scroll at 390 px width.
- `tests/test_app.py` (API contract) passes unchanged.
- Manual: Andrew and one teammate run the flow once on a real album on Colab; a screenshot of before/after goes in the PR (not committed).

24.1:
- Spike result recorded in PLAN Decisions (how `web/dist` is served).
- e2e (Python Playwright against the React page): the same flow as 24.0.
- Popup Spotify sign-in: the code reaches the app; a message from another origin is ignored; with popups blocked, the redirect fallback still works.
- No token in browser storage (the e2e checks localStorage/sessionStorage after connecting).

## 10. Touchpoints with Track 4

- **One-click generate / optional brief:** already done in the pipeline. Step 3 keeps the brief optional.
- **Phase 23 "My montages" gallery:** when it lands, the simple page gets a "My montages" link in the header, and reopening a montage drops the user into the after-cut view. Phase 24 doesn't build the gallery.
- **Story order (phase 17)** and any planner changes need no UI changes.
