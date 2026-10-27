# Phases 7-8: Pipeline, CLI, Gradio app

## 1. Pipeline API (`montage/pipeline.py`), Phase 7

```python
class Session(BaseModel):           # persisted to DATA_ROOT/sessions/<session_id>.json
    session_id: str
    media_paths: list[str]
    song_path: str
    asset_ids: list[str]
    song_hash: str
    plan_history: list[EditPlan] = []
    notes_history: list[str] = []
    pins: list[str] = []            # candidate ids
    excluded: list[str] = []        # candidate ids
    runs: list[str] = []            # run ids, newest last

def analyze(media_paths, song_path, progress=None) -> Session        # ingest, candidates, features, importance, full-song music (all cached)
def make_plan(session, brief) -> EditPlan                           # director (LLM or rules)
def plan_edit(session, plan, seed, system="full") -> EDL            # excerpt, slots, costs, beam/baseline, in/out, EDL
def render_edl(edl, preview=True) -> str                             # path to mp4
def generate(media_paths, song_path, brief, seed=0, preview=True, system="full") -> RunResult
def recut(session_id, note=None, pins=None, excluded=None, seed=None, preview=True) -> RunResult
def export(run_id) -> str                                            # re-render that EDL at 1080p
```
- Every stage wrapped in `log.stage(...)`; `RunResult.timings` has per-stage seconds; `fallbacks` lists every degraded component.
- `progress(fraction, message)` callback is optional (Gradio passes one).
- `generate` writes `plan.json`, `edl.json`, `result.json`, video, `curves.png`, `storyboard.jpg` to the run dir.

**CLI** (`montage/cli.py`):
```
python -m montage.cli generate --media DIR --song FILE --brief "..." [--seed 0] [--export] [--system full]
python -m montage.cli recut --session ID --note "too fast in the middle"
python -m montage.cli export --run RUN_ID
```

**Phase 7 acceptance:** on fixtures, `generate` produces a valid video end to end with every optional component disabled; on one real album (the user's demo album on Drive) a 50 s preview renders in under 3 minutes cold and under 30 s warm (cached features).

## 2. App layout (`montage/app/gradio_app.py`), Phase 8

Gradio Blocks, queue enabled, `concurrency_limit=1` (one GPU).

**Inputs (left column)**
- Media: `gr.File(file_count="multiple")`, or a "Demo album" dropdown listing folders in `DATA_ROOT/demo/` (demo mode, spec 11). If both are given, the dropdown wins.
- Song: `gr.Audio(type="filepath")` or a dropdown of files in `DATA_ROOT/demo/songs/`.
- Director's brief: `gr.Textbox(lines=4)`, placeholder: "Warm, nostalgic summer recap. Open slow on the beach at sunrise, build through the road trip, save the party for the drop, end on the sunset. Skip anything blurry."
- Advanced (accordion): seed, target length (0 = auto, 45-60), excerpt start override (seconds, blank = auto), "Let the director see thumbnails" checkbox (default off), system dropdown (full / naive / beat_random / greedy).
- Buttons: **Generate**, **Export 1080p**.

**Outputs (right column)**
- Video player (preview).
- Status line: "Director: Claude | Emotion: MERT | Importance: learned+heuristic | 41 shots | 118 BPM", showing fallbacks honestly.
- Editor's note (markdown).
- Director's plan: readable bullets (vibe, per-section intent, pacing, avoid, opening/closing) plus raw JSON in an accordion.

**Notes and storyboard (below)**
- Notes: textbox + **Recut** button; list of previous notes.
- Storyboard: `gr.Gallery` of shot thumbnails captioned `0:12  a003 photo`. Selecting a shot shows its `reason` and score breakdown and enables **Pin** / **Exclude** buttons (candidate level). Pins/excludes apply on the next Recut.

**Tabs**
- *Under the hood*: (1) curves plot (`app/plots.py`, matplotlib PNG): intensity, arousal, valence vs excerpt time; beat ticks, bold downbeats, shaded labeled sections, cut lines (forced cuts in a distinct color); (2) per-shot stacked bars of cost components (importance, mood, director, pairwise); (3) stage timings table.
- *Compare*: **Render naive cut** button renders the `naive` baseline for the current session at preview quality and shows it beside the current cut.
- *Rate* (spec 11).

**Behavior**
- Generate: `analyze` (cached) -> `make_plan` -> `plan_edit` -> `render_edl(preview=True)` -> editor's note. Progress shows stage names.
- Recut: reuses the session; only director, planner, and render run.
- Export: re-renders the current EDL at 1080p and offers a download (`gr.File`).
- `UserError` -> `gr.Warning` with the message; other exceptions -> short error plus "see logs", full traceback to the log.

## 3. Launch

`python -m montage.app.gradio_app --share [--port 7860]`
- `demo.queue().launch(share=True, prevent_thread_lock=True)`, write the share URL to `DATA_ROOT/jobs/<user>/app.url` and print it, then `demo.block_thread()`.
- Sessions are stored under `DATA_ROOT/sessions/` with ids prefixed by user, so two people can run their own app instances on the shared Drive without clashing.
- Always started through `scripts/run_job.py --name app` on Colab.

## 4. Phase 8 acceptance

- App starts as a job; `app.url` appears within 60 s.
- Using the Gradio Python client (`gradio_client`) in a test: generate with fixtures returns a video path and a plan; recut with a note returns a new run id.
- Error path: 3 uploaded files shows the friendly minimum-count warning.

## 5. Day 2 additions (Phase 9A)

- Notes go through `director.notes.revise_plan` (LLM, rules fallback).
- Pin/Exclude wired to `recut(pins, excluded)`.
- Editor's note from the LLM.
- Thumbnail toggle sends the contact sheet (spec 05 section 8).
- Phase 9A acceptance: brief -> plan via LLM shows `Director: Claude`; the note "lose the shot at 0:05" removes that asset in the recut; "faster" increases the shot count; pin forces inclusion.
