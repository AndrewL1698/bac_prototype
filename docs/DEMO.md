# DEMO: 3-minute run-of-show

Presenter: Keshav. Everything runs in the Gradio app on the Colab A100; the backups on Drive cover any failure.

## Before the demo (start 15 minutes early)

1. Open the Colab notebook, Runtime -> A100. Run the bootstrap cell **from the Colab UI** so Secrets (API key) load:
   ```
   !git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null
   %run /content/bac_prototype/scripts/colab_bootstrap.py --branch main
   ```
2. Warm everything and render backups (about 3-6 minutes per album):
   ```
   !cd /content/bac_prototype && python scripts/run_job.py --name demo_prep -- python -m montage.app.demo_mode --prepare
   !cd /content/bac_prototype && python scripts/job_status.py demo_prep
   ```
   The last lines print a readiness checklist: GPU, Claude key, emotion/ranker weights, a backup per album, vote pairs, warm generate time (must be under 30 s).
3. Start the app and get the link:
   ```
   !cd /content/bac_prototype && python scripts/run_job.py --name app -- python -m montage.app.gradio_app --share
   !cd /content/bac_prototype && python scripts/job_status.py app
   ```
4. Open the share link on the laptop **and** on a phone. Do one warm Generate with the demo album so the first live one is instant.
5. Laptop plugged in, sleep off, Colab tab open (do not close it), Do Not Disturb on.
6. Have `montage_data/demo/backups/<album>.mp4` and the latest `runs/<run_id>/curves.png` open in a Drive tab as the fallback.

## Run-of-show

**0:00-0:20 The problem and the idea.**
"Everyone has hundreds of photos and clips from a trip; turning them into a video takes hours. Here you are the director: you say what to show and how it should feel, the music sets the timing, and the system does the editing."

**0:20-1:20 Live: direct a cut.**
- In the left column pick **...or a demo album** and **...or a demo song** (or upload).
- Type the brief in **Director's brief**, e.g. "Warm summer recap. Open slow on the beach at sunrise, save the party for the drop, end on the sunset. Skip anything blurry."
- Click **Generate**. While it runs (progress bar), explain in one breath: *it sees* (CLIP embeddings, aesthetics, faces, motion, mood), *it listens* (beats, downbeats, energy, sections), *it plans* (Claude turns the brief into an edit plan), *it cuts on the beat* (a beam search picks and orders shots to fit the plan and the music).

**1:20-1:50 Watch the result.**
- Play the **Preview**. Point at the status line ("Director: Claude | Emotion: ... | N shots | BPM").
- Read the editor's note under it, then the plan bullets (vibe, looking for, avoiding, open on, end on).

**1:50-2:20 Give a note and recut.**
- In **Notes for the editor** type "more of the beach, slower at the start" and click **Recut**. Show that the opening is slower and the notes list updated.
- Optional: click a storyboard tile, read its "Why", click **Exclude shot**, **Recut**.

**2:20-2:40 Under the hood.**
- Tab **Under the hood**: the curves plot (intensity/arousal, beat ticks, bold downbeats, shaded sections, cut lines; section cuts are solid purple). Point at one storyboard tile's reason ("top-ranked moment; matches 'golden hour beach'").

**2:40-3:00 Compare and close.**
- Tab **Compare**: click **Render naive cut** (random shots, fixed 2 s cuts) and play both side by side.
- One metrics slide (from `docs/RESULTS.md`): beat-hit rate and pace coupling vs naive; live vote result if collected (`python -m montage.eval.vote --analyze`).
- Click **Export 1080p** if someone wants the file.

## Audience vote (optional, before or after)

Tab **Rate**: **Load a pair**, watch Left and Right (order randomized), click **Left better**, **Tie**, or **Right better**. Votes go to `montage_data/eval/votes/<user>.csv`; summary: `python -m montage.eval.vote --analyze`.

## If something breaks

| Problem | Do this |
|---|---|
| Share link dead or Colab disconnected | Play `montage_data/demo/backups/<album>.mp4` from Drive; walk through `curves.png`. Restart: rerun bootstrap, then the `app` job (new link). |
| Generate shows an error | Read it out (it is a friendly message), pick the demo album and song, Generate again. |
| Claude API down / no key | Nothing to do: the status line says "Director: rules" and the cut still works. Say so. |
| Slow first run | Talk through the pipeline; demo mode pre-warms caches so this should not happen. |
| Audio or video stutters on the projector | Use **Export 1080p** and play the downloaded file locally. |
