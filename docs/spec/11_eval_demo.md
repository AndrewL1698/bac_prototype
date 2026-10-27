# Phase 11: Evaluation, demo mode, backups, DEMO.md

Goal: evidence that fits on one or two slides, plus a demo that cannot fail on stage.

## 1. Evaluation grid (`eval/run_grid.py`, run as a job)

- Albums: `DATA_ROOT/eval/albums/<name>/` (2 albums minimum). Optional `brief.txt` per album; default brief: "A fun recap of these moments, matched to the music."
- Songs: `DATA_ROOT/eval/songs/` (3 songs of different tempo and mood).
- Systems: `full`, `naive`, `beat_random`, `greedy`, and ablations `no_learned`, `no_mood`, `no_pairwise`, `no_director`, `no_forced` (spec 06 section 7).
- Seeds: 3 per (album, song, system).
- Plans: one plan per (album, song) via the director, shared by all systems so the comparison isolates the editor.
- EDLs only (no rendering) for metrics; it is fast. Save every EDL under `DATA_ROOT/eval/edls/`.

## 2. Metrics (`eval/metrics.py`), computed from EDL + MusicProfile + features

Cut times = internal cut times (exclude 0 and L).
- **beat_hit_rate**: fraction of cuts within +/-70 ms of a detected beat (mir_eval's default tolerance). **downbeat_hit_rate**: same for downbeats. **mean_offset_ms**: mean absolute distance to the nearest beat.
- **boundary_hit_rate**: fraction of section boundaries with a cut within +/- one beat.
- **pace_coupling**: Spearman between local cut rate (cuts per second in a 4 s window, sampled on the 2 Hz grid) and the intensity curve.
- **mood_coupling**: Spearman between each shot's visual arousal (batch z) and its slot's music arousal.
- **diversity**: mean pairwise CLIP cosine among chosen shots (lower = more varied); **max_adjacent_cos**; **dup_violations** (pairs sharing a dup cluster); **coverage** (unique assets used / assets uploaded); **same_source_adjacent** count.
- **quality_lift**: mean `importance.final` of chosen shots minus the pool mean.
- **director_adherence**: mean director score D of chosen shots minus the pool mean (when the plan has queries).

Note in the report that beat-hit rate for `full` is high by construction; the point of that metric is the contrast with `naive`.

## 3. Render check (`eval/render_check.py`)

For `full`, seed 0, every (album, song): render a preview, detect cuts with PySceneDetect `ContentDetector`, match detected cuts to EDL frame boundaries within +/-1 frame. Report precision (detected cuts that match an EDL cut) and recall (EDL cuts detected; visually similar adjacent shots can be missed, so recall below 100% is expected). The goal is precision ~1.0: no cut lands where the EDL did not put one, which proves no drift.

## 4. Report (`eval/report.py`)

`DATA_ROOT/eval/report.md` with:
- Table: each metric, mean +/- std across (album, song, seed) per system.
- Model results pulled from `eval/emotion/metrics.json` and `eval/importance/metrics.json` (if present).
- Render check summary.
- Figures in `DATA_ROOT/eval/figures/` sized for slides (16:9, large fonts): (1) beat_hit_rate and pace_coupling by system; (2) diversity and quality_lift by system; (3) ablation deltas vs full.
- 3-5 plain-language takeaways at the top. Honest: if an ablation does not hurt, say so.

Copy the summary tables and takeaways (text only) into `docs/RESULTS.md` in the repo, with the date, author, git commit, and the Drive path of the full report and figures. Append a `docs/LOG.md` entry with the headline numbers.

## 5. Live vote (`eval/vote.py` + the app's Rate tab)

- The Rate tab shows two unlabeled videos (current cut vs naive or beat_random, pre-rendered for the demo album), left/right order randomized per rater.
- Buttons: Left better / Right better / Tie, optional rater name. Appends to `DATA_ROOT/eval/votes/<user>.csv` (timestamp, rater, pair_id, left_system, right_system, choice); per-user files avoid concurrent writes on the shared Drive.
- `python -m montage.eval.vote --analyze` (reads all `votes/*.csv`): win rate of `full` excluding ties, two-sided exact binomial test (`scipy.stats.binomtest`, p=0.5), Wilson 95% interval. Prints a one-line summary for a slide.

## 6. Demo mode (`app/demo_mode.py`)

`python -m montage.app.demo_mode --prepare` (job):
1. Analyze every album in `DATA_ROOT/demo/` and every song in `demo/songs/` (fills all caches; also runs MERT on each song).
2. For each album (with its `brief.txt` or the default), generate and **export** a backup video to `DATA_ROOT/demo/backups/<album>.mp4`, plus a naive-cut backup.
3. Pre-render the vote pair for the Rate tab.
4. Time a warm `generate` (preview); must be under 30 s. Print a readiness checklist (GPU, API key, weights present, backups present, app URL).

## 7. DEMO.md (`docs/DEMO.md`)

Write a 3-minute run-of-show with exact clicks and talking points:
1. (0:20) The problem: hours of footage, minutes of editing; the idea: you direct, it edits to the music.
2. (1:00) Live: pick the demo album and song, type the brief, Generate. While it runs, explain the pipeline in one breath (sees, listens, plans, cuts on the beat).
3. (0:30) Watch the result; read the editor's note and the director's plan.
4. (0:30) Give a note ("more of the beach, slower at the start"), Recut, show the change.
5. (0:20) Under the hood: curves with cut marks; why-this-shot on one storyboard tile.
6. (0:20) Compare with the naive cut; one metrics slide.
Plus: pre-demo checklist (start the app 15 min early, run `--prepare`, open the share link on a phone, laptop plugged in and awake, Colab tab open) and a fallback plan (if Colab or the link dies, play the backups from Drive and walk through the curves PNG).

## Acceptance tests

- `run_grid` finishes as a job on 2 albums x 3 songs x all systems x 3 seeds and writes `report.md` and figures.
- `test_metrics_known`: a hand-built EDL with cuts exactly on beats gives beat_hit_rate 1.0; cuts shifted by 200 ms give 0.0.
- `test_vote_analysis`: synthetic votes (15 wins, 5 losses, 2 ties) give win rate 0.75 and the correct binomial p-value.
- `demo_mode --prepare` completes; backups exist; warm generate under 30 s.
- `docs/DEMO.md` exists and matches the app's actual buttons.
- `docs/RESULTS.md` filled for every phase that ran; README "What works today" reflects the final state; final `docs/LOG.md` entry says what is demo-ready and where the backups are.
