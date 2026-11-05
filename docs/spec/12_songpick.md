# Phase 12: Song pick from listening history (Track 1, ROADMAP F1)

Goal: "Pick the song for me." The user uploads photos plus their Spotify listening history. The app shortlists 3-5 songs they actually listened to around when the photos were taken, explains each pick, and the user supplies the audio file for the one they choose. Then the existing pipeline runs unchanged (`pipeline.generate` gets a song path as today).

**Status:** 12.1-12.4 (the non-learned fallback) are implemented on branch `phase-12-songpick`. 12.5-12.7 (vibe match, app tab, audio matching) are specified here at outline level and get detailed when the team agrees on the roadmap.

**Why the user supplies the audio:** Spotify cannot provide it. Its SDKs only play inside Spotify's DRM player, preview clips and audio features were removed for new apps in Nov 2024, and its developer terms forbid syncing its content with video or training ML on it. The app therefore only reads the user's own exported history, never calls Spotify, and never downloads audio.

## 1. Inputs and privacy rules

- **Input:** the JSON files from the user's Spotify data export.
  - Extended streaming history: `Streaming_History_Audio_*.json`, older `endsong_*.json`.
  - Account-data history: `StreamingHistory_music_*.json`, `StreamingHistory*.json`.
  - A folder is searched recursively.
- **Personal data.** The history must never be committed and track lists must never be logged.
  - Logs carry counts only.
  - The CLI prints the shortlist to the user's own terminal and writes nothing.
  - `tests/test_repo_hygiene.py` fails on any tracked file named `Streaming_History*.json`, `StreamingHistory*.json` or `endsong*.json`, and `.gitignore` ignores them.
  - Tests use only the synthetic history from `scripts/fixtures_history.py` (made-up artists).
- **Fields kept:** only the fields of `ListeningEvent`. IP address, user agent, country, platform and the other export fields are never read.
- **No network in the fallback.** The vibe stage (12.5) may send track and artist names, and nothing else, to a tag service. That is opt-in by config and is disclosed in the UI.

## 2. Parsing (`montage/songpick/history.py`) — done

- **Extended format, one row per play:**
  - `ts` (ISO UTC, end of play) becomes a naive UTC datetime.
  - `ms_played`.
  - `master_metadata_track_name`, `master_metadata_album_artist_name`, `master_metadata_album_album_name`.
  - `spotify_track_uri`.
  - `skipped = skipped or reason_end == "fwdbtn"`.
- **Account format:** `endTime` ("YYYY-MM-DD HH:MM", UTC), `artistName`, `trackName`, `msPlayed`. These rows have no uri and no skip flag.
- **Dropped rows:**
  - podcast episodes and audiobooks (episode fields set, or no track name);
  - rows without an artist, a parseable time, or a valid `ms_played`.
- **Merging:** rows from all files are merged, sorted by `(ts, artist, track)`, and exact duplicates are dropped. A duplicate is the same `(ts, ms_played, track, artist)`; this happens when the same file is given twice.
- **Errors:** no files, unreadable JSON, a non-list file, or zero music plays raise `UserError` with a friendly message.

## 3. Album period (`montage/songpick/period.py`) — done

- **Dates:** the capture dates are `Asset.taken_at` (naive local wall clock; ingest falls back to file time).
  - The earliest and latest `songpick.period_trim_frac` of the dates are ignored. Default 5%, so one stray old photo does not stretch the period over years.
  - The remaining span is padded by `pad_days` on both sides (default 7), then stretched symmetrically to at least `min_window_days` (default 3).
- **Time zones:** capture dates are local time and history times are UTC. They are compared as is; the error (at most about a day) is far smaller than the padding.
- **Widening** (`pick.choose_period`): if fewer than `min_window_events` plays fall inside the period, the padding is multiplied by `widen_factor` until it exceeds `max_pad_days`. The source is then `album_widened`, with a note.
  - If that still finds too few plays, or no item has a date, the whole history is used (`history`, with a note).

## 4. Period affinity (`montage/songpick/affinity.py`) — done

- **Track key:** `"artist|track"`, lowercased with whitespace collapsed. URIs are not used for the key, because the same song has different URIs on a single and on an album.
- **Weight per play:** `min(ms_played / full_play_ms, 1)` for plays of at least `min_play_ms` (else 0), minus `skip_penalty` if the play was skipped.
- **Per track:**
  - `plays_window` = sum of weights inside the period; `plays_total` = sum over the whole history.
  - `n_plays_window` counts plays of at least `min_play_ms`; `n_skips_window` counts skipped plays.
  - `expected = max(plays_total, 0) * T_win / T_all`, where `T_win` and `T_all` are the window and history totals over all tracks.
  - `lift = (plays_window + lift_prior) / (expected + lift_prior)`. Above 1 means "more than usual for you"; the prior keeps a single play from producing a huge lift.
  - `affinity = plays_window * lift ** sig_weight`, max-normalized to [0, 1].
- **Eligibility:** tracks with fewer than `min_window_plays` real plays, or a non-positive `plays_window`, are dropped. If that leaves nothing, single plays are accepted (note). If there are still none, the whole history is used.
- **Whole-history mode:** every play counts, with recency decay `0.5 ** (age_days / recency_half_life_days)` measured from the latest play (deterministic, no clock). Skip penalties are not decayed. `lift = 1`.
- **Reason line, shown in the UI:**
  - "You played it N times between <start> and <end>[, X.Yx more than usual for you][; skipped Kx]."
  - Whole-history mode: "One of your most played songs (N times, most recently <date>)."

## 5. Shortlist (`montage/songpick/pick.py`) — done

- `shortlist(events, taken_at, k=None) -> SongPick`, with `k` defaulting to `shortlist_size` (5).
- `score = alpha * affinity + (1 - alpha) * vibe` once a vibe stage fills `TrackCandidate.vibe`. Until then `vibe is None`, `score == affinity`, and `vibe_source == "none"`.
- Sorted by `(-score, -n_plays_window, key)`, so the result is deterministic for a given history.
- Usage from code:
  ```python
  from montage.songpick import load_history, shortlist
  pick = shortlist(load_history([history_dir]), [a.taken_at for a in assets])
  ```
- CLI:
  ```
  python -m montage.songpick --history DIR --media ALBUM_DIR [--k 5] [--json]
  python -m montage.songpick --history DIR --start 2026-07-01 --end 2026-07-14
  ```
- A track's duration is not in the history, so "too short for a 45 s excerpt" is checked only once the audio file exists (12.7).

## 6. Later in this phase (outline; detail when the roadmap is agreed)

- **12.5 Vibe match, stage A (metadata only):**
  - track/artist -> tags (Last.fm or MusicBrainz, opt-in, cached per track key) -> tag text -> CLIP text embedding.
  - Cosine similarity to the album's pooled image embedding, plus the distance between the album's mood valence/arousal (`features/mood.py`) and valence/arousal predicted from the tags.
  - Fills `vibe` and sets `vibe_source = "tags"`. With no network or no tags, `vibe` stays None.
- **12.6 App tab "Pick a song for me":** upload history -> shortlist with reasons -> choose -> upload or match the audio -> the existing Generate flow. The history is held under the session's DATA_ROOT folder only.
- **12.7 Local library matching:** match the chosen track to a file in a user-given music folder using ID3 tags (`mutagen`) plus fuzzy matching on names (`rapidfuzz`). Check the audio's duration against the excerpt length.
- **Stage B (contrastive image-set <-> music model)** is its own phase (ROADMAP F1 step 5) and gets its own spec.
- **Long-term change (ROADMAP F1 steps 2 and 6):** the play-count ranking in sections 3-5 is temporary.
  - **Window:** the period becomes an age-scaled window centered on the album's median capture date: about a month for photos from a year ago, a season for photos from five years ago (config table `songpick.window_by_age_days`).
  - **Ranking:** songs are ranked by sentiment match: album valence/arousal vs the song's valence/arousal from tags, or from the MERT head once audio exists.
  - **Affinity:** period affinity only decides eligibility and breaks ties.
  - **Code impact:** `choose_period` and the `score` formula change; the `SongPick`/`TrackCandidate` fields stay the same (`vibe` carries the sentiment match).

## 7. Config (`songpick` in `configs/default.yaml`)

`min_play_ms 30000`, `full_play_ms 180000`, `skip_penalty 0.5`, `period_trim_frac 0.05`, `pad_days 7`, `min_window_days 3`, `min_window_events 20`, `widen_factor 2.0`, `max_pad_days 60`, `recency_half_life_days 365`, `lift_prior 1.0`, `sig_weight 0.5`, `min_window_plays 2`, `shortlist_size 5`, `alpha 1.0`.

## 8. Schemas (`montage/types.py`, `docs/spec/schemas.md`)

New models: `ListeningEvent`, `TrackCandidate`, `SongPick` (see schemas.md, section "Song pick"). No existing model changes.

## 9. Acceptance tests (`tests/test_songpick.py`, no marks, run locally)

- **Parser:**
  - extended and account formats parse;
  - podcasts and bad rows are dropped;
  - `fwdbtn` counts as a skip;
  - IP and other export fields never reach the event;
  - a folder of split files merges without double counting;
  - unreadable or empty input raises `UserError`.
- **Period:**
  - a stray old photo is trimmed;
  - padding and the minimum length are applied;
  - no dates gives `None`.
- **Affinity:** a song played only during the period outranks a song played just as often in the period but all year round (lift > 1 vs < 1).
- **Shortlist, on the synthetic history** (album dates July 3-10, 2026):
  - "Summer Anthem" is first, with lift > 1.5 and "more than usual" in its reason;
  - the always-skipped song is never shortlisted;
  - the result is deterministic;
  - with no dates the all-time favourite wins and the period is `history`;
  - an album just before the history starts widens the period;
  - an album years before it falls back to the whole history;
  - the fallback works with no network and no weights.
- **CLI:** prints the shortlist; exits 2 with a friendly message on a missing history.
- **Hygiene:** `test_repo_hygiene.py` flags tracked `Streaming_History*.json` and `StreamingHistory*.json` files.
- **Later (12.6/12.7):** a chosen pick hands a valid song path to `pipeline.generate`.
