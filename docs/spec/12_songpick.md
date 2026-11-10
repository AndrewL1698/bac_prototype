# Phase 12: Song pick from listening history (Track 1, ROADMAP F1)

Goal: "Pick the song for me."
1. The user presses **Connect Spotify**. This is read-only: recent plays and top songs, nothing else.
2. The user adds photos, chooses how to pick (three modes, section 6), presses **Suggest songs**, and gets 3-5 songs from their own listening, each with a reason:
   - **Songs from back then** (default): songs they listened to in the past year that came out around when the photos were taken;
   - **Best vibe fit**: dates ignored (ranked by listening until the vibe match exists);
   - **My full Spotify history**: songs they actually played back then, from the uploaded data export.
3. The user clicks **Use this song**. With the YouTube opt-in ticked, the app fetches the audio automatically; otherwise, or when YouTube blocks the fetch, it links to a YouTube search so the user can upload the file.
4. The existing pipeline runs unchanged: `pipeline.generate` gets a song path, as today.

**Status:** all of it is implemented on branch `phase-12-songpick`:
- 12.1-12.4: data-export parsing, album period, period affinity, shortlist;
- 12.5: Spotify sign-in;
- 12.6: YouTube audio (opt-in);
- 12.7: the app UI;
- 12.7b (2026-11-08): the three modes, release-date windows, YouTube cookies + manual fallback.

Still to do: the vibe/sentiment match (12.8, the next PR) and local library matching (optional).

**Where the audio comes from:** Spotify cannot provide it. Its SDKs only play inside Spotify's DRM player, preview clips and audio features were removed for new apps in Nov 2024, and its developer terms forbid syncing its content with video. The app uses Spotify only for listening *data*.
- **YouTube (opt-in):** the audio is fetched from YouTube with `yt-dlp`. That breaks YouTube's terms, and the songs are copyrighted. Andrew accepted that risk for the class demo (PLAN Decisions, 2026-11-05). It is off by default and runs only after the user ticks the opt-in box and clicks **Use this song**.
- **Upload:** the user can always upload the song instead.

**Spotify Development Mode limits** (since Feb/Mar 2026):
- the app owner needs Spotify Premium;
- at most 5 allowlisted users per Client ID;
- a reduced set of endpoints.

Sign-in exposes only the last 50 plays (with dates) and top tracks for about 4 weeks, 6 months and 1 year (ranks, no dates). For photos older than about a year, the full **data export** (sections 1-5) stays as an optional upload.

## 1. Inputs and privacy rules

- **Input (optional; sign-in is the default, section 6):** the JSON files from the user's Spotify data export.
  - Extended streaming history: `Streaming_History_Audio_*.json`, older `endsong_*.json`.
  - Account-data history: `StreamingHistory_music_*.json`, `StreamingHistory*.json`.
  - A folder is searched recursively.
- **Personal data.** The history must never be committed and track lists must never be logged.
  - Logs carry counts only.
  - The CLI prints the shortlist to the user's own terminal and writes nothing.
  - `tests/test_repo_hygiene.py` fails on any tracked file named `Streaming_History*.json`, `StreamingHistory*.json` or `endsong*.json`, and `.gitignore` ignores them.
  - Tests use only the synthetic history from `scripts/fixtures_history.py` (made-up artists).
- **Fields kept:** only the fields of `ListeningEvent`. IP address, user agent, country, platform and the other export fields are never read.
- **What leaves the machine:**
  - the export path (sections 2-5) uses no network;
  - Spotify sign-in calls Spotify's API with the user's token;
  - the YouTube fetch sends only the search text "<artist> - <track>";
  - a future tag-based vibe stage would send only track and artist names.
  - No photos, audio or other listening data are sent anywhere.

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
- **Recent plays** (sign-in) count inside this padded period (`pad_days`).
- **Export listening window** (`period.listening_window`, mode "export"; 2026-11-08): centered on the album's median capture date (`median_date`, trimmed). Its total width grows with the photos' age (`window_by_age_days`, piecewise linear: 21 days for new photos, 30 for a year old, 90 for 5 years, 180 for 10+). It always covers the album span (at least `min_window_days`). Age is counted in whole days, so a pick does not change within a day.
- **Widening** (`pick.choose_period`): if fewer than `min_window_events` plays fall inside the window, its width is multiplied by `widen_factor` until it reaches `max_window_days` (365). The source is then `album_widened`, with a note.
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

## 6. Sign-in, audio and app (12.5-12.7) — done

### 12.5 Spotify sign-in (`montage/songpick/spotify.py`)
- **OAuth flow:** Authorization Code with **PKCE** (RFC 7636, S256), so there is no client secret.
  - The Client ID comes from the Colab Secret `SPOTIFY_CLIENT_ID`; `scripts/colab_bootstrap.py` copies it to the env file, like `APP_PASSWORD`.
  - Scopes: `user-read-recently-played user-top-read` only.
- **Redirect:** the app's share URL changes on every restart, so the registered redirect URI is a fixed static page, `docs/spotify/callback.html`, served by GitHub Pages.
  - `state` is `<nonce>~<base64url(app URL)>`, and the page sends the browser back to `<app URL>?spotify_code=...&state=...`.
  - It only redirects to hosts allowed by `songpick.spotify.return_hosts`: `https://*.gradio.live`, or a loopback host. Python mirrors this rule in `safe_return_url`, which is tested.
  - Otherwise the page shows `<code> <state>` to paste into the app.
- **Server side:** `begin_auth(app_url)` keeps `{state: verifier}` in process memory for `auth_ttl_s`. `finish_auth(code, state)` exchanges the code once; a reused or expired state gives a friendly `UserError`.
- **Tokens and listening data** live only in the user's `gr.State`: never written to disk, never logged (counts only).
- **Data:**
  - `recently_played` -> `ListeningEvent(source="recent")`. A listed play counts as a full play.
  - `top_tracks(short_term|medium_term|long_term)` -> `TopTrack(rank)`.
  - HTTP 401, 403 (not allowlisted in Development Mode), 429 and other errors become friendly `UserError`s.

### Ranking: three modes (`pick.suggest(taken_at, mode=...)`; 2026-11-08)
- **Sources:**
  - **export:** sections 3-5 (in modes release/vibe: the whole history with the recency decay, to enlarge the pool).
  - **recent:** play affinity inside the padded album period.
  - **top:** every available list (`spotify.range_days` 28/182/365, all ending now). Value: `(L + 1 - rank) / L * range_weight` (1.0 / 0.7 / 0.5), the best over the lists.
- **Combining:** each source is max-normalized, weighted by `w_sources`, summed per track key, then re-normalized. Reasons from every source are kept, and `sources` lists them. Release fields: the first reliable date wins.
- **Mode `release`** ("Songs from back then", the default `default_mode`). Sign-in only reaches about a year back, so "from that time" uses the song's **album release date** (`release.py`):
  - Spotify's `album.release_date` at year/month/day precision; a less precise date stands for its whole year/month.
  - **Remaster heuristic** (instead of a MusicBrainz first-release lookup, PLAN 2026-11-08): compilations (`album_type`) and album/track names with `release.unreliable_words` (remaster, deluxe, anniversary, greatest hits, best of, collection, edition, expanded) are skipped, with a note giving the count.
  - **Release window** (`period.release_window`): from `median - before` to `median + after_frac * before`, asymmetric because songs are mostly heard after they come out. `before` scales with photo age (`release.before_days_by_age`: 60 days for new photos, 90 for a year old, 180 for 5 years, 365 for 10+). It always covers the album span.
  - **Eligible:** reliable release date inside the window, **or** played during the album: recent plays inside the padded album period, or the 4-week top list when the whole padded album falls inside those 4 weeks.
  - **Widening:** `before` is multiplied by `widen_factor` until `release.min_candidates` (5) songs qualify (or all that can), up to `release.max_before_days` (1095). The source is `release_widened`, with a note.
  - **Ranking:** listening affinity re-normalized over the eligible songs; ties go to the release closest to the median. The reason starts "Came out <Mon YYYY>, near your photos (<Mon YYYY>)."
  - **Fallbacks:** no eligible song at all -> mode `vibe` with the note "None of the songs you played this year came out around <Mon YYYY>...". No capture dates -> `vibe` with a note.
- **Mode `vibe`** ("Best vibe fit"): dates ignored; for now the merged pool ranked by listening affinity, with a note that vibe matching is coming. 12.8 plugs in through `TrackCandidate.vibe` and `alpha`.
- **Mode `export`** ("My full Spotify history"): requires the export (else `UserError`). Export affinity inside the listening window (section 3), plus recent plays inside it; no top lists.
- `SongPick.mode` reports the mode actually used (a release pick that fell back says `vibe`).

### 12.6 Audio from YouTube (`montage/songpick/youtube.py`, opt-in)
- **Search:** `search("<first artist> - <track>", max_results)` reads metadata only. Only this text leaves the machine.
- **`pick_best`** (pure, tested) scores each result:
  - track/artist similarity (difflib);
  - `+topic_bonus` for "<Artist> - Topic" channels (official audio);
  - `-bad_penalty` for live, cover, remix, karaoke, sped up, slowed, nightcore, 8D, instrumental or reverb, unless the word is in the track name;
  - `-duration_penalty` when the length is more than `duration_tol_s` from Spotify's `duration_ms`.
  - It rejects ids that are not 11 characters, lengths outside `min_song_s..max_song_s`, and scores below `min_match`.
- **Download:** `fetch_audio(video_id)` downloads best audio as m4a into `DATA_ROOT/cache/youtube_audio/<id>.m4a`, written atomically and downloaded once. That is the private Drive; media is gitignored and hygiene-blocked.
- **Colab is blocked by default.** YouTube flags cloud IPs ("Sign in to confirm you're not a bot"), and yt-dlp needs a JavaScript runtime for YouTube's challenges. So (2026-11-08):
  - `deno` (pinned) is in `requirements.txt`; `yt-dlp[default]` brings `yt-dlp-ejs`. `yt-dlp` has a lower bound rather than a pin, because YouTube changes break old releases.
  - **Cookies** (optional): the Colab Secret `YOUTUBE_COOKIES` holds a cookies.txt export from a **spare** Google account, as is or base64. `colab_bootstrap.py` writes it to `/content/.yt_cookies.txt` (local VM disk, mode 600; never Drive, never the repo, never printed) and exports `MONTAGE_YT_COOKIES`. `youtube.cookies_path()` reads env `youtube.cookies_env`, else `youtube.cookies_file`.
  - **Risk** (explained to Andrew, PLAN 2026-11-08): the cookies work as that account's full Google login; YouTube can flag or block the account; the cookies rotate and must be re-exported now and then.
- **Failures** become a `UserError`: without cookies it suggests adding the secret; with cookies it says they may have expired. The app then shows a YouTube search link (`search_url`) and asks the user to upload the file.

### 12.7 App (`montage/app/songpick_ui.py`, wired in `gradio_app.py`)
- **Where:** a "Pick a song for me (Spotify)" accordion above the Song input, containing:
  - Connect Spotify, which reloads the page through Spotify;
  - Suggest songs, which reads the capture dates of the chosen album or uploads with `ingest_with_notes`;
  - an optional upload of the history export;
  - the suggestions as a radio list with reasons;
  - a **How to pick** radio (the three modes; default `songpick.default_mode`); the export upload shows only in the export mode;
  - the YouTube opt-in checkbox (default `youtube.enabled: false`) and **Use this song**, which fills the Song player, or (fetch off or blocked) leaves it as it is and shows a YouTube search link plus "drop the file in the Song box";
  - a paste box for the callback page's fallback code.
- **Serving:** the `cache/youtube_audio` folder is never served (it is in `blocked_paths`; anyone with the share link could fetch songs by video id). `use_song` returns a copy in the system temp dir (`youtube.serve_copy`), which Gradio moves into its own content-hashed cache.

### Later
- **12.8 Vibe/sentiment match (ROADMAP F1 pieces 4-5), the next PR:**
  - stage A: tags -> CLIP text embedding vs the album's pooled CLIP embedding, plus album vs tag valence/arousal;
  - ranking by sentiment match: album valence/arousal vs the song's valence/arousal from tags, or from the MERT head on the fetched audio.
- **MusicBrainz first-release dates** instead of the remaster heuristic (optional; 1 request/s, names only).
  - `vibe` carries it; affinity becomes eligibility and the tie-break.
- **Optional:** local library matching (ID3 tags via `mutagen`, fuzzy names), for users who keep music files.
- **Stage B** (contrastive image-set <-> music model) is its own phase.

## Human setup (once)
1. On developer.spotify.com (Premium account), create an app.
   - Redirect URI: `https://andrewl1698.github.io/bac_prototype/spotify/callback.html`.
   - Add up to 5 users under User Management.
2. Enable GitHub Pages for the repo from `main` `/docs`. Until the branch merges, use branch `phase-12-songpick` `/docs`.
3. Add the Colab Secret `SPOTIFY_CLIENT_ID`, then rerun the start cell.
4. Optional, for the YouTube fetch on Colab: a **spare** Google account (never a personal one), signed in to YouTube in a private/incognito window. Export youtube.com cookies with a cookies.txt extension, close the window (using the account afterwards rotates the cookies), and paste the file's base64 (`base64 -i cookies.txt`) into the Colab Secret `YOUTUBE_COOKIES`. The bootstrap line then says `YouTube cookies yes`.

## 7. Config (`songpick` in `configs/default.yaml`)

`min_play_ms 30000`, `full_play_ms 180000`, `skip_penalty 0.5`, `period_trim_frac 0.05`, `pad_days 7`, `min_window_days 3`, `min_window_events 20`, `widen_factor 2.0`, `recency_half_life_days 365`, `lift_prior 1.0`, `sig_weight 0.5`, `min_window_plays 2`, `shortlist_size 5`, `alpha 1.0`, `w_sources {export, recent, top: 1.0}`, `default_mode release`, `window_by_age_days [[0,21],[1,30],[5,90],[10,180]]`, `max_window_days 365` (replaces `max_pad_days`).

`songpick.release`: `before_days_by_age [[0,60],[1,90],[5,180],[10,365]]`, `after_frac 0.25`, `min_candidates 5`, `max_before_days 1095`, `unreliable_words`.

`songpick.spotify`: `client_id_env`, `redirect_uri`, `scopes`, `return_hosts`, `auth_ttl_s 600`, `timeout_s 10`, `top_limit 50`, `range_days {28, 182, 365}`, `range_weight {1.0, 0.7, 0.5}`.

`songpick.youtube`: `enabled false`, `max_results 6`, `duration_tol_s 10`, `min_song_s 60`, `max_song_s 900`, `min_match 0.55`, `topic_bonus 0.25`, `bad_words`, `bad_penalty 0.4`, `duration_penalty 0.4`, `cookies_file`, `cookies_env MONTAGE_YT_COOKIES`.

## 8. Schemas (`montage/types.py`, `docs/spec/schemas.md`)

New models: `ListeningEvent` (with `source`, `duration_ms`), `TopTrack`, `TrackCandidate` (with `duration_ms`, `sources`), `SongPick` (with `sources`). See schemas.md, section "Song pick". No existing model changes.
2026-11-08: `ListeningEvent`, `TopTrack`, `TrackCandidate` gain `release_date`, `release_precision`, `release_reliable`; `SongPick` gains `mode`, and `period_source` gains `release` / `release_widened`.

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
- **`tests/test_songpick_connect.py`** (no network):
  - Spotify sign-in: the PKCE RFC 7636 test vector; return-URL rule cases; one-use and expiring state; no client secret sent; friendly 401/403/429.
  - API parsing and fetch.
  - Release-date parsing (fixtures carry release dates, one remastered album); mode `release` for recent photos (played-then songs win), old photos (a 2021 song, not this month's #1; widening), the remaster skip and the fallback to `vibe`; mode `vibe` ignores dates; mode `export` needs the export; photos without dates; determinism.
  - YouTube cookies from env, friendly messages with and without cookies, the search link; `use_song` returns the manual route when the fetch is off or blocked.
- **`tests/test_songpick_modes.py`:** release parsing at three precisions and bad dates; the remaster flag (whole words only); `age_scaled_days`; `median_date`; `release_window` asymmetry, age scaling and album cover; `listening_window` widths at ages 0/1/5/10; the bootstrap's `YOUTUBE_COOKIES` formats (raw, base64, wrapped base64, junk) and mode 600; the start_app re-run guard.
- **Hygiene:** also flags tracked `*cookies*.txt`.
  - YouTube matching: Topic preferred; live/cover penalized; bad ids, lengths and weak matches rejected; the cache downloads once.
  - The app handlers.
- **On Colab:** `@pytest.mark.network` live YouTube search, plus a manual end-to-end run (connect, suggest, use, Generate, export).
