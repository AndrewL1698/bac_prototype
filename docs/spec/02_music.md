# Phase 2: Music analysis

Goal: from one song file, produce beats, downbeats, energy curves, sections, an emotion curve, an intensity curve, and a chosen 45-60 s excerpt, packaged as `MusicProfile`.

Two layers:
- **Full-song analysis** (cached by song hash): beats, downbeats, curves on a 2 Hz grid, section boundaries.
- **Excerpt profile** (cheap, per plan): pick the window, shift everything to the excerpt timeline.

## 1. Loading (`music/load.py`)

- Decode with librosa (via soundfile/audioread; FFmpeg fallback: decode to 44.1 kHz WAV in scratch first for mp3/m4a/aac).
- Keep `y22` (22,050 Hz mono) for DSP and `y24` (24,000 Hz mono) for MERT.
- Song hash: sha1 of file bytes (16 hex).
- Reject songs shorter than `music.min_song_s` (40 s) with `UserError`.

## 2. Beats and downbeats (`music/beats.py`)

Primary: Beat This! (`from beat_this.inference import File2Beats`, checkpoint `final0`, device cuda) returns beat and downbeat times in seconds.

Fallback (if import or inference fails): `librosa.beat.beat_track(y=y22, sr=22050, units="time", trim=False)`; downbeats by phase: compute onset strength at each beat, for phase p in 0..3 sum strength of beats with index % 4 == p (use low-frequency RMS, below 200 Hz, for more weight on kicks), choose the max phase.

Tempo = 60 / median inter-beat interval.

Beat mode:
- `tempo > music.max_bpm` (default 160): **half**, keep every other beat (phase aligned to downbeats).
- `tempo < music.min_bpm` (default 70): **double**, insert midpoints.
- else **normal**. Downbeats are unchanged in all modes.

Record `beat_source`.

## 3. Energy curves (`music/energy.py`)

On `y22`, hop 512:
- `rms_db` = 20*log10(RMS + 1e-6)
- `onset_env` = `librosa.onset.onset_strength`; `onset_density` = onsets per second in a sliding 2 s window (from `librosa.onset.onset_detect`)
- `centroid` = spectral centroid

Resample all to the **2 Hz grid** (`grid_times = 0, 0.5, 1.0, ...` song time) by averaging frames in each 0.5 s bin.

## 4. Structure (`music/structure.py`)

1. Beat-synchronous features: 20 MFCCs + 12 chroma (CQT), averaged between consecutive beats; z-score each dimension.
2. Self-similarity matrix S (cosine similarity, beats x beats).
3. Foote novelty: slide a Gaussian-tapered checkerboard kernel of half-width `music.structure.kernel_beats` (16) along the diagonal.
4. Peak-pick novelty: min distance `music.structure.min_section_beats` (16), threshold mean + `music.structure.peak_k` (0.5) * std.
5. Snap each boundary to the nearest downbeat. Drop boundaries within 2 bars of the song start/end.
6. Interpolate novelty to the 2 Hz grid (normalized 0-1) for the profile.

## 5. Emotion (`music/emotion.py`)

Interface: `predict_emotion(y24, grid_times) -> (arousal, valence, source)`, arrays aligned to the grid, values in [-1, 1].

**MERT path** (when `music.emotion.use_mert` and `DATA_ROOT/weights/emotion_head.pt` exists):
- For each grid time t, take a window of `music.emotion.window_s` (5.0) **centered** at t (zero-pad at edges). This alignment must match training (spec 09).
- Batch windows (e.g. 64), fp16 on CUDA, `output_hidden_states=True`, stack 13 layers, mean over time -> (13, 768) per window; apply the trained head (layer weights + MLP) -> (valence, arousal); clip to [-1, 1].
- Efficiency: MERT runs on full 5 s windows; a 60 s excerpt is ~120 windows, a full song ~420. Run on the full song once and cache by song hash.

**Heuristic path** (fallback):
- `arousal = tanh(0.5*z(rms_db) + 0.3*z(onset_density) + 0.2*z(centroid))`
- `valence = 0` everywhere (neutral; we do not guess valence without a model).
- `source = "heuristic"`.

## 6. Intensity (`music/profile.py`)

`intensity = gauss_smooth( w_a*z(arousal) + w_r*z(rms_db) + w_o*z(onset_density), sigma = music.intensity.smooth_s )`
Defaults `w_a=0.5, w_r=0.3, w_o=0.2, smooth_s=1.5`. If emotion source is heuristic, arousal already contains rms/onsets; still fine.
Then rescale to unit std. Also smooth arousal and valence with sigma 1.0 s.

## 7. Sections and labels

From boundaries, sections cover the song. For each: mean intensity, slope (linear fit of intensity over time).
- `peak`: the section containing the global intensity maximum.
- `building`: slope > `music.sections.slope_thr` (0.05 per s).
- `falling`: slope < -slope_thr.
- otherwise `high` if mean intensity >= song median, else `low`.

## 8. Excerpt selection (`music/excerpt.py`)

Inputs: full analysis, `target_length_s` (plan or `music.excerpt.default_length_s` = 50), `excerpt_hint`, `excerpt_start_s`.

1. Bar duration = median downbeat spacing. Length L = round(target / bar) bars, adjusted so L is within [`min_length_s` 45, `max_length_s` 60]. If the song is shorter than L + 2 s, use the longest valid whole-bar length.
2. Candidate starts: every downbeat s with s + L <= song end.
3. Score each start (all terms on z-scored intensity):
   - `mean_intensity` over [s, s+L]
   - `peak_placement`: position p of the max intensity inside the window as a fraction; score = exp(-((p - 0.7) / 0.15)^2)
   - `start_boundary`: 1 if a section boundary is within 1 bar of s
   - `end_quality`: 1 if a section boundary is within 1 bar of s+L, or intensity at the end is falling
   - `leading_silence` penalty: if RMS over the first 2 s is below `music.excerpt.silence_db` (-40 dB)
   Score = `w_mean*mean + w_peak*peak_placement + w_start*start_boundary + w_end*end_quality - w_sil*silence` (weights in config).
4. Hints: `start` = first downbeat after leading silence; `peak` = argmax of peak_placement only; `custom` = downbeat nearest `excerpt_start_s`; `auto` = argmax of the score.
5. `fade_out = music.excerpt.fade_out_s` (1.5).

Then `build_profile(full, excerpt) -> MusicProfile`: shift beats, downbeats, grid curves, sections, and peak to the excerpt timeline [0, L]; keep only items inside.

## 9. Song summary for the director (`music/profile.py: summarize(profile) -> str`)

Plain text, one line per section, e.g.:
```
Song: 120 BPM, excerpt 0:48 (starts at song 1:02). Peak at 0:34.
S0 0.0-8.0s low
S1 8.0-24.0s building
S2 24.0-40.0s peak
S3 40.0-48.0s falling
```

## Acceptance tests

- `test_beats_click`: on `click_090/120/150.wav`, mir_eval beat F-measure (70 ms window) >= 0.95 against truth, both with Beat This! (gpu mark) and the librosa fallback (>= 0.9).
- `test_downbeats_click`: downbeat F-measure >= 0.9 on `click_120.wav` for both paths.
- `test_half_time`: with `music.max_bpm=140`, `click_150.wav` gives `beat_mode="half"` and about half the beats.
- `test_sections`: on `song_sections.wav`, both true boundaries found within +/- 1 bar.
- `test_intensity_steps`: on `click_120_dynamic.wav`, mean intensity of the 3 loudness segments is strictly increasing.
- `test_excerpt_rules`: excerpt length in [45, 60], a whole number of bars (+/- 1 frame at 30 fps), starts on a downbeat; `peak` hint places the max intensity at 0.55-0.85 of the window.
- `test_emotion_fallback`: with no head weights, returns arrays of grid length, `source="heuristic"`, valence all zeros.
- `test_music_cache`: second analysis of the same song loads from cache (no Beat This! call).
- Full-song analysis of a 3.5 min song under 20 s on Colab (MERT excluded).
