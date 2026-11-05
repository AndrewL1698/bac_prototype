# Phase 13 (render half): per-shot render cache

Track 4, ROADMAP item 7. The Beat This! mirror (Track 2) is the other half of phase 13 and has its own spec.

Goal: a recut re-encodes only the shots that changed. Today every render encodes every shot, so a recut that swaps one photo costs as much as a fresh generate (warm generate 29.8 s against a 30 s target).

## 1. What is cached

- **Shots.** Each encoded shot (`render/shots.py: render_shot`) is stored as `<key>.mp4`.
- **Song audio.** The AAC-encoded excerpt with its fade-out is stored as `audio_<key>.m4a`. It is the slowest step of assembly (about 3.4 s for 50 s of audio on a laptop), and most recuts keep the same excerpt.

Assembly still runs on every render: stream-copy concat, mux, and the frame/audio count check. It is cheap (well under a second for a 40-shot cut).

## 2. Keys (`render/shot_cache.py`)

`shot_key(shot, asset, W, H, crf, face)` is a `cache.text_hash` of:
- `SHOT_CACHE_VERSION` (bump when rendering code changes pixels without a config change);
- `asset.hash` (source bytes) and `shot.kind`;
- the file actually read: `photo`, or for video `original` vs HDR `proxy`, and whether tonemapping runs (`needs_tonemap and zscale_available()`);
- `src_in` (rounded to 1 ms), frame count `frame_end - frame_start`, `framing`, `kenburns`;
- output `W`, `H`, `crf`, and the face center (rounded to 1e-4);
- the `render` config, except fields that do not change pixels (`workers`, `ffmpeg_bin`, `ffprobe_bin`, `audio_bitrate`, `preview`, `export`, `shot_cache`; size and CRF are keyed directly).

Not in the key: `slot_idx`, `start`/`end`, `frame_start`, `scores`, `reason`. A shot that moves to another slot of the cut, unchanged, is reused.

`audio_key(song_hash, song_start, length, fade)` adds `render.audio_bitrate` and the version. `song_hash` is the content hash of the song file actually being rendered (`song_file_hash(song_path)`, memoized per process by path, mtime and size), not `EDL.song_hash`: a plan rendered with a different song file must not reuse the first song's audio.

## 3. Storage and safety

- Config: `render.shot_cache: {enabled, location, max_gb}`. `location: scratch` (default) stores in `SCRATCH/shot_cache/` (local SSD; recuts happen in the same runtime, and Drive writes would slow cold renders). `location: drive` stores in `DATA_ROOT/cache/shots/`, shared and surviving runtime recycles.
- Every entry is written with `paths.atomic_write`, so parallel renders (threads or processes) never see a partial file; a failed encode leaves no entry.
- A shot that appears twice in one EDL is encoded once.
- After each render, `prune()` deletes least-recently-used entries (by mtime, refreshed on every hit) until the cache is under `max_gb`, never deleting the files of the render that just finished.
- `enabled: false` restores the old behavior exactly (per-run scratch files).

## 4. Acceptance tests (`tests/test_render_cache.py`)

No media:
- the key ignores timeline position (slot index, frame offset);
- the key changes with length, framing, Ken Burns, output size, face, source bytes, video in-point, and pixel-affecting render config; it ignores `workers`, `audio_bitrate`, `shot_cache.max_gb`;
- the audio key changes with song, start, length, fade, bitrate; the song hash changes when the file at the same path changes;
- hit/miss: the second request is a hit, encodes once, and leaves no temp file; a failed encode leaves no entry;
- prune drops least recently used files and honors `keep`.

Media:
- a 6-shot cut rendered twice encodes 6 shots, then 0; a recut that swaps two shots and changes one encodes exactly 1, with the right frame count and order;
- a cached render matches an uncached one (same frame count, mean pixel difference < 1);
- the same plan rendered with two different song files stores two audio entries;
- a 40-shot recut that changes 3 shots takes less than half the cold render time (laptop: 8.0 s cold, 1.7 s recut).
