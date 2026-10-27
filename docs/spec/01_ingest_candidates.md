# Phase 1: Ingest and candidates

Goal: turn a folder of messy phone media into a clean list of `Asset`s and `Candidate`s, with low-res analysis frames on fast scratch.

## 1. Ingest (`montage/ingest/`)

`assets.py: ingest(paths: list[str]) -> list[Asset]`
- Accept: `.jpg .jpeg .png .heic .heif .webp` photos; `.mp4 .mov .m4v .avi .mkv` videos. Ignore others with a warning. Use `pillow_heif.register_heif_opener()` for HEIC.
- Validate counts: `ingest.min_assets` (default 10) to `ingest.max_assets` (100). Below min: `UserError("Need at least 10 photos or videos")`.
- Total video duration cap `ingest.max_total_video_s` (default 1200 s). Beyond it, analyze only the first `ingest.max_per_video_s` (default 120 s) of each long video and add an EDL note.
- Hash: sha1 of file bytes (stream in chunks), first 16 hex chars. Cached probe results at `cache/assets/<hash>.json`.

`probe.py`
- Videos: `ffprobe -v error -print_format json -show_streams -show_format`. Extract width/height, rotation (from `side_data_list` display matrix or `tags.rotate`), duration, avg fps, codec, color transfer (`smpte2084` or `arib-std-b67` means HDR), creation_time.
- Report **display** width/height (swap if rotation is 90/270).
- Photos: Pillow; apply `ImageOps.exif_transpose` before reading size; EXIF DateTimeOriginal for `taken_at`.
- `taken_at` fallback: file mtime (flag in notes as unreliable).
- Asset ids: sort by (taken_at, filename), assign `a000`, `a001`, ...

`frames.py: extract_frames(asset) -> list[(t, path)]`
- Videos: one FFmpeg call per video writing JPEGs at `ingest.analysis_fps` (default 2) and short side `ingest.analysis_short_side` (default 448) to `SCRATCH/frames/<hash>/`. FFmpeg autorotates by default; keep it on.
- HDR: when `is_hdr`, insert `zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p` before scaling. If zscale is unavailable, fall back to plain scaling and add a note.
- Motion needs consecutive frames: also extract a **motion pair** for every analysis frame: the frame at t and at t + 1/15 s (two JPEGs, 320 px short side). Store as `m_<idx>_a.jpg`, `m_<idx>_b.jpg`. (One FFmpeg pass at 15 fps scaled small, keeping only needed frames, is fine.)
- Photos: write one resized RGB JPEG (exif-transposed) at 448 short side plus keep path to the full-res original for rendering.
- Parallelize across videos with a thread pool (`ingest.workers`, default 8).

## 2. Candidates (`montage/candidates/`)

`scenes.py: detect_scenes(asset) -> list[(start, end)]`
- PySceneDetect `AdaptiveDetector` on a 360p proxy stream (or ContentDetector if Adaptive misbehaves), downscale factor auto. Minimum scene length 1.0 s.
- If detection yields nothing, the whole video is one scene.

`windows.py: make_candidates(assets, scenes) -> list[Candidate]`
- Photos: one candidate each (`id = asset.id`).
- Videos: within each scene, windows of `candidates.window_s` (3.0) with hop `candidates.hop_s` (1.5). Scenes shorter than window but >= `candidates.min_window_s` (1.0) give one window covering the scene. Drop the first and last `candidates.edge_trim_s` (0.3) of each video (phones capture shaky starts/stops).
- `src_max_end`: end of the window's scene (or the video end minus edge trim), so the planner can extend a window forward to fill a slot longer than 3 s.
- `frame_times`: analysis frame timestamps inside the window.
- `thumb_path`: the frame nearest the window center (photos: the 448 px image), resized to 256 px wide, written to `SCRATCH/thumbs/`.
- Window ids: `{asset_id}_w{nn}`; scene ids `{asset_id}_s{n}`.

## Acceptance tests

- `test_probe_rotation`: `vid_rotated.mov` reports display size with width < height (or the swapped orientation matching the metadata) and rotation 90.
- `test_probe_hdr_flag`: a fixture encoded with `-color_trc arib-std-b67` is flagged HDR.
- `test_ingest_counts`: 5 files raises `UserError`; unsupported files are skipped with a warning.
- `test_frames_rate`: a 12 s video yields 24 +/- 1 analysis frames at 2 fps and the same number of motion pairs.
- `test_scenes_twoscenes`: `vid_twoscenes.mp4` splits near 5.0 s (+/- 0.3).
- `test_windows`: 12 s video yields windows of 3 s with 1.5 s hop after edge trim; all windows inside source bounds; ids unique.
- `test_cache_hit`: second ingest of the same files performs zero ffprobe calls (patch subprocess and count).
- Timing: 20 fixture files ingest + candidates in under 30 s on Colab.
