# Phase 6: Render

Goal: turn an EDL into a frame-exact MP4 with the song, at preview (720p) or export (1080p) quality.

Output is always 16:9. Preview: 1280x720, CRF 23. Export: 1920x1080, CRF 18. Both libx264 `-preset veryfast`, `yuv420p`, 30 fps, AAC 192 kbps, `+faststart`. All shots of one render use identical encode settings so they can be concatenated without re-encoding.

## 1. Framing (`render/framing.py: decide(asset, slot, feats) -> (framing, kenburns)`)

Let a = width / height (display), T = 16/9.
- |a - T| / T < 0.02 -> `native` (scale only).
- a < 1.1 (portrait or square) -> `fill_blur`.
- 1.1 <= a < T: vertical crop loss = 1 - a/T. Loss <= `render.max_crop_loss` (0.2) -> `crop`, else `fill_blur`.
- a > T: horizontal crop loss = 1 - T/a. If a >= `render.pano_aspect` (2.5) and photo -> `pan`; elif loss <= 0.2 -> `crop`; else `fill_blur`.

Crop placement: center on the face union center (`face_cx`, `face_cy`) when present, else frame center, clamped to the image.

`fill_blur`: background = the same frame scaled to cover 16:9, Gaussian blurred (sigma `render.blur_sigma` 30), darkened (brightness -0.15); foreground = frame scaled to fit inside 16:9, centered on top.

Ken Burns (photos only, not for `pan`):
- `zoom_from = 1.0`, `zoom_to = 1.0 + kb_base + kb_gain * intensity_rank_of_slot` (defaults 0.04, 0.08).
- pan: drift toward the face center if any, else a small seeded drift; |pan| <= 0.03 of frame width/height.
- easing: smoothstep over the shot.
`pan` framing (panoramas): window of full height, 16:9 wide, moving from left to right (or reverse, seeded) across the image over the shot.

## 2. Shot rendering (`render/shots.py`)

Each shot renders to `SCRATCH/render/<run_id>/shot_<jjj>.mp4` with **exactly** `n = frame_end - frame_start` frames (`-frames:v n`). Video audio is dropped (`-an`).

**Video shots** (FFmpeg):
```
ffmpeg -y -ss {src_in:.3f} -i {src} -t {n/fps + 0.25:.3f}
  -vf "[hdr_tonemap,]{framing_filters},fps=30,tpad=stop_mode=clone:stop_duration=1,setsar=1,format=yuv420p"
  -frames:v {n} -an -c:v libx264 -preset veryfast -crf {crf} -r 30 {out}
```
- HDR tonemap chain from spec 01 when `is_hdr` (zscale available) else none.
- `crop`: `crop=w:h:x:y,scale=W:H`. `native`: `scale=W:H`.
- `fill_blur`: `split[a][b];[a]scale=W:H:force_original_aspect_ratio=increase,crop=W:H,gblur=sigma=30,eq=brightness=-0.15[bg];[b]scale=W:H:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2`.
- `tpad` guarantees n frames even if the source ends early.
- Rotation: FFmpeg autorotate stays on; do not apply rotation manually.

**Photo shots** (Python Ken Burns, subpixel):
- Load the full-res original (HEIC via pillow-heif), `exif_transpose`, RGB. Downscale so the long side <= 3x output width (speed).
- Build the static composed frame for `fill_blur` (blurred cover background + fitted foreground) at 1.15x output size so zoom has room; for `crop`/`native`, use the image itself.
- For frame k: s = smoothstep(k / (n - 1)); zoom = z0 + (z1 - z0) * s; center = c0 + (c1 - c0) * s; compute the source rectangle of size (W / zoom, H / zoom) in output-normalized coordinates, map with `cv2.warpAffine` (INTER_AREA when shrinking, else INTER_LINEAR) to W x H. Never round the rectangle to integer pixels (that is what causes zoompan jitter).
- Pipe raw `rgb24` frames to FFmpeg stdin: `-f rawvideo -pix_fmt rgb24 -s WxH -r 30 -i -` with the same encode settings and `-frames:v n`.

Render shots in parallel (`render.workers`, 8). Total shot time is logged.

## 3. Assemble (`render/assemble.py`)

1. `concat.txt` listing shots in order; `ffmpeg -f concat -safe 0 -i concat.txt -c copy video.mp4`.
2. Audio: `ffmpeg -ss {song_start} -i {song} -t {L} -af "afade=t=out:st={L - fade}:d={fade},apad,atrim=0:{L}" -c:a aac -b:a 192k audio.m4a`.
3. Mux: `ffmpeg -i video.mp4 -i audio.m4a -map 0:v -map 1:a -c copy -movflags +faststart out.mp4`.
4. Verify with ffprobe (`-count_packets`): video packet count == total frames; audio duration within 1 frame of L. Mismatch raises (bug, not a user error).
5. Copy `out.mp4` to `DATA_ROOT/runs/<run_id>/preview.mp4` or `video.mp4`; delete scratch shots.

## Acceptance tests (media/gpu marks as needed)

- `test_framing_table`: aspect cases -> expected framing (16:9 native, 4:3 crop, 3:4 fill_blur, 1:1 fill_blur, 3:1 photo pan, 2.2:1 video fill_blur).
- `test_shot_frame_count`: each rendered shot has exactly n frames (ffprobe `-count_frames`).
- `test_total_frames`: final video frames == round(L * 30); audio length == L +/- 1/30 s.
- `test_fill_blur_no_black`: portrait fixture shot: left and right 10% columns have mean luma > 10.
- `test_rotated_upright`: rotated fixture renders with the expected orientation (compare to a reference rendered from a pre-rotated copy).
- `test_kenburns_smooth`: estimated per-frame zoom from phase correlation is monotonic with no reversals.
- `test_cut_positions`: render an EDL of solid-color fixture shots; PySceneDetect finds cuts at the EDL frame indices +/- 1.
- Timing: 40-shot preview under 25 s, export under 60 s on Colab.
