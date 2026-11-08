# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Non-professionals: people with a phone full of photos and clips from a trip, party or season who want a short music video of it without learning an editor. They arrive with 30-80 files and one song, and want something worth sharing in a few minutes. Secondary audience: workshop demo viewers and the team running evaluations (Compare, Rate, under-the-hood views).

## Product Purpose

Montage turns 30-80 photos/videos plus one song into a 45-60 second, 16:9, 1080p montage. Cuts land on beats, footage mood follows the song's energy, and the user steers with a plain-language director's brief and revision notes. Success: a first cut the user likes, then fast recuts that clearly respond to their notes.

## Positioning

The user directs in words, the music owns the timing, the system does everything else (selection, order, trims, framing, render), and every choice explains itself (editor's note, a "why" per shot).

## Operating Context

Gradio app served from a Colab A100 through a share link, used in a desktop browser. Flow: add media + song (or a demo album/song), write a brief, Generate (about 30 s warm), watch the preview, pin/exclude shots and add notes, Recut, Export 1080p. Live demo use: must work every time.

## Capabilities and Constraints

- Cuts only in v1: no transitions, filters or text overlays. Clip audio muted; only the song plays.
- UI is Gradio (pinned 6.30); restyled with a theme and CSS, not a custom frontend. The `/generate`, `/recut`, `/export`, `/naive`, `/load_pair`, `/vote_*` API endpoints and the first 11 outputs of generate/recut are a tested contract.
- Every learned component has a fallback; the UI must show which ran without alarming non-pros.
- Public repo: no personal media, songs or secrets in the repo.

## Brand Commitments

- Name: Montage. Tagline: "You direct, the music sets the timing, the system does the rest."
- Palette commitment from the user: orange and black, softened.
- Workspace commitment from the user: a professional video editor's layout (media bin, program monitor, inspector, timeline strip), made friendly for non-pros; "future" expressed as precise-instrument detail (timecode, BPM, beat readouts, fine lines), not glow.

## Evidence on Hand

Real pipeline outputs per run: preview video, EDL with per-shot reasons and scores, music curves and score plots, stage timings. No testimonials, customer names or benchmarks for the UI; do not invent them.

## Product Principles

1. The video is the hero; everything else serves getting a better cut.
2. Explain every choice in plain words; numbers are for those who look.
3. One obvious next action at each stage: add, generate, revise, export.
4. Demo-safe: degrade quietly, never crash, never confuse.
