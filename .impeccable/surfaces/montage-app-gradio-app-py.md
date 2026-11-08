---
version: 1
slug: "montage-app-gradio-app-py"
primary_target: "montage/app/gradio_app.py"
related_targets: []
---

# Surface: Montage editor workspace (montage/app/gradio_app.py)

Mode: Operate. Audience: non-professionals turning a camera roll plus one song into a short music video; demo viewers secondary.
Job: add media + song, write a brief, Generate, watch, pin/exclude, note, Recut, Export.
Constraints: Gradio 6.30 restyle only (theme + CSS + layout); API endpoints and first 11 generate/recut outputs unchanged; cuts-only product.

## Direction contract

THESIS: A friendly editor's workspace where the song is the clock: the program monitor leads, and every number reads in timecode, tempo and shot counts like a precise instrument. Refuses the stacked upload-form page and the near-black-plus-neon-glow AI look.

OWN-WORLD: Warm graphite ground and panels one step lighter, 1px warm hairlines, soft apricot-orange for the primary action, selection and readout highlights, dark ink on orange. Workhorse sans for UI, tabular mono for timecode and readouts. Small radii, no glow, no gradients. Viewfinder corner marks on the monitor.

STORY: The visitor sees where media goes, where the video will play and where to tell the director what they want; generates; reads the editor's note and the readout strip; clicks shots on the timeline to see why they were picked; pins or excludes, adds a note, recuts; exports.

FIRST VIEWPORT: 1440x900. Top bar: wordmark + tagline left, Export right. Three columns: media bin (left, ~22%), program monitor with readout strip beneath (center, ~50%), director panel with brief and Generate (right, ~28%), Generate visible without scrolling. Full-width timeline strip of shot thumbnails with mono timecodes, pin/exclude and the shot inspector, below. Lab drawer (curves, compare, rate, system details) collapsed at the bottom.

FORM: Pinned by the user brief (NLE workspace, orange/black softened, precise-instrument futurism); no concept roll. seed: none (direction pinned by user)
Signature move: the monitor readout strip, live tempo / shots / length / director set in tabular mono under a viewfinder-marked program monitor.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
