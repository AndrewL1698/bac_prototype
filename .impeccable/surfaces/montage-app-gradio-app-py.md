---
version: 1
slug: "montage-app-gradio-app-py"
primary_target: "montage/app/gradio_app.py"
related_targets: []
---

# Surface: Montage editor workspace (montage/app/gradio_app.py)

Mode: Operate. Audience: non-professionals turning a camera roll plus one song into a short music video; demo viewers secondary.
Job: add media + song, write a brief, Generate, watch, keep/remove shots, note, Recut, Export.
Constraints: Gradio 6.30 restyle only (theme + CSS + layout); API endpoints and first 11 generate/recut outputs unchanged; cuts-only product. User pins: sleek, not pastel, not AI-looking; built for non-professionals.

## Direction contract

THESIS: Your shots are prints laid out on a dark table: pick one up, keep it or put it aside, and the music decides the order. Refuses both category defaults: the dark graphite pro NLE and the pastel soft-card consumer app.

OWN-WORLD: Neutral black table (no warm tint), panels one step up, 1px neutral hairlines, crisp white prints, one signal red reserved for action, selection and kept marks, white text on red. Archivo for everything, tabular figures for numbers. Small radii on panels, near-square print corners, no glow, no gradients, no handwriting, no tilt.

STORY: The visitor sees where media goes, where the video plays and where to tell the director what they want; generates; reads big tempo, shot and length figures; picks a print from the timeline, reads why it is there, keeps or removes it, adds a note and recuts; exports.

FIRST VIEWPORT: 1440x900. Top bar: wordmark + tagline left, Export right. Three columns: Media panel, dark monitor screen with the readout strip of large figures beneath, Director panel with brief and Generate. Below, the Revise panel: a row of white-framed prints whose wide bottom margin carries timecode and a state cell, then Selected shot (as a print) / Note / Changes with Recut, all above the fold. Lab drawer collapsed at the bottom.

FORM: Print Table (instant prints on a dark table), my candidate 7 of 7, assigned by the roll; chosen by the user over Liner Notes (pick), Boarding Pass, Feed Native and Warm Consumer. seed:377e480ce57a6abd
Signature move: the timeline as a row of white instant prints; each print's bottom margin holds its timecode and a fixed state cell (hollow open, red filled kept, struck removed).
Raises: scale-contrast hierarchy in the readout (large figures, small labels); state as a mark in a fixed cell; the dock sequences the next decision (Keep/Remove stay disabled until a print is picked, Recut carries the pending count).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
