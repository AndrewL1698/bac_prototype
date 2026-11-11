---
name: Montage
description: Print Table. Your shots are white instant prints on a neutral black table; one signal red marks action, selection and what you kept.
colors:
  signal-red: "#d22f22"
  signal-red-hover: "#e5402f"
  signal-red-wash: "rgba(210, 47, 34, 0.16)"
  red-ink: "#ffffff"
  table-black: "#121212"
  panel: "#1a1a1a"
  monitor-well: "#0a0a0a"
  input: "#232323"
  hairline: "#2b2b2b"
  hairline-strong: "#3c3c3c"
  text: "#f2f2f2"
  muted-text: "#9ba1a9"
  placeholder-text: "#8f959e"
  print-white: "#fafafa"
  print-ink: "#121212"
typography:
  headline:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 600
    letterSpacing: "-0.015em"
  readout-figure:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "28px"
    fontWeight: 800
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
  readout-word:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 600
  title:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 600
  body:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  body-small:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  readout-label:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
  print-caption:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    fontFeature: "tnum"
  dock-label:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 500
    letterSpacing: "0.06em"
rounded:
  print: "2px"
  panel: "6px"
spacing:
  inner: "8px"
  panel: "10px"
  readout-x: "16px"
  page-x: "20px"
  dock: "20px"
  readout-gap: "32px"
components:
  button-primary:
    backgroundColor: "{colors.signal-red}"
    textColor: "{colors.red-ink}"
  button-primary-hover:
    backgroundColor: "{colors.signal-red-hover}"
  button-secondary:
    backgroundColor: "{colors.input}"
    textColor: "{colors.text}"
  button-secondary-hover:
    backgroundColor: "{colors.hairline}"
  button-disabled:
    backgroundColor: "{colors.input}"
    textColor: "{colors.placeholder-text}"
  input:
    backgroundColor: "{colors.input}"
    textColor: "{colors.text}"
  panel:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.panel}"
    padding: "10px"
  readout:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    typography: "{typography.readout-figure}"
    padding: "11px 16px"
  print:
    backgroundColor: "{colors.print-white}"
    textColor: "{colors.print-ink}"
    typography: "{typography.print-caption}"
    rounded: "{rounded.print}"
    padding: "5px 5px 24px"
    width: "168px"
  monitor:
    backgroundColor: "{colors.monitor-well}"
    rounded: "{rounded.panel}"
---

# Design System: Montage

## Overview

**Creative North Star: "The Print Table"**

Your shots are prints laid out on a dark table: pick one up, keep it or put it aside, and the music decides the order. The table is a neutral black with no warm tint; panels sit one step up on it, divided by 1px neutral hairlines. The footage itself arrives as crisp white instant prints, and the only colour the interface adds is one signal red, spent on action, selection and kept marks.

The world refuses both category defaults: the dark graphite pro editor and the pastel soft-card consumer app. It is dark in every OS colour scheme. Hierarchy comes from scale contrast (large figures, small quiet labels) and from state shown as a mark in a fixed cell, not from fills, glows or badges. Archivo carries everything, with tabular figures wherever a number can change.

Density is a workspace, not a landing page: three panels and a revise dock fit a 1440x900 viewport, and each panel draws exactly one frame so inputs inside it sit flat.

**Key Characteristics:**
- Neutral black table, panels one step up, 1px hairlines; no warm tint.
- White instant prints whose wide bottom margin carries timecode and a state cell.
- One signal red for action, selection and kept; white text on red.
- Archivo only; tabular figures for every number.
- Small radii on panels, near-square print corners, no shadows, no glow, no gradients, no tilt.
- One authored motion: readout values settle in when a cut lands.

## Colors

A neutral grey ladder from monitor black to print white, with a single signal red.

### Primary
- **Signal Red** (signal-red): the filled primary button (Generate before a cut, Recut after), focus outlines, the text caret, the selected-print ring, the kept mark in a print's state cell, list markers in Changes, radio and checkbox selection, error borders, and the one lit readout figure (tempo).
- **Signal Red Lift** (signal-red-hover): hover on the primary button.
- **Red Wash** (signal-red-wash): text selection, selected checkbox-label fill, and the error panel background.
- **Red Ink** (red-ink): white text on any red fill (5.0:1).

### Neutral
- **Table Black** (table-black): the page ground.
- **Panel** (panel): the one step up for workspace panels and the readout strip.
- **Monitor Well** (monitor-well): the 16:9 program monitor, the darkest surface, so footage reads as the brightest thing on screen.
- **Input** (input): text fields, secondary buttons, the upload drop zone, disabled buttons.
- **Hairline** (hairline): panel borders, panel-title and top-bar rules, the dock divider.
- **Hairline Strong** (hairline-strong): input and secondary-button borders, the dashed upload outline, the empty selected-shot frame, scrollbars.
- **Text** (text): primary text.
- **Muted Text** (muted-text): labels, hints, readout labels, idle copy (6.7:1 on panel).
- **Placeholder Text** (placeholder-text): placeholders, disabled text, idle readout figures (5.2:1 on input).
- **Print White** (print-white): the instant-print frame, in the timeline and the selected-shot card.
- **Print Ink** (print-ink): timecode and state-cell strokes written on a print's margin (17.9:1).

### Named Rules
**The One Signal Rule.** Red is reserved for action, selection, kept marks and errors. Nothing decorative is red. A screen with nothing selected and nothing to do carries one red element at most.

**The One Lit Figure Rule.** Exactly one readout value is lit red: tempo, because the music owns timing. Every other figure stays in text colour. Do not light a second figure.

**The Neutral Table Rule.** Greys stay neutral (no warm or cool tint in surfaces). Colour on the table comes from the footage and the one red, never from the chrome.

## Typography

**Display Font:** Archivo (with system-ui, sans-serif)
**Body Font:** Archivo
**Label/Mono Font:** Archivo with tabular figures; there is no monospace face.

**Character:** One grotesque at many weights. Numbers get tabular figures so timecodes and readouts never jitter as they change.

### Hierarchy
- **Headline** (600, 19px, -0.015em): the wordmark only.
- **Readout Figure** (800, 28px, -0.02em, tabular; 22px under 480px): tempo, shot count, length.
- **Readout Word** (600, 16px): a readout value that is a word rather than a number (the director in use).
- **Title** (600, 13px): panel titles (Media, Director, Revise), with an optional 400-weight muted hint after them.
- **Body** (400, 14px, 1.5): the editor's note and general copy.
- **Body Small** (400, 13px, 1.5): tagline, selected-shot reasoning, Changes list, empty-state copy.
- **Readout Label** (400, 12px, muted): the small word beside each readout figure.
- **Print Caption** (600, 11px, tabular, print ink): timecode and short id on a print's margin, with " · kept" or " · out" appended.
- **Dock Label** (500, 11px, 0.06em, uppercase, muted): column labels inside the Revise dock only (Selected shot, Note for the editor, Changes).

### Named Rules
**The Scale Contrast Rule.** In the readout, figures are large and heavy, labels small and muted, on one baseline. Hierarchy is the size gap, not colour or boxes.

**The Tabular Rule.** Every number that can change (timecode, BPM, counts, length) uses tabular figures.

## Layout

A single page, 20px side padding. A top bar (wordmark and tagline left, Export right) sits over a 1px hairline. Below it, three columns in a 3 : 6 : 3 ratio: the Media panel, the monitor with the readout strip attached beneath it, and the Director panel. Under those, the full-width Revise panel: one horizontal row of prints (168px each, scrolls sideways), a hairline, then the dock with three columns: Selected shot, Note, Changes. The Lab accordion stays collapsed at the bottom.

Spacing is tight and consistent: 10px panel padding, 8px between blocks inside a panel, 20px between dock columns, 32px between readout cells. Under 480px the top bar stacks, the readout becomes a two-column grid, and the brief field grows taller.

## Elevation & Depth

Flat. There are no shadows anywhere (block and input shadows are explicitly off). Depth comes from tonal steps: monitor well below the table, the table, panels one step up, inputs one more step up, and the white prints as the brightest objects. The selected print lifts 4px and gains a 2px red ring; that is a state, not elevation.

### Named Rules
**The Tonal Step Rule.** Separate surfaces with a one-step grey change and a 1px hairline, never with a shadow or glow.

## Shapes

Small radii on panels and the monitor (6px), near-square corners on prints (2px), and none on the image inside a print. The monitor and readout share one silhouette: the monitor rounds its top corners, the readout its bottom ones, with a 1px overlap so they read as one object. Borders are always 1px solid hairlines, except the upload drop zone, which is dashed. Drawn marks (the state cell, the accordion chevron) are built from 1.5px strokes.

## Components

### Buttons
- **Shape:** small radius from the theme's small radius scale.
- **Primary:** signal red with white text; hover lifts to signal red lift. Only one primary is visible at a time: Generate before a cut exists, Recut after (Generate drops to secondary). Recut carries the pending count ("Recut · 2 changes").
- **Secondary:** input grey, text colour, strong hairline border; hover goes to hairline grey. Used for Export, Keep/Remove, Put back removed shots.
- **Disabled:** input grey, placeholder text, hairline border, full opacity, not-allowed cursor. Keep/Remove and Recut stay disabled until there is something to act on.
- **Focus:** 2px signal red outline, 2px offset, on buttons, radios and fields.

### Inputs / Fields
- **Style:** input grey fill, 1px strong hairline, no shadow, red caret.
- **Focus:** border turns signal red.
- **Upload:** the drop zone is a dashed strong-hairline frame on input grey.
- **Choice toggles:** source choices (Upload / Demo) are radio pills; selected gets a red wash, red border and red dot.

### Cards / Containers (workspace panel)
- **Corner Style:** 6px.
- **Background:** panel grey with a 1px hairline frame; blocks inside are transparent and borderless, so each input keeps a single frame.
- **Title:** 13px semibold over a hairline rule, optional muted hint inline.
- **Internal Padding:** 10px, 8px gaps.

### Readout Strip (signature)
Attached under the monitor: label and figure pairs on one baseline. Idle (before any cut) every figure is placeholder grey; once a cut lands, values settle in with the one authored motion (0.45s, opacity 0.25 to 1 and 3px rise, cubic-bezier(.16, 1, .3, 1)), disabled under reduced motion. Tempo is the one lit figure.

### Instant Print (signature)
The timeline is a row of white prints. Each print is a 16:9 image inside a print-white frame with a 5px border and a 24px bottom margin. The margin holds a fixed 9px state cell and the timecode caption in print ink:
- **Open:** hollow square, 1.5px print-ink stroke.
- **Kept:** square filled signal red.
- **Removed:** square struck with a diagonal; the image goes grayscale at 60% opacity.
- **Selected:** 2px signal red ring and a 4px lift.

The state is keyed off the caption suffix (" · kept" / " · out"), so the app writes state as words and the theme draws the mark. The selected-shot card in the dock repeats the print form (6px frame, 22px bottom margin, max 180px); with nothing selected it is an empty strong-hairline frame.

### Revise Dock
Three columns under the print row: Selected shot (print card, reasoning, Keep/Remove), Note for the editor, Changes (Recut first, then a red-marked list of pending changes, then Put back removed shots). Changes announces updates politely to assistive tech.

### Accordion
Disclosure uses a drawn chevron (two 1.5px muted strokes) that rotates 90 degrees over 0.2s, not a text triangle.

### Empty and Error States
Empty print row: plain muted line "Your shots appear here as prints after you Generate." with no icon. Errors use the red wash fill, red border, text-colour copy and a red icon.

## Do's and Don'ts

### Do:
- **Do** keep every surface on the neutral grey ladder (table-black, panel, input) and separate steps with 1px hairlines.
- **Do** show footage as white prints with the timecode and state cell on the bottom margin.
- **Do** show state as a mark in a fixed cell (hollow open, red kept, struck removed), written by the app as a caption suffix.
- **Do** keep exactly one filled red button visible; move primary to the next decision (Generate, then Recut).
- **Do** use tabular figures for every changing number and scale contrast for hierarchy in the readout.
- **Do** keep the app dark regardless of the visitor's OS colour scheme.

### Don't:
- **Don't** use red for decoration, or light a second readout figure.
- **Don't** add shadows, glows or gradients; the struck state cell is the only drawn diagonal.
- **Don't** tilt prints, add handwriting, or warm-tint the greys.
- **Don't** introduce a second typeface or a monospace face for numbers.
- **Don't** draw a second frame inside a panel; blocks in a panel sit flat.
- **Don't** add transitions or overlays that imply effects between shots; the only motion is the readout settle and the chevron turn.
