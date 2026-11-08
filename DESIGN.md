---
name: Montage
description: An editor's workspace where the song is the clock; warm graphite, soft apricot, tabular-mono readouts.
colors:
  apricot: "#f2995c"
  apricot-hover: "#f7ab76"
  apricot-soft: "rgba(242, 153, 92, 0.14)"
  apricot-ink: "#1c130b"
  graphite-ground: "#191714"
  graphite-panel: "#221f1b"
  monitor-well: "#0c0b0a"
  graphite-input: "#2b2722"
  hairline: "#36312b"
  hairline-strong: "#48413a"
  bone-text: "#efe9e1"
  muted-text: "#a69d91"
  placeholder-text: "#9e9589"
typography:
  headline:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 600
    letterSpacing: "-0.015em"
  title:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 600
  body:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
  readout:
    fontFamily: "Geist Mono, ui-monospace, monospace"
    fontSize: "15px"
    fontWeight: 500
    fontFeature: "tnum"
  timecode:
    fontFamily: "Geist Mono, ui-monospace, monospace"
    fontSize: "11px"
    fontWeight: 400
    fontFeature: "tnum"
rounded:
  panel: "6px"
spacing:
  panel-gap: "8px"
  panel-pad: "10px"
  page-gutter: "20px"
  strip-pad: "11px 16px"
components:
  button-primary:
    backgroundColor: "{colors.apricot}"
    textColor: "{colors.apricot-ink}"
  button-primary-hover:
    backgroundColor: "{colors.apricot-hover}"
  button-secondary:
    backgroundColor: "{colors.graphite-input}"
    textColor: "{colors.bone-text}"
  button-secondary-hover:
    backgroundColor: "{colors.hairline}"
  button-disabled:
    backgroundColor: "{colors.graphite-input}"
    textColor: "{colors.placeholder-text}"
  input:
    backgroundColor: "{colors.graphite-input}"
    textColor: "{colors.bone-text}"
  panel:
    backgroundColor: "{colors.graphite-panel}"
    rounded: "{rounded.panel}"
    padding: "10px"
  readout-strip:
    backgroundColor: "{colors.graphite-panel}"
    typography: "{typography.readout}"
    padding: "11px 16px"
  timeline-shot:
    width: "192px"
    typography: "{typography.timecode}"
---

# Design System: Montage

## Overview

**Creative North Star: "The Song Is the Clock"**

Montage is a friendly editor's workspace, not an upload form. The program monitor leads the screen, and everything the system knows about a cut (tempo, shot count, length, which director planned it) reads like an instrument: tabular mono numerals in a strip bolted under the monitor. Media sits in a bin on the left, the director's brief on the right, the shots run as one horizontal strip below.

The world is warm graphite: a dark ground, panels one step lighter, 1px warm hairlines between everything. One accent, a soft apricot, carries the primary action, the current selection, viewfinder corner marks and the single highlighted readout value. Dark ink sits on the apricot, never white. The app stays dark regardless of the visitor's OS colour scheme.

The system refuses the near-black-plus-neon-glow AI look: no glow, no gradients as surfaces, no drop shadows. Depth comes from tone steps and hairlines. Density is moderate and workspace-like: small radii, tight 8px gaps, 13px panel titles.

**Key Characteristics:**
- Warm graphite tonal layering (ground, panel, input, well) with 1px hairlines; no shadows.
- One apricot accent, dark ink on it; used for action, selection and one live value.
- Geist for UI, Geist Mono with tabular numerals for every timecode and readout.
- Viewfinder corner marks on the 16:9 monitor well.
- Instrument stack: monitor, readout strip and shot inspector share one frame.

## Colors

A warm, low-chroma graphite ramp with a single soft apricot voice.

### Primary
- **Soft Apricot** (apricot): the primary button fill, focus outlines, input focus border, caret, checked radios, selected timeline shot ring, pinned-shot bar, viewfinder corners, and the tempo value in the readout.
- **Apricot Lift** (apricot-hover): primary button hover only.
- **Apricot Wash** (apricot-soft): text selection and the fill behind a selected radio/checkbox label.
- **Apricot Ink** (apricot-ink): text on apricot fills (8.7:1).

### Neutral
- **Graphite Ground** (graphite-ground): page background.
- **Graphite Panel** (graphite-panel): workspace panels, readout strip, inspector, blocks.
- **Monitor Well** (monitor-well): the near-black 16:9 program monitor; the darkest surface, reserved for the video.
- **Graphite Input** (graphite-input): inputs, secondary buttons, disabled buttons, upload drop zones, code.
- **Hairline** (hairline): every 1px panel border and title divider; secondary button hover fill.
- **Strong Hairline** (hairline-strong): input and secondary button borders, dashed upload outline, scrollbar thumb.
- **Bone** (bone-text): primary text.
- **Muted Stone** (muted-text): labels, hints, captions, readout keys, chevrons (6.1:1 on panel).
- **Placeholder Stone** (placeholder-text): placeholders, disabled text, idle readout values (5.0:1 on input).

### Named Rules
**The One Voice Rule.** Apricot is the only hue. It marks what to do next, what is selected, and one live value; it never decorates.

**The Ink-On-Apricot Rule.** Text on an apricot fill is Apricot Ink, never white.

**The Always-Dark Rule.** Light and dark theme variables carry identical values; the workspace is dark whatever the OS says.

## Typography

**Display Font:** none; the wordmark is the largest type.
**Body Font:** Geist (with system-ui, sans-serif)
**Label/Mono Font:** Geist Mono (with ui-monospace, monospace)

**Character:** A plain workhorse sans for everything a person reads, and a tabular mono for everything a machine measures.

### Hierarchy
- **Headline** (600, 19px, -0.015em): the Montage wordmark only.
- **Title** (600, 13px): panel titles (Media, Director, Timeline), with an inline 400-weight muted hint.
- **Body** (400, 13-14px, 1.5): tagline, editor's note, shot inspector.
- **Label** (400, 12px): readout keys (Tempo, Shots, Length, Director) in muted stone.
- **Readout** (500, 15px, tabular): readout values; the Director value switches to Geist since it is a word.
- **Timecode** (400, 11px, tabular): timeline thumbnail captions.

### Named Rules
**The Tabular Numbers Rule.** Any timecode, BPM, count or duration is set in Geist Mono with tabular numerals. Words stay in Geist, even inside the readout.

## Layout

A three-column NLE workspace under a top bar. At 1440px: media bin (scale 3, min 260px), monitor column (scale 6, min 420px), director panel (scale 3, min 280px), then a full-width timeline panel and a collapsed Lab accordion. The top bar holds wordmark plus tagline left and Export right, closed by a hairline. Page gutter 20px, bottom padding 28px; panels pad 10px with 8px gaps.

The timeline is a single horizontal scroll strip of 16:9 thumbnails, 192px wide each. Below 480px the readout reflows into a two-column grid and the brief textarea grows to 9em; Gradio stacks the columns.

## Elevation & Depth

Flat. Block and input shadows are explicitly `none`. Depth is tonal: the monitor well sits darkest, ground next, panels one step lighter, inputs one step lighter again, each edged by a 1px hairline. The only ring is the 2px apricot `box-shadow` used as a selection outline on the chosen timeline shot.

### Named Rules
**The Tone-Not-Shadow Rule.** Separate surfaces by a tone step and a hairline, never by a cast shadow or glow.

## Shapes

Small, quiet corners: 6px on workspace panels and on the outer corners of the monitor stack; Gradio's small radius scale on inputs and buttons. The monitor, readout strip and inspector stack into one frame: the monitor rounds only its top corners, the inspector only its bottom, and the strip between overlaps by -1px so the hairlines merge. Upload drop zones use a dashed strong hairline. Disclosure chevrons are drawn as two 1.5px borders rotated, not a text glyph.

## Components

### Buttons
Calm and solid.
- **Shape:** Gradio small radius.
- **Primary:** apricot fill, apricot border, ink text. Only one primary at a time: Generate until a cut exists, then Recut takes over and Generate drops to secondary.
- **Hover / Focus:** hover lifts to Apricot Lift; focus-visible draws a 2px apricot outline at 2px offset.
- **Secondary:** graphite input fill, strong hairline border, bone text; hover fills with hairline.
- **Disabled:** graphite input fill, placeholder text, hairline border, full opacity, not-allowed cursor.

### Chips (radio source toggles)
- **Style:** graphite input fill with strong hairline border.
- **State:** selected takes the Apricot Wash fill and an apricot border; text stays bone.

### Cards / Containers (workspace panels)
- **Corner Style:** 6px.
- **Background:** graphite panel.
- **Shadow Strategy:** none (see Elevation).
- **Border:** 1px hairline; blocks inside sit flat and borderless so each input keeps a single frame.
- **Internal Padding:** 10px, 8px gaps. Title row closes with a hairline and 10px bottom padding.

### Inputs / Fields
- **Style:** graphite input fill, 1px strong hairline, placeholder stone text, apricot caret.
- **Focus:** border turns apricot; keyboard focus adds the 2px apricot outline.

### Navigation
None beyond the top bar and the Lab accordion, whose disclosure is a drawn chevron rotating from -45deg to 45deg over 0.2s.

### Program Monitor with Readout Strip (signature)
A 16:9 monitor well with four apricot viewfinder corners (18px by 1.5px L-marks, inset 12px, 0.85 opacity). Attached beneath, a readout strip of key/value pairs (muted 12px keys, 15px tabular mono values, 32px column gap); the tempo value is lit apricot. When idle, values show dashes in placeholder stone. When a cut lands, values settle in (0.45s, opacity 0.25 to 1, rising 3px). The shot inspector closes the stack below.

### Timeline Strip
Horizontal row of 192px 16:9 thumbnails with mono timecode captions. Selected shot: 2px apricot ring. Pinned: a 3px apricot bar across the top edge. Excluded: 35% opacity, greyscale.

## Do's and Don'ts

### Do:
- **Do** keep apricot to action, selection, focus, viewfinder marks and the one lit readout value.
- **Do** set every number that measures time, tempo or count in Geist Mono with tabular numerals.
- **Do** separate surfaces with a tone step plus a 1px warm hairline.
- **Do** keep exactly one primary button live per state (Generate before a cut, Recut after).
- **Do** honour `prefers-reduced-motion`: the readout settle and chevron rotation switch off.

### Don't:
- **Don't** add glow, neon, gradient surfaces or drop shadows (the near-black-plus-neon AI look).
- **Don't** put white text on apricot; use Apricot Ink.
- **Don't** add a light theme or let the OS scheme lighten the workspace.
- **Don't** add a second accent hue.
- **Don't** add motion beyond the readout settle and the chevron rotation.
