"""Montage look: warm graphite + soft apricot, editor-workspace chrome, tabular-mono readouts. Dark in every mode."""

import inspect

import gradio as gr

# contrast checked: MUTED on PANEL 6.1:1, PLACEHOLDER on INPUT 5.0:1, INK on ACCENT 8.7:1
GROUND, PANEL, WELL, INPUT = "#191714", "#221f1b", "#0c0b0a", "#2b2722"
LINE, LINE_STRONG = "#36312b", "#48413a"
TEXT, MUTED, PLACEHOLDER = "#efe9e1", "#a69d91", "#9e9589"
ACCENT, ACCENT_HOVER, ACCENT_SOFT, INK = "#f2995c", "#f7ab76", "rgba(242, 153, 92, 0.14)", "#1c130b"

_vars = dict(
    body_background_fill=GROUND, body_text_color=TEXT, body_text_color_subdued=MUTED,
    background_fill_primary=PANEL, background_fill_secondary=GROUND,
    block_background_fill=PANEL, block_border_color=LINE, block_label_background_fill=PANEL,
    block_label_text_color=MUTED, block_label_border_color=LINE, block_title_text_color=MUTED, block_info_text_color=MUTED,
    border_color_primary=LINE, border_color_accent=ACCENT,
    input_background_fill=INPUT, input_background_fill_focus=INPUT, input_border_color=LINE_STRONG,
    input_border_color_focus=ACCENT, input_placeholder_color=PLACEHOLDER,
    button_primary_background_fill=ACCENT, button_primary_background_fill_hover=ACCENT_HOVER,
    button_primary_text_color=INK, button_primary_border_color=ACCENT,
    button_secondary_background_fill=INPUT, button_secondary_background_fill_hover=LINE,
    button_secondary_text_color=TEXT, button_secondary_border_color=LINE_STRONG,
    color_accent=ACCENT, color_accent_soft=ACCENT_SOFT, link_text_color=ACCENT, loader_color=ACCENT,
    checkbox_background_color=INPUT, checkbox_background_color_selected=ACCENT, checkbox_border_color=LINE_STRONG,
    checkbox_label_background_fill=INPUT, checkbox_label_background_fill_selected=ACCENT_SOFT,
    checkbox_label_border_color=LINE_STRONG, checkbox_label_border_color_selected=ACCENT,
    checkbox_label_text_color=TEXT, checkbox_label_text_color_selected=TEXT,
    panel_background_fill=PANEL, panel_border_color=LINE, code_background_fill=INPUT,
    table_even_background_fill=PANEL, table_odd_background_fill=INPUT, table_border_color=LINE,
)
# same values for light and dark: the app is dark whatever the visitor's OS says
_has_dark = inspect.signature(gr.themes.Base.set).parameters
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(block_shadow="none", input_shadow="none", input_border_width="1px", **_vars,
      **{k + "_dark": v for k, v in _vars.items() if k + "_dark" in _has_dark})

_corner = f"linear-gradient({ACCENT} 0 0)"
CSS = f"""
.gradio-container {{ padding: 0 20px 28px !important; }}
::selection {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
input, textarea {{ caret-color: {ACCENT}; }}
* {{ scrollbar-color: {LINE_STRONG} transparent; scrollbar-width: thin; }}
button:focus-visible, [role="radio"]:focus-visible, input:focus-visible, textarea:focus-visible {{
  outline: 2px solid {ACCENT}; outline-offset: 2px; }}

#topbar {{ align-items: center; padding: 14px 0 12px; margin-bottom: 4px; border-bottom: 1px solid {LINE}; }}
#wordmark h1 {{ font-size: 19px; font-weight: 600; letter-spacing: -0.015em; margin: 0 0 2px; }}
#wordmark p {{ margin: 0; font-size: 13px; color: {MUTED}; }}

/* workspace panels: one frame per panel, blocks inside sit flat so each input keeps a single frame */
.ws-panel {{ background: {PANEL} !important; border: 1px solid {LINE} !important; border-radius: 6px !important;
  padding: 10px !important; gap: 8px !important; }}
.ws-panel .block {{ background: transparent !important; border: 0 !important; box-shadow: none !important;
  padding: 4px 0 !important; }}
.ws-panel .block:has(.upload-container) {{ border: 1px dashed {LINE_STRONG} !important; background: {INPUT} !important; }}
/* Gradio groups neighbouring inputs in a .form whose background is its divider colour; flatten it in panels */
.ws-panel .form {{ background: transparent !important; border: 0 !important; gap: 8px !important; }}
@media (max-width: 480px) {{ #brief textarea {{ min-height: 9em; }} }}
.panel-title, .panel-head {{ padding-bottom: 10px !important; border-bottom: 1px solid {LINE}; }}
.panel-head .panel-title {{ padding-bottom: 0 !important; border-bottom: 0; }}
.panel-title p {{ margin: 0; font-size: 13px; font-weight: 600; color: {TEXT}; }}
.panel-title span.hint {{ font-weight: 400; color: {MUTED}; margin-left: 10px; }}

/* accordion disclosure: a drawn chevron instead of a rotated text triangle */
.label-wrap .icon {{ font-size: 0 !important; width: 7px; height: 7px; margin-right: 4px; transform: none !important;
  border-right: 1.5px solid {MUTED}; border-bottom: 1.5px solid {MUTED}; rotate: -45deg;
  transition: rotate .2s cubic-bezier(.2, .8, .2, 1); }}
.label-wrap.open .icon {{ rotate: 45deg; }}

button.primary:disabled, button.secondary:disabled {{ background: {INPUT}; color: {PLACEHOLDER};
  border-color: {LINE}; opacity: 1; cursor: not-allowed; }}
#song .upload-container {{ height: 120px !important; }}
#song .audio-container {{ height: auto !important; }}

/* program monitor: a 16:9 dark well with viewfinder corner marks, readout strip attached below */
#monitor-col {{ gap: 0 !important; }}
#monitor {{ position: relative; aspect-ratio: 16 / 9; height: auto !important; background: {WELL};
  border-radius: 6px 6px 0 0; }}
#monitor::after {{ content: ""; position: absolute; inset: 12px; pointer-events: none; z-index: 5; opacity: .85;
  background: {_corner} top left / 18px 1.5px no-repeat, {_corner} top left / 1.5px 18px no-repeat,
    {_corner} top right / 18px 1.5px no-repeat, {_corner} top right / 1.5px 18px no-repeat,
    {_corner} bottom left / 18px 1.5px no-repeat, {_corner} bottom left / 1.5px 18px no-repeat,
    {_corner} bottom right / 18px 1.5px no-repeat, {_corner} bottom right / 1.5px 18px no-repeat; }}

/* the readout strip under the monitor */
#readout {{ margin-top: -1px; background: none !important; border: 0 !important; }}
#readout .html-container {{ padding: 0 !important; }}
.readout {{ display: flex; flex-wrap: wrap; gap: 6px 32px; padding: 11px 16px; border: 1px solid {LINE};
  background: {PANEL}; }}
/* the shot inspector closes the instrument stack: strip, then inspector, one frame */
#inspector {{ margin-top: -1px; padding: 10px 16px 4px !important; border: 1px solid {LINE}; border-top: 0;
  border-radius: 0 0 6px 6px; background: {PANEL}; font-size: 13px; line-height: 1.5; }}
#inspector p {{ margin: 0 0 6px; }}
#inspector em {{ font-style: normal; color: {MUTED}; }}
.readout span {{ display: inline-flex; align-items: baseline; gap: 9px; white-space: nowrap; }}
.readout em {{ font-style: normal; font-size: 12px; color: {MUTED}; }}
.readout b {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 500; font-size: 15px;
  color: {TEXT}; }}
.readout b.lit {{ color: {ACCENT}; }}
.readout b.word {{ font-family: var(--font); }}
.readout.idle b {{ color: {PLACEHOLDER}; }}
/* the one authored motion: readout values settle in when a cut lands */
.readout:not(.idle) b {{ animation: settle .45s cubic-bezier(.16, 1, .3, 1) both; }}
@keyframes settle {{ from {{ opacity: .25; transform: translateY(3px); }} }}
@media (prefers-reduced-motion: reduce) {{ .readout b, .label-wrap .icon {{ animation: none; transition: none; }} }}
@media (max-width: 480px) {{ .readout {{ display: grid; grid-template-columns: 1fr 1fr; }} }}

#editor-note {{ font-size: 14px; line-height: 1.5; }}
.muted p {{ color: {MUTED}; }}

/* timeline: one horizontal strip of shots, mono timecodes */
#timeline .grid-wrap {{ overflow-x: auto; overflow-y: hidden; }}
#timeline .grid-container {{ grid-auto-flow: column !important; grid-template-columns: none !important;
  grid-template-rows: auto !important; grid-auto-columns: 192px !important; height: auto !important; }}
#timeline .thumbnail-item {{ aspect-ratio: 16 / 9; height: auto !important; }}
#timeline .caption-label {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 11px; }}
#timeline button.selected, #timeline .thumbnail-item.selected {{ box-shadow: 0 0 0 2px {ACCENT} !important; }}
/* pin/exclude marks on the thumbnail itself, keyed off the caption the app writes into alt */
#timeline .thumbnail-item:has(img[alt$="· pinned"])::before {{ content: ""; position: absolute; inset: 0 0 auto 0;
  height: 3px; background: {ACCENT}; z-index: 2; }}
#timeline img[alt$="· out"] {{ opacity: .35; filter: grayscale(1); }}
"""
