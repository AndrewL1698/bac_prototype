"""Montage look (Print Table): neutral black table, white instant-print frames, one signal red. Dark in every mode."""

import inspect

import gradio as gr

# contrast checked: INK on ACCENT 5.0:1, MUTED on PANEL 6.7:1, PLACEHOLDER on INPUT 5.2:1, PRINT_INK on PRINT 17.9:1
GROUND, PANEL, WELL, INPUT = "#121212", "#1a1a1a", "#0a0a0a", "#232323"
LINE, LINE_STRONG = "#2b2b2b", "#3c3c3c"
TEXT, MUTED, PLACEHOLDER = "#f2f2f2", "#9ba1a9", "#8f959e"
ACCENT, ACCENT_HOVER, ACCENT_SOFT, INK = "#d22f22", "#e5402f", "rgba(210, 47, 34, 0.16)", "#ffffff"
PRINT, PRINT_INK = "#fafafa", "#121212"  # the white print frame and the ink written on its margin

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
    button_primary_border_color_hover=ACCENT_HOVER,
    error_background_fill=ACCENT_SOFT, error_border_color=ACCENT, error_text_color=TEXT, error_icon_color=ACCENT,
)
# same values for light and dark: the app is dark whatever the visitor's OS says
_has_dark = inspect.signature(gr.themes.Base.set).parameters
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red, neutral_hue=gr.themes.colors.neutral,  # Base defaults leak blue into hovers
    font=[gr.themes.GoogleFont("Archivo"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Archivo"), "system-ui", "sans-serif"],  # numbers use tabular figures, not a mono
    radius_size=gr.themes.sizes.radius_sm,
).set(block_shadow="none", input_shadow="none", input_border_width="1px", **_vars,
      **{k + "_dark": v for k, v in _vars.items() if k + "_dark" in _has_dark})

CSS = f"""
:root {{ color-scheme: dark; }}
.gradio-container {{ padding: 0 20px 28px !important; }}
::selection {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
input, textarea {{ caret-color: {ACCENT}; }}
* {{ scrollbar-color: {LINE_STRONG} transparent; scrollbar-width: thin; }}
button:focus-visible, [role="radio"]:focus-visible, input:focus-visible, textarea:focus-visible {{
  outline: 2px solid {ACCENT}; outline-offset: 2px; }}

#topbar {{ align-items: center; padding: 8px 0; margin-bottom: 4px; border-bottom: 1px solid {LINE}; }}
@media (max-width: 480px) {{ #topbar {{ flex-direction: column; align-items: flex-start !important; }} }}
footer {{ display: none !important; }}  /* Gradio's footer brings an off-palette logo */
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
.panel-title {{ padding-bottom: 10px !important; border-bottom: 1px solid {LINE}; }}
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

/* program monitor: a 16:9 dark screen, readout strip attached below */
#monitor-col {{ gap: 0 !important; }}
#monitor {{ position: relative; aspect-ratio: 16 / 9; height: auto !important; background: {WELL};
  border: 0 !important; border-radius: 6px 6px 0 0; }}

/* the readout strip: large figures, small quiet labels (hierarchy by scale contrast) */
#readout {{ margin-top: -1px; background: none !important; border: 0 !important; }}
#readout .html-container {{ padding: 0 !important; }}
.readout {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 32px; padding: 11px 16px; border: 1px solid {LINE};
  border-radius: 0 0 6px 6px; background: {PANEL}; }}
.readout span {{ display: inline-flex; align-items: baseline; gap: 9px; white-space: nowrap; }}
.readout em {{ font-style: normal; font-size: 12px; color: {MUTED}; }}
.readout b {{ font-variant-numeric: tabular-nums; font-weight: 800; font-size: 28px; letter-spacing: -.02em;
  color: {TEXT}; }}
.readout b.lit {{ color: {ACCENT}; }}
.readout b.word {{ font-size: 16px; font-weight: 600; letter-spacing: 0; }}
.readout.idle b {{ color: {PLACEHOLDER}; }}
/* the one authored motion: readout values settle in when a cut lands */
.readout:not(.idle) b {{ animation: settle .45s cubic-bezier(.16, 1, .3, 1) both; }}
@keyframes settle {{ from {{ opacity: .25; transform: translateY(3px); }} }}
@media (prefers-reduced-motion: reduce) {{ .readout b, .label-wrap .icon {{ animation: none; transition: none; }} }}
@media (max-width: 480px) {{ .readout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; }}
  .readout b {{ font-size: 22px; }} }}

#editor-note {{ font-size: 14px; line-height: 1.5; }}
.muted p {{ color: {MUTED}; }}

/* timeline: one horizontal row of white instant prints; each print's wide bottom margin carries its timecode
   and a fixed state cell (hollow open, red kept, struck removed), keyed off the caption the app writes into alt */
#timeline .grid-wrap {{ overflow-x: auto; overflow-y: hidden; }}
#strip .empty .icon {{ display: none; }}
#strip .empty::after {{ content: "Your shots appear here as prints after you Generate."; color: {MUTED}; font-size: 13px; }}
#timeline .grid-container {{ grid-auto-flow: column !important; grid-template-columns: none !important;
  grid-template-rows: auto !important; grid-auto-columns: 168px !important; height: auto !important; }}
#timeline .thumbnail-item {{ aspect-ratio: 16 / 9; height: auto !important; background: {PRINT} !important;
  padding: 5px 5px 24px !important; border-radius: 2px !important; box-shadow: none !important; overflow: visible !important; }}
#timeline .thumbnail-item img {{ border-radius: 0 !important; }}
#timeline .caption-label {{ left: 6px !important; right: 6px !important; bottom: 4px !important; background: none !important;
  border: 0 !important; padding: 0 !important; color: {PRINT_INK} !important; font-size: 11px; font-weight: 600;
  font-variant-numeric: tabular-nums; display: flex; align-items: center; gap: 6px; }}
#timeline .caption-label::before {{ content: ""; flex: 0 0 9px; height: 9px; border: 1.5px solid {PRINT_INK}; }}
#timeline .thumbnail-item:has(img[alt$="kept"]) .caption-label::before {{ background: {ACCENT}; border-color: {ACCENT}; }}
#timeline .thumbnail-item:has(img[alt$="out"]) .caption-label::before {{
  background: linear-gradient(45deg, transparent 42%, {PRINT_INK} 42% 58%, transparent 58%); }}
#timeline .thumbnail-item:has(img[alt$="out"]) img {{ opacity: .6; filter: grayscale(1); }}
#timeline button.selected, #timeline .thumbnail-item.selected {{ box-shadow: 0 0 0 2px {ACCENT} !important; translate: 0 -4px; }}

/* the Revise dock under the strip: selected shot (shown as a print), note, changes + the one filled action */
#revise {{ gap: 20px !important; padding-top: 12px; border-top: 1px solid {LINE}; }}
.signpost p, #note-box [data-testid="block-info"] {{ margin: 0; font-size: 11px; font-weight: 500;
  letter-spacing: .06em; text-transform: uppercase; color: {MUTED}; }}
#shot-card {{ aspect-ratio: 16 / 9; height: auto !important; background: {PRINT} !important; border: 0 !important;
  border-radius: 2px; padding: 6px 6px 22px !important; box-sizing: content-box; max-width: 180px; }}
#shot-card:not(:has(img)) {{ background: transparent !important; border: 1px solid {LINE_STRONG} !important; }}
#shot-card img {{ object-fit: cover; width: 100%; height: 100%; }}
#shot-card .icon-button-wrapper {{ display: none; }}  /* empty toolbar Gradio keeps even with buttons=[] */
#shot-why, #changes {{ font-size: 13px; line-height: 1.5; }}
#shot-why p, #changes p {{ margin: 0 0 6px; }}
#shot-why em {{ font-style: normal; color: {MUTED}; }}
#changes .html-container {{ padding: 0 !important; }}
#changes .idle {{ color: {MUTED}; }}
#changes ul {{ margin: 0 0 8px; padding-left: 16px; }}
#changes li::marker {{ color: {ACCENT}; }}
"""
