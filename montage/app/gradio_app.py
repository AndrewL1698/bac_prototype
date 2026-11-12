"""Gradio app: upload media + song + brief -> beat-synced cut; notes/recut, storyboard, under-the-hood, compare.

    python -m montage.app.gradio_app --share [--port 7860]
On Colab start it through the job runner: scripts/run_job.py --name app -- python -m montage.app.gradio_app --share
"""

import argparse
import functools
import html
import json
import os
import traceback

import gradio as gr

from montage import paths
from montage.app.demo_mode import demo_albums, demo_songs  # noqa: F401  (also re-exported for callers)
from montage.app.theme import CSS, THEME
from montage.log import get_logger
from montage.types import UserError

log = get_logger("app")
BRIEF_PLACEHOLDER = ("Warm, nostalgic summer recap. Open slow on the beach at sunrise, build through the road trip, "
                     "save the party for the drop, end on the sunset. Skip anything blurry.")
SHOT_IDLE = "*No shot selected. Click a shot in the timeline above.*"  # theme.py mutes em in the Revise panel
NO_CHANGES = "No changes yet. Keep or remove shots, or write a note."

def status_line(res, edl, st) -> str:
    fb = set(res.fallbacks)
    director = "Claude" if edl.plan.source == "llm" else "rules"
    emotion = "heuristic" if "emotion:heuristic" in fb else "MERT"
    imp = "heuristic" if "importance:heuristic" in fb else "learned+heuristic"
    beats = "librosa" if "beats:librosa" in fb else "Beat This!"
    return (f"Director: {director} | Emotion: {emotion} | Importance: {imp} | Beats: {beats} | "
            f"{len(edl.shots)} shots | {st.music.tempo:.0f} BPM")


def readout_html(tempo="---", shots="--", length="-:--", director="--", idle=True) -> str:
    cells = [("Tempo", f"{tempo} BPM", "lit"), ("Shots", shots, ""), ("Length", length, ""),
             ("Director", director, "word")]
    return (f'<div class="readout{" idle" if idle else ""}">'
            + "".join(f'<span><em>{k}</em><b class="{c}">{v}</b></span>' for k, v, c in cells) + "</div>")


def plan_markdown(plan, profile=None) -> str:
    lines = [f"**Vibe:** {plan.vibe or '(none)'}", f"**Rationale:** {plan.rationale}"]
    if plan.vibe_queries:
        lines.append("**Looking for:** " + ", ".join(f"`{q}`" for q in plan.vibe_queries))
    if plan.avoid_queries:
        lines.append("**Avoiding:** " + ", ".join(f"`{q}`" for q in plan.avoid_queries))
    if plan.opening_query:
        lines.append(f"**Open on:** `{plan.opening_query}`")
    if plan.closing_query:
        lines.append(f"**End on:** `{plan.closing_query}`")
    pace = "faster" if plan.pacing > 0.05 else "slower" if plan.pacing < -0.05 else "follows the music"
    lines.append(f"**Pacing:** {pace} ({plan.pacing:+.1f}); **energy bias:** {plan.energy_bias:+.1f}; "
                 f"**chronological:** {'yes' if plan.chronological else 'no'}")
    for s in plan.segments:
        where = f"section {s.section_index}" if s.anchor == "section" else s.anchor
        q = ", ".join(f"`{x}`" for x in s.content_queries) or "-"
        lines.append(f"- **{where}:** {q}; mood {s.mood}; pacing {s.pacing:+.1f}")
    if plan.exclude:
        lines.append("**Excluded:** " + ", ".join(plan.exclude))
    if plan.must_include:
        lines.append("**Must include:** " + ", ".join(plan.must_include))
    return "\n\n".join(lines)


def build_app():
    from montage import pipeline
    from montage.app.storyboard import gallery_items
    from montage.music.profile import mmss

    def outputs_for(res, ui):
        edl = pipeline.load_edl(res.run_id)
        s = pipeline.session_for_run(res.run_id)
        st = pipeline.state(s)
        rd = paths.run_dir(res.run_id)
        cands = {c.id: c for c in st.media.candidates}
        shots = [sh for sh in edl.shots if sh.candidate_id in cands]
        board = [[t, cap, sh.candidate_id] for (t, cap), sh in zip(gallery_items(edl, cands), shots)]
        ui = dict(ui or {})
        # labels outlive the cut: an excluded shot drops out of the next EDL but should still read as "0:12 a003"
        labels = {**ui.get("labels", {}), **{c: " ".join(cap.split()[:2]) for _, cap, c in board}}
        # applied_* = what the last cut used; the Changes list is the difference to what the user has set since
        ui.update(session_id=s.session_id, run_id=res.run_id, selected=None, current_video=res.video_path,
                  pins=list(s.pins), excluded=list(s.excluded), applied_pins=list(s.pins),
                  applied_excluded=list(s.excluded), board=board, labels=labels)
        timings = [[k, round(v, 2)] for k, v in res.timings.items()]
        notes = "\n".join(f"- {n}" for n in s.notes_history) or "(no notes yet)"
        warn = ("\n\n" + "\n".join(f"> {n}" for n in edl.notes)) if edl.notes else ""
        return (res.video_path, status_line(res, edl, st), res.editor_note + warn, plan_markdown(edl.plan),
                json.dumps(json.loads(edl.plan.model_dump_json()), indent=1), marked(ui),
                str(rd / "curves.png") if (rd / "curves.png").exists() else None,
                str(rd / "costs.png") if (rd / "costs.png").exists() else None, timings, notes, ui,
                recut_update(ui), gr.update(interactive=True), gr.update(interactive=True), *pin_buttons(ui),
                SHOT_IDLE,
                readout_html(f"{st.music.tempo:.0f}", len(edl.shots), mmss(edl.shots[-1].end) if edl.shots else "-:--",
                             "Claude" if edl.plan.source == "llm" else "Built-in", idle=False),
                gr.update(variant="secondary"),  # once a cut exists, Recut is the primary action
                None, changes_html(ui), "", restore_update(ui))  # shot frame, Changes, note box (sent), put-back

    def marked(ui):
        # caption = "0:04 a002" (+ " · kept" / " · out"): the timecode always fits; theme.py keys marks off the suffix
        pins, excl = ui.get("pins", []), ui.get("excluded", [])
        return [(t, ui["labels"][c] + (" · kept" if c in pins else " · out" if c in excl else ""))
                for t, _, c in ui.get("board", [])]

    def changes(ui):
        lab = lambda c: ui.get("labels", {}).get(c, c)  # noqa: E731
        p, e = ui.get("pins", []), ui.get("excluded", [])
        ap, ae = ui.get("applied_pins", []), ui.get("applied_excluded", [])
        return ([f"Keep {lab(c)}" for c in p if c not in ap] + [f"Stop keeping {lab(c)}" for c in ap if c not in p]
                + [f"Remove {lab(c)}" for c in e if c not in ae] + [f"Put back {lab(c)}" for c in ae if c not in e])

    def changes_html(ui):
        # a live region so screen readers hear each Keep/Remove; "Removed earlier" is the undo path after a recut
        rows, lab = changes(ui), ui.get("labels", {})
        gone = [lab.get(c, c) for c in ui.get("applied_excluded", []) if c in ui.get("excluded", [])]
        body = (f"<ul>{''.join(f'<li>{html.escape(r)}</li>' for r in rows)}</ul>" if rows
                else f'<p class="idle">{NO_CHANGES}</p>')
        if gone:
            body += f'<p class="idle">Removed earlier: {html.escape(", ".join(gone))}</p>'
        return f'<div role="status" aria-live="polite">{body}</div>'

    def restore_update(ui):
        return gr.update(interactive=bool(ui.get("excluded")))

    def recut_update(ui):
        n = len(changes(ui))
        return gr.update(value=f"Recut · {n} change{'s' * (n > 1)}" if n else "Recut", interactive=True)

    def pin_buttons(ui):
        c = ui.get("selected")
        return (gr.update(value="Stop keeping" if c in ui.get("pins", []) else "Keep", interactive=bool(c)),
                gr.update(value="Put back" if c in ui.get("excluded", []) else "Remove", interactive=bool(c)))

    def friendly(fn):
        @functools.wraps(fn)  # keeps the signature so Gradio still injects gr.Progress
        def wrapped(*a, **k):
            try:
                return fn(*a, **k)
            except UserError as e:
                raise gr.Error(str(e), title="Can't make that video")
            except gr.Error:
                raise
            except Exception as e:  # noqa: BLE001
                log.error("unexpected error: %s\n%s", e, traceback.format_exc())
                raise gr.Error(f"Something went wrong ({type(e).__name__}). See the app log.")
        return wrapped

    def media_list(files, album):
        return [str(paths.data_root() / "demo" / album)] if album else [f if isinstance(f, str) else f.name
                                                                         for f in (files or [])]

    @friendly
    def on_generate(files, album, song_file, song_choice, brief, seed, target_len, excerpt_start, thumbs, system, ui,
                    progress=gr.Progress()):
        from montage import config
        media = media_list(files, album)
        song = str(paths.data_root() / "demo" / "songs" / song_choice) if song_choice else song_file
        if not media:
            raise UserError("Add photos and videos, or pick a demo album.")
        if not song:
            raise UserError("Add a song, or pick a demo song.")
        over = {}
        if target_len:
            over["target_length_s"] = float(min(60, max(45, target_len)))
        if excerpt_start not in (None, ""):
            over.update(excerpt_hint="custom", excerpt_start_s=float(excerpt_start))
        with config.override({"director.send_thumbnails": bool(thumbs)}):
            res = pipeline.generate(media, song, brief or "", seed=int(seed or 0), system=system or "full",
                                    progress=lambda f, m: progress(f, desc=m), overrides=over or None)
        return outputs_for(res, ui)

    @friendly
    def on_recut(note, ui, progress=gr.Progress()):
        ui = ui or {}
        if not ui.get("session_id"):
            raise UserError("Generate a cut first.")
        res = pipeline.recut(ui["session_id"], note=note or None, pins=ui.get("pins"), excluded=ui.get("excluded"),
                             progress=lambda f, m: progress(f, desc=m))
        return outputs_for(res, ui)

    @friendly
    def on_export(ui, progress=gr.Progress()):
        if not (ui or {}).get("run_id"):
            raise UserError("Generate a cut first.")
        return gr.update(value=pipeline.export(ui["run_id"], progress=lambda f, m: progress(f, desc=m)), visible=True)

    @friendly
    def on_naive(ui, progress=gr.Progress()):
        if not (ui or {}).get("session_id"):
            raise UserError("Generate a cut first.")
        s = pipeline.load_session(ui["session_id"])
        progress(0.3, desc="planning the naive cut")
        edl = pipeline.plan_edit(s, s.plan_history[-1], seed=0, system="naive")
        progress(0.6, desc="rendering")
        return pipeline.render_edl(edl), ui.get("current_video")

    def on_select(ui, evt: gr.SelectData):
        ui = dict(ui or {})
        if not ui.get("run_id"):
            return SHOT_IDLE, ui, *pin_buttons(ui), None
        thumb, _, cid = ui["board"][evt.index]
        shot = next(sh for sh in pipeline.load_edl(ui["run_id"]).shots if sh.candidate_id == cid)
        ui.update(selected=cid, selected_idx=evt.index)
        # engine scores stay in the Lab's score chart; the card says it in words
        return (f"**{mmss(shot.start)}** {shot.asset_id} · {shot.kind}  \nWhy: {shot.reason}", ui, *pin_buttons(ui),
                thumb)

    def toggle(kind):
        def fn(ui):
            ui = dict(ui or {})
            c = ui.get("selected")
            if not c:
                return gr.update(), ui, gr.update(), *pin_buttons(ui), gr.update(), gr.update()
            lst = list(ui.get(kind, []))
            other = "excluded" if kind == "pins" else "pins"
            if c in lst:
                lst.remove(c)
            else:
                lst.append(c)
                ui[other] = [x for x in ui.get(other, []) if x != c]
            ui[kind] = lst
            return revised(ui)
        return fn

    def revised(ui):
        return (changes_html(ui), ui, gr.update(value=marked(ui), selected_index=ui.get("selected_idx")),
                *pin_buttons(ui), recut_update(ui), restore_update(ui))

    def put_back_all(ui):
        ui = dict(ui or {})
        ui["excluded"] = []
        return revised(ui)

    def pick_source(src):
        # clear the hidden input so it can never silently win in on_generate
        if not src.startswith("Demo"):
            return gr.update(visible=True), gr.update(visible=False, value="")
        return gr.update(visible=False, value=None), gr.update(visible=True)

    def on_load_pair(rs):
        import random
        from montage.app.demo_mode import demo_root, load_pairs
        pairs = load_pairs()
        if not pairs:
            raise gr.Error("No vote pairs yet: run demo_mode --prepare first.", title="Nothing to rate")
        rs = dict(rs or {})
        k = rs.get("next", 0) % len(pairs)
        pr = pairs[k]
        a = (str(demo_root() / "vote" / pr["a"]), pr["a_system"])
        b = (str(demo_root() / "vote" / pr["b"]), pr["b_system"])
        left, right = (a, b) if random.random() < 0.5 else (b, a)
        rs.update(next=k + 1, pair_id=pr["pair_id"], left_system=left[1], right_system=right[1], voted=False)
        return left[0], right[0], "Watch both, then vote.", rs

    def vote_handler(choice):
        def fn(name, rs):
            from montage.eval.vote import analyze, record_vote, summary_line
            rs = dict(rs or {})
            if not rs.get("pair_id"):
                return "Load a pair first.", rs
            if rs.get("voted"):
                return "Already counted. Load another pair to vote again.", rs
            record_vote(name, rs["pair_id"], rs["left_system"], rs["right_system"], choice)
            rs["voted"] = True
            return f"Thanks! Votes so far: {summary_line(analyze())}", rs
        return fn

    # ---- Pick a song for me (phase 12): Spotify sign-in -> suggestions -> opt-in YouTube audio ----
    from montage.app import songpick_ui

    @friendly
    def on_sp_connect(page_url):
        return songpick_ui.connect(page_url)

    def on_sp_load(sp, request: gr.Request):
        new, msg = songpick_ui.on_page_load(dict(request.query_params) if request else {})
        return (new if new is not None else sp), msg

    @friendly
    def on_sp_paste(text):
        return songpick_ui.paste_code(text)

    @friendly
    def on_sp_suggest(files, album, hist_files, sp, mode, progress=gr.Progress()):
        progress(0.02, desc="reading your photos")
        hist = [f if isinstance(f, str) else f.name for f in (hist_files or [])]
        choices, pick, md = songpick_ui.suggest(media_list(files, album), hist or None, sp, mode,
                                                progress=lambda f, m: progress(f, desc=m))
        return gr.Radio(choices=choices, value=choices[0][1] if choices else None), pick, md

    @friendly
    def on_sp_use(choice, pick, yt_ok, progress=gr.Progress()):
        progress(0.2, desc="finding the song on YouTube")
        path, md = songpick_ui.use_song(choice, pick, bool(yt_ok))
        if path is None:  # fetch off or blocked: keep the Song box as it is, show the manual route
            return gr.update(), gr.update(), md, gr.update()
        return path, "", md, "Upload song"  # show the Song box the picked song lands in

    with gr.Blocks(title="Montage: AI music video editor", fill_width=True) as demo:
        ui = gr.State({})
        albums, songs = demo_albums(), demo_songs()
        with gr.Row(elem_id="topbar", equal_height=True):
            gr.Markdown("# Montage\nYou direct, the music sets the timing, the system does the rest.",
                        elem_id="wordmark")
            exp_btn = gr.Button("Export 1080p", interactive=False, scale=0, min_width=150)
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=3, min_width=260, elem_classes="ws-panel"):
                gr.Markdown("Media", elem_classes="panel-title")
                # the source radios only show when demo media exists; they keep exactly one input live
                media_src = gr.Radio(["Upload media", "Demo album"], value="Upload media", show_label=False,
                                     container=False, visible=bool(albums))
                files = gr.File(label="Photos and videos", file_count="multiple", height=140)
                album = gr.Dropdown(choices=[""] + albums, value="", label="Demo album", visible=False)
                song_src = gr.Radio(["Upload song", "Demo song"], value="Upload song", show_label=False,
                                    container=False, visible=bool(songs))
                song_file = gr.Audio(label="Song", type="filepath", elem_id="song")
                song_choice = gr.Dropdown(choices=[""] + songs, value="", label="Demo song", visible=False)
                sp_state = gr.State({})
                sp_pick = gr.State(None)
                with gr.Accordion("Pick a song for me (Spotify)", open=False):
                    gr.Markdown("Connect Spotify first (read-only: recent plays and top songs), then add photos, "
                                "choose how to pick, and press **Suggest songs**. Connecting reloads this page.")
                    from montage.config import get_config as _cfg
                    sp_mode = gr.Radio(label="How to pick", choices=songpick_ui.MODE_CHOICES,
                                       value=_cfg().songpick.default_mode)
                    with gr.Row():
                        sp_connect = gr.Button("Connect Spotify")
                        sp_suggest = gr.Button("Suggest songs")
                    sp_status = gr.Markdown("")
                    sp_url_in = gr.Textbox(visible=False)
                    sp_url_out = gr.Textbox(visible=False)
                    sp_hist = gr.File(label="Your Spotify history (data export JSON files)", file_count="multiple",
                                      file_types=[".json"], visible=_cfg().songpick.default_mode == "export")
                    sp_choice = gr.Radio(label="Suggested songs", choices=[])
                    sp_md = gr.Markdown("")
                    sp_yt = gr.Checkbox(label="Fetch the song's audio from YouTube (only for songs you're allowed to use)",
                                        value=_cfg().songpick.youtube.enabled)
                    sp_use = gr.Button("Use this song")
                    with gr.Accordion("Spotify page asked you to paste a code?", open=False):
                        sp_code = gr.Textbox(label="Code from the Spotify sign-in page")
                        sp_code_btn = gr.Button("Finish connecting")
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    target_len = gr.Number(label="Target length (0 = auto, 45-60)", value=0, precision=0)
                    excerpt_start = gr.Textbox(label="Excerpt start override (seconds, blank = auto)", value="")
                    thumbs = gr.Checkbox(label="Let the director see thumbnails", value=False)
                    system = gr.Dropdown(choices=["full", "naive", "beat_random", "greedy"], value="full", label="System")
            with gr.Column(scale=6, min_width=420, elem_id="monitor-col"):
                video = gr.Video(label="Preview", show_label=False, interactive=False, elem_id="monitor")
                readout = gr.HTML(readout_html(), elem_id="readout")
                export_file = gr.File(label="1080p export", visible=False)
            with gr.Column(scale=3, min_width=280, elem_classes="ws-panel"):
                gr.Markdown("Director", elem_classes="panel-title")
                brief = gr.Textbox(label="Director's brief", lines=4, placeholder=BRIEF_PLACEHOLDER, elem_id="brief")
                gen_btn = gr.Button("Generate", variant="primary")
                note_md = gr.Markdown("Your editor's note about each cut shows up here.", elem_id="editor-note")
                notes_md = gr.Markdown("(no notes yet)", elem_classes="muted")
                with gr.Accordion("The plan", open=False):
                    plan_md = gr.Markdown("")
        # the whole revise loop in one panel, in task order: pick a shot, keep/remove it, note, recut
        with gr.Column(elem_id="timeline", elem_classes="ws-panel"):
            gr.Markdown('Revise<span class="hint">pick a shot, keep or remove it, add a note, then recut</span>',
                        elem_classes="panel-title")
            gallery = gr.Gallery(label="Storyboard", show_label=False, rows=1, height=114, object_fit="cover",
                                 allow_preview=False, buttons=[], elem_id="strip")
            with gr.Row(elem_id="revise"):
                with gr.Column(scale=4, min_width=280):
                    gr.Markdown("Selected shot", elem_classes="signpost")
                    with gr.Row():
                        shot_img = gr.Image(type="filepath", interactive=False, show_label=False, buttons=[],
                                            elem_id="shot-card", scale=1, min_width=140)
                        with gr.Column(scale=1, min_width=140):
                            shot_md = gr.Markdown(SHOT_IDLE, elem_id="shot-why")
                            with gr.Row():
                                pin_btn = gr.Button("Keep", size="sm", interactive=False, min_width=90)
                                excl_btn = gr.Button("Remove", size="sm", interactive=False, min_width=90)
                with gr.Column(scale=4, min_width=240):
                    note_box = gr.Textbox(label="Note for the editor", lines=4, elem_id="note-box",
                                          placeholder="slower at the start, more of the beach…")
                with gr.Column(scale=3, min_width=220):
                    gr.Markdown("Changes", elem_classes="signpost")
                    # Recut first: its label carries the count, so it stays above the fold however long the list gets
                    recut_btn = gr.Button("Recut", variant="primary", interactive=False)
                    changes_out = gr.HTML(changes_html({}), elem_id="changes")
                    restore_btn = gr.Button("Put back removed shots", size="sm", interactive=False)
        with gr.Accordion("Lab", open=False, elem_id="lab"):
            status = gr.Markdown("")
            with gr.Tab("Under the hood"):
                curves = gr.Image(label="Music curves and cuts", type="filepath")
                costs = gr.Image(label="Per-shot score components", type="filepath")
                timings = gr.Dataframe(headers=["stage", "seconds"], label="Stage timings")
                with gr.Accordion("Director's plan (JSON)", open=False):
                    plan_json = gr.Code(language="json")
            with gr.Tab("Compare"):
                naive_btn = gr.Button("Render naive cut", interactive=False)
                with gr.Row():
                    naive_video = gr.Video(label="Naive cut (random shots, fixed 2 s cuts)")
                    cur_video = gr.Video(label="Current cut")
            with gr.Tab("Rate"):
                gr.Markdown("Which cut is better? Two versions of the same album and song; the order is random.")
                rate_state = gr.State({})
                with gr.Row():
                    rater = gr.Textbox(label="Your name (optional)", scale=3)
                    load_pair = gr.Button("Load a pair", scale=1)
                with gr.Row():
                    left_video = gr.Video(label="Left")
                    right_video = gr.Video(label="Right")
                with gr.Row():
                    left_btn = gr.Button("Left better")
                    tie_btn = gr.Button("Tie")
                    right_btn = gr.Button("Right better")
                vote_md = gr.Markdown("")

        # first 11 outputs are the /generate and /recut API contract (tests/test_app.py indexes them); append only
        need_cut = [recut_btn, exp_btn, naive_btn]
        outs = [video, status, note_md, plan_md, plan_json, gallery, curves, costs, timings, notes_md, ui,
                *need_cut, pin_btn, excl_btn, shot_md, readout, gen_btn, shot_img, changes_out, note_box, restore_btn]
        revise_outs = [changes_out, ui, gallery, pin_btn, excl_btn, recut_btn, restore_btn]
        media_src.change(pick_source, [media_src], [files, album])
        song_src.change(pick_source, [song_src], [song_file, song_choice])
        # Drop ?spotify_code=&state= from the address bar once read: a refresh would retry a used-up code.
        demo.load(on_sp_load, [sp_state], [sp_state, sp_status]).then(
            None, None, None, js="() => { history.replaceState(null, '', location.pathname); }")
        sp_connect.click(on_sp_connect, [sp_url_in], [sp_url_out, sp_status],
                         js="(u) => window.location.href.split('?')[0].split('#')[0]"
                         ).success(None, [sp_url_out], None, js="(u) => { if (u) { window.location.href = u; } }")
        sp_code_btn.click(on_sp_paste, [sp_code], [sp_state, sp_status])
        sp_mode.change(lambda m: gr.update(visible=m == "export"), [sp_mode], [sp_hist])
        sp_suggest.click(on_sp_suggest, [files, album, sp_hist, sp_state, sp_mode], [sp_choice, sp_pick, sp_md])
        sp_use.click(on_sp_use, [sp_choice, sp_pick, sp_yt], [song_file, song_choice, sp_md, song_src])
        gen_btn.click(on_generate, [files, album, song_file, song_choice, brief, seed, target_len, excerpt_start,
                                    thumbs, system, ui], outs, api_name="generate", show_progress_on=video)
        # progress shows on the monitor only, not on every output panel (export's own output starts hidden)
        recut_btn.click(on_recut, [note_box, ui], outs, api_name="recut", show_progress_on=video)
        exp_btn.click(on_export, [ui], [export_file], api_name="export", show_progress_on=video)
        naive_btn.click(on_naive, [ui], [naive_video, cur_video], api_name="naive", show_progress_on=naive_video)
        gallery.select(on_select, [ui], [shot_md, ui, pin_btn, excl_btn, shot_img])
        pin_btn.click(toggle("pins"), [ui], revise_outs)
        excl_btn.click(toggle("excluded"), [ui], revise_outs)
        restore_btn.click(put_back_all, [ui], revise_outs)
        load_pair.click(on_load_pair, [rate_state], [left_video, right_video, vote_md, rate_state], api_name="load_pair")
        for btn, choice in ((left_btn, "left"), (tie_btn, "tie"), (right_btn, "right")):
            btn.click(vote_handler(choice), [rater, rate_state], [vote_md, rate_state], api_name=f"vote_{choice}")
    return demo


def served_paths() -> tuple[list[str], list[str]]:
    """(allowed, blocked) for demo.launch. Serve only generated outputs (videos/plots in runs/, storyboard
    thumbnails). Never the whole Drive folder or /content: /content/montage_env.sh holds the API key and the
    share link is public. Fetched YouTube songs (cache/youtube_audio) stay private: the app plays a temp copy."""
    allowed = [str(paths.data_root() / "runs"), str(paths.scratch_dir("thumbs")),
               str(paths.data_root() / "demo" / "vote"), str(paths.data_root() / "demo" / "backups")]
    blocked = [str(paths.COLAB_ENV_FILE), str(paths.data_root() / "sessions"), str(paths.data_root() / "jobs"),
               str(paths.cache_dir("youtube_audio"))]
    return allowed, blocked


def main(argv: list[str] | None = None) -> int:
    from montage.config import get_config
    paths.load_colab_env()
    cfg = get_config().app
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", default=False)
    ap.add_argument("--port", type=int, default=cfg.port)
    args = ap.parse_args(argv)
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    allowed, blocked = served_paths()
    # Optional password (Colab Secret APP_PASSWORD -> MONTAGE_APP_PASSWORD via the bootstrap). Never logged.
    pw = os.environ.get("MONTAGE_APP_PASSWORD") or None
    auth = (os.environ.get("MONTAGE_APP_USER") or "montage", pw) if pw else None
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port, prevent_thread_lock=True,
                theme=THEME, css=CSS,
                allowed_paths=allowed, blocked_paths=blocked, show_error=True, auth=auth,
                auth_message="Montage demo: sign in with the username and password you set." if auth else None)
    url = getattr(demo, "share_url", None) or getattr(demo, "local_url", None) or f"http://127.0.0.1:{args.port}"
    paths.atomic_write(paths.jobs_dir() / "app.url", url + "\n")
    print(f"APP URL: {url} (password {'on' if auth else 'off'})", flush=True)
    demo.block_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
