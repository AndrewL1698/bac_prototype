"""Gradio app: upload media + song + brief -> beat-synced cut; notes/recut, storyboard, under-the-hood, compare.

    python -m montage.app.gradio_app --share [--port 7860]
On Colab start it through the job runner: scripts/run_job.py --name app -- python -m montage.app.gradio_app --share
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from montage import paths
from montage.log import get_logger
from montage.types import UserError

log = get_logger("app")
BRIEF_PLACEHOLDER = ("Warm, nostalgic summer recap. Open slow on the beach at sunrise, build through the road trip, "
                     "save the party for the drop, end on the sunset. Skip anything blurry.")
MEDIA_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".mp4", ".mov", ".m4v", ".avi", ".mkv"}
SONG_EXT = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}


def demo_albums() -> list[str]:
    d = paths.data_root() / "demo"
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name not in ("songs", "backups")
                  and any(f.suffix.lower() in MEDIA_EXT for f in p.iterdir()))


def demo_songs() -> list[str]:
    d = paths.data_root() / "demo" / "songs"
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in SONG_EXT) if d.exists() else []


def status_line(res, edl, st) -> str:
    fb = set(res.fallbacks)
    director = "Claude" if edl.plan.source == "llm" else "rules"
    emotion = "heuristic" if "emotion:heuristic" in fb else "MERT"
    imp = "heuristic" if "importance:heuristic" in fb else "learned+heuristic"
    beats = "librosa" if "beats:librosa" in fb else "Beat This!"
    return (f"Director: {director} | Emotion: {emotion} | Importance: {imp} | Beats: {beats} | "
            f"{len(edl.shots)} shots | {st.music.tempo:.0f} BPM")


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
    import gradio as gr

    from montage import pipeline
    from montage.app.storyboard import gallery_items

    def outputs_for(res, ui):
        edl = pipeline.load_edl(res.run_id)
        s = pipeline.session_for_run(res.run_id)
        st = pipeline.state(s)
        rd = paths.run_dir(res.run_id)
        cands = {c.id: c for c in st.media.candidates}
        ui = dict(ui or {})
        ui.update(session_id=s.session_id, run_id=res.run_id, selected=None,
                  pins=list(s.pins), excluded=list(s.excluded))
        timings = [[k, round(v, 2)] for k, v in res.timings.items()]
        notes = "\n".join(f"- {n}" for n in s.notes_history) or "(no notes yet)"
        warn = ("\n\n" + "\n".join(f"> {n}" for n in edl.notes)) if edl.notes else ""
        return (res.video_path, status_line(res, edl, st), res.editor_note + warn, plan_markdown(edl.plan),
                json.dumps(json.loads(edl.plan.model_dump_json()), indent=1), gallery_items(edl, cands),
                str(rd / "curves.png") if (rd / "curves.png").exists() else None,
                str(rd / "costs.png") if (rd / "costs.png").exists() else None, timings, notes, ui)

    def friendly(fn):
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
        wrapped.__name__ = fn.__name__
        return wrapped

    @friendly
    def on_generate(files, album, song_file, song_choice, brief, seed, target_len, excerpt_start, thumbs, system, ui,
                    progress=gr.Progress()):
        from montage import config
        media = [str(paths.data_root() / "demo" / album)] if album else [f if isinstance(f, str) else f.name
                                                                          for f in (files or [])]
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
        return pipeline.export(ui["run_id"], progress=lambda f, m: progress(f, desc=m))

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
            return "", ui
        edl = pipeline.load_edl(ui["run_id"])
        shot = edl.shots[evt.index]
        ui["selected"] = shot.candidate_id
        sc = shot.scores
        txt = (f"**{shot.asset_id}** ({shot.kind}, {shot.framing}) at {shot.start:.1f}-{shot.end:.1f}s\n\n"
               f"**Why:** {shot.reason}\n\nimportance {sc.importance:.2f} | mood cost {sc.mood:.2f} | "
               f"director {sc.director:+.2f} | transition cost {sc.pairwise_in:.2f}\n\n"
               f"Pinned: {', '.join(ui.get('pins', [])) or '-'} | Excluded: {', '.join(ui.get('excluded', [])) or '-'}")
        return txt, ui

    def toggle(kind):
        def fn(ui):
            ui = dict(ui or {})
            c = ui.get("selected")
            if not c:
                return "Select a storyboard shot first.", ui
            lst = list(ui.get(kind, []))
            other = "excluded" if kind == "pins" else "pins"
            if c in lst:
                lst.remove(c)
            else:
                lst.append(c)
                ui[other] = [x for x in ui.get(other, []) if x != c]
            ui[kind] = lst
            return (f"Pinned: {', '.join(ui.get('pins', [])) or '-'} | Excluded: {', '.join(ui.get('excluded', [])) or '-'}"
                    " (applies on the next Recut)"), ui
        return fn

    with gr.Blocks(title="Montage: AI music video editor") as demo:
        gr.Markdown("# Montage\nYou direct, the music sets the timing, the system does the rest.")
        ui = gr.State({})
        with gr.Row():
            with gr.Column(scale=4):
                files = gr.File(label="Photos and videos", file_count="multiple")
                album = gr.Dropdown(choices=[""] + demo_albums(), value="", label="...or a demo album")
                song_file = gr.Audio(label="Song", type="filepath")
                song_choice = gr.Dropdown(choices=[""] + demo_songs(), value="", label="...or a demo song")
                brief = gr.Textbox(label="Director's brief", lines=4, placeholder=BRIEF_PLACEHOLDER)
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    target_len = gr.Number(label="Target length (0 = auto, 45-60)", value=0, precision=0)
                    excerpt_start = gr.Textbox(label="Excerpt start override (seconds, blank = auto)", value="")
                    thumbs = gr.Checkbox(label="Let the director see thumbnails", value=False)
                    system = gr.Dropdown(choices=["full", "naive", "beat_random", "greedy"], value="full", label="System")
                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary")
                    exp_btn = gr.Button("Export 1080p")
                export_file = gr.File(label="1080p export")
            with gr.Column(scale=6):
                video = gr.Video(label="Preview")
                status = gr.Markdown("")
                note_md = gr.Markdown("")
                plan_md = gr.Markdown("")
                with gr.Accordion("Director's plan (JSON)", open=False):
                    plan_json = gr.Code(language="json")
        with gr.Row():
            with gr.Column(scale=4):
                note_box = gr.Textbox(label="Notes for the editor", placeholder="slower at the start, more of the beach")
                recut_btn = gr.Button("Recut", variant="primary")
                notes_md = gr.Markdown("(no notes yet)")
            with gr.Column(scale=6):
                gallery = gr.Gallery(label="Storyboard", columns=8, height=260, allow_preview=False)
                shot_md = gr.Markdown("")
                with gr.Row():
                    pin_btn = gr.Button("Pin shot")
                    excl_btn = gr.Button("Exclude shot")
        with gr.Tab("Under the hood"):
            curves = gr.Image(label="Music curves and cuts", type="filepath")
            costs = gr.Image(label="Per-shot score components", type="filepath")
            timings = gr.Dataframe(headers=["stage", "seconds"], label="Stage timings")
        with gr.Tab("Compare"):
            naive_btn = gr.Button("Render naive cut")
            with gr.Row():
                naive_video = gr.Video(label="Naive cut (random shots, fixed 2 s cuts)")
                cur_video = gr.Video(label="Current cut")
        with gr.Tab("Rate"):
            gr.Markdown("Blind A/B voting arrives with the demo tools (phase 11).")

        outs = [video, status, note_md, plan_md, plan_json, gallery, curves, costs, timings, notes_md, ui]
        gen_btn.click(on_generate, [files, album, song_file, song_choice, brief, seed, target_len, excerpt_start,
                                    thumbs, system, ui], outs, api_name="generate")
        recut_btn.click(on_recut, [note_box, ui], outs, api_name="recut")
        exp_btn.click(on_export, [ui], [export_file], api_name="export")
        naive_btn.click(on_naive, [ui], [naive_video, cur_video], api_name="naive")
        gallery.select(on_select, [ui], [shot_md, ui])
        pin_btn.click(toggle("pins"), [ui], [shot_md, ui])
        excl_btn.click(toggle("excluded"), [ui], [shot_md, ui])
        video.change(lambda v, u: {**(u or {}), "current_video": v}, [video, ui], [ui])
    return demo


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
    allowed = [str(paths.data_root()), str(paths.scratch()), str(Path("/content").resolve())]
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port, prevent_thread_lock=True,
                allowed_paths=allowed, show_error=True)
    url = getattr(demo, "share_url", None) or getattr(demo, "local_url", None) or f"http://127.0.0.1:{args.port}"
    paths.atomic_write(paths.jobs_dir() / "app.url", url + "\n")
    print(f"APP URL: {url}", flush=True)
    demo.block_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
