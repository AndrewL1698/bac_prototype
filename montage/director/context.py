"""Text context for the director (only text goes to the API by default)."""

from __future__ import annotations

from collections import Counter

from montage.music.profile import mmss, summarize
from montage.types import EDL, Asset, Candidate, MusicProfile


def song_summary(profile: MusicProfile) -> str:
    return summarize(profile)


def media_inventory(assets: list[Asset], candidates: list[Candidate], tags: dict[str, list[str]],
                    max_lines: int = 100) -> str:
    """One line per asset: `a012 video 0:24 2026-07-14 | beach, ocean, ...` (video tags = most frequent)."""
    by_asset: dict[str, list[str]] = {}
    for c in candidates:
        by_asset.setdefault(c.asset_id, []).extend(tags.get(c.id, []))
    lines = []
    for a in assets[:max_lines]:
        t = [w for w, _ in Counter(by_asset.get(a.id, [])).most_common(5)]
        dur = f" {mmss(a.duration)}" if a.kind == "video" and a.duration else ""
        date = f" {a.taken_at:%Y-%m-%d}" if a.taken_at else ""
        lines.append(f"{a.id} {a.kind}{dur}{date} | {', '.join(t)}")
    if len(assets) > max_lines:
        lines.append(f"... and {len(assets) - max_lines} more")
    return "\n".join(lines)


def edl_summary(edl: EDL, tags: dict[str, list[str]] | None = None) -> str:
    tags = tags or {}
    return "\n".join(f"{mmss(s.start)}-{mmss(s.end)} {s.asset_id} {s.kind} | {', '.join(tags.get(s.candidate_id, [])[:3])}"
                     for s in edl.shots)


def allowed_queries_hint() -> str:
    from montage.features.tags import vocabulary
    return ", ".join(vocabulary())


def contact_sheet(assets: list[Asset], candidates: list[Candidate], importance: list[float]) -> bytes:
    """One JPEG grid of low-res thumbnails (one per asset, its most important window), asset id on each tile.
    Sent only when director.send_thumbnails is on; never full-resolution media."""
    import io

    from PIL import Image, ImageDraw

    from montage.config import get_config
    cfg = get_config().director.contact_sheet
    best: dict[str, tuple[float, str]] = {}
    for c, imp in zip(candidates, importance):
        if c.thumb_path and (c.asset_id not in best or imp > best[c.asset_id][0]):
            best[c.asset_id] = (imp, c.thumb_path)
    items = [(a.id, best[a.id][1]) for a in assets if a.id in best][: cfg.max_cells]
    cols = max(1, min(10, len(items)))
    rows = max(1, -(-len(items) // cols))
    cell = cfg.cell_px
    sheet = Image.new("RGB", (cols * cell, rows * cell), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    for k, (aid, path) in enumerate(items):
        x, y = (k % cols) * cell, (k // cols) * cell
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((cell - 2, cell - 2))
                sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        except OSError:
            pass
        d.rectangle([x + 2, y + 2, x + 40, y + 16], fill=(0, 0, 0))
        d.text((x + 4, y + 3), aid, fill=(255, 255, 0))
    for q in (80, 65, 50):
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=q)
        if buf.tell() < 1_500_000:
            break
    return buf.getvalue()
